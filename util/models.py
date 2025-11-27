from typing import Callable, Self, Literal
from sklearn.exceptions import NotFittedError
import pandas as pd
import numpy as np
from scipy.sparse import spmatrix
from pydantic import validate_call

class FeedForwardNeuralNetwork():
    @validate_call
    def __init__(self, sizes_of_hidden_layers: list[int], epochs: int, learning_rate: float, batch_size: int = 0,regression: bool = False, 
                 hidden_activation_func: None | Literal["relu"] | tuple[Literal["parametric_relu", "elu"], float] = None, 
                 output_activation_func: None | Literal["sigmoid", "softmax", "linear"] = None, 
                 regularization_setting: None | tuple[int, float] = None,
                 random_state: None | int = None, verbose: bool = False):
        """
        Initialize the Feed Forward Neural Network.

        Parameters
        ----------
        sizes_of_hidden_layers : tuple[int]
            A tuple where each integer represents the number of neurons in a specific hidden layer. 
            For example, [10, 5] creates two hidden layers with 10 and 5 neurons respectively.
        epochs : int
            The number of complete passes through the training dataset.
        learning_rate : float
            The step size (alpha) used for parameter updates during gradient descent.
        batch_size : float, default=0
            Number of samples in training batches, if 0 takes the whole dataset. 
        regression : bool, default=False
            Determines the estimator type.
            - If True: Configures the network for regression tasks.
            - If False: Configures the network for classification tasks.
        hidden_activation_func : None | str | tuple, default=None
            The activation function applied to hidden layers.
            - If None: Automatically selected based on task type.
            - If str: Can be "relu".
            - If tuple: Format (name, alpha) for parameterized functions, 
              e.g., ("parametric_relu", 0.01) or ("elu", 1.0).
        output_activation_func : None | str, default=None
            The activation function applied to the output layer.
            Options include "sigmoid", "softmax", or "linear".
            If None, it is automatically selected based on the task type.
        regularization_setting : None | tuple[int, float], default=None
            Configuration for regularization.
            Format: (order, lambda).
            - order (int): 1 for L1 regularization, 2 for L2 regularization.
            - lambda (float): The regularization strength.
        random_state : None | int, default=None
            Seed for the random number generator to ensure reproducibility of weight initialization.
        verbose : bool, default=False
            If True, prints loss metrics every 100 epochs during training.
        """
        self.size_of_hidden_layers = sizes_of_hidden_layers
        self.epochs = epochs
        self.alpha = learning_rate
        self.seed = random_state
        self.verbose = verbose
        self.regularization_setting = regularization_setting
        self._estimator_type = "classifier" if not regression else "regressor"
        self.batch_size = batch_size
        if isinstance(hidden_activation_func, tuple):
            self.hidden_activation_name = hidden_activation_func[0]  
            self.hidden_activation_args = hidden_activation_func[1:]
        else: 
            self.hidden_activation_name = hidden_activation_func
            self.hidden_activation_args = (0,)
        self.output_activation_name = output_activation_func        
        self.is_fitted_: bool = False
        if self.seed:
            np.random.seed(self.seed)
        self.n_layers = len(sizes_of_hidden_layers)+2

    def _get_activation_funcs(self) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
        
        def relu(input: np.ndarray) -> np.ndarray:
            return np.where(input < 0, 0, input)

        def sigmoid(input: np.ndarray) -> np.ndarray:
            return 1/(1+np.exp(-input))
        
        def parametric_relu(input: np.ndarray, a: float) -> np.ndarray:
            return np.maximum(a*input, input)
        
        def elu(input: np.ndarray, a: float) -> np.ndarray:
            return np.where(input <= 0, a*(np.exp(input)-1), input)
        
        def softmax(input: np.ndarray) -> np.ndarray: 
            return np.exp(input)/np.exp(input).sum(axis=1, keepdims=True)
        
        def linear(input: np.ndarray) -> np.ndarray:
            return input
        
        if not (self.hidden_activation_name and self.output_activation_name):
            method_to_activation: dict[str, tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]] = {
                "Binary Classification": (relu, sigmoid),
                "Multiclass Classification": (relu, softmax),
                "Regression": (relu, linear)
            }

            self.hidden_activation_name, self.output_activation_name = map(getattr, method_to_activation[self.method], ["__name__"]*2)
            return method_to_activation[self.method]
        else:
                activation_name_to_activation_func: dict[str, Callable[[np.ndarray], np.ndarray]] = {
                    "relu": relu,
                    "sigmoid": sigmoid,
                    "parametric_relu": lambda input: parametric_relu(input, *self.hidden_activation_args),
                    "elu": lambda input: elu(input, *self.hidden_activation_args),
                    "softmax": softmax,
                    "linear": lambda input: input
                }
                if self.hidden_activation_name in activation_name_to_activation_func and self.output_activation_name in activation_name_to_activation_func:
                    return activation_name_to_activation_func[self.hidden_activation_name], activation_name_to_activation_func[self.output_activation_name]
                else:
                    raise NotImplementedError(f"Wanted activation function is not in implemented functions {list(activation_name_to_activation_func.keys())}.")
        
    def _get_error_func(self) -> Callable[[np.ndarray, np.ndarray], float]:
        def BinaryCrossEntropy(y_real: np.ndarray, output: np.ndarray) -> float:
            return -np.mean(y_real * np.log(output + 1e-15) + (1 - y_real) * np.log(1 - output + 1e-15))
        
        def MSE(y_real: np.ndarray, output: np.ndarray) -> float:
            return np.mean(0.5*(output-y_real)**2)
        
        def CrossEntropy(y_real: np.ndarray, output: np.ndarray) -> float:
            return -np.mean(np.sum(y_real * np.log(output + 1e-15), axis=1))
            
        method_to_error: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
            "Binary Classification": BinaryCrossEntropy,
            "Multiclass Classification": CrossEntropy,
            "Regression": MSE
        }

        return method_to_error[self.method]

    def _determine_method(self, y: np.ndarray) -> str:
        if self._estimator_type == "classifier":
            if len(self.classes_) == 2:
                return "Binary Classification"
            else:
                return "Multiclass Classification"
        else:
            return "Regression"
        
    def _get_outup_node_count(self, y: np.ndarray) -> int:
        method_to_count: dict[str, int] = {
            "Binary Classification": 1,
            "Multiclass Classification": y.shape[1],
            "Regression": 1
        }
        return method_to_count[self.method]
            
    def _createParameterArray(self, n: int | tuple, mean: float, std: float) -> np.ndarray:
        return np.random.normal(mean, std, size=n)*0.01 # to help with overflows

    def _forwardPass(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        layer_outputs = list()
        weighted_sums = list()
        
        n_hidden_layers = self.n_layers - 2 # input and output
        input_array = x
        layer_outputs.append(x)
        weighted_sums.append(x)

        for i in range(n_hidden_layers):
            w = self.layer_weights[i]
            b = self.layer_biases[i]
            weighted_sum = input_array @ w + b
            input_array = self.activation_func_layers(weighted_sum)
            weighted_sums.append(weighted_sum)
            layer_outputs.append(input_array)
        
        output_weighted_sum = input_array @ self.layer_weights[-1] + self.layer_biases[-1]
        output = self.activation_func_output(output_weighted_sum)
        weighted_sums.append(output_weighted_sum)
        layer_outputs.append(output)
        return weighted_sums, layer_outputs
        
    def _gradients(self, y: np.ndarray, weighted_sums: list[np.ndarray], layer_outputs: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
        grads_w: list[np.ndarray] = [np.zeros_like(w) for w in self.layer_weights]
        grads_b: list[np.ndarray] = [np.zeros_like(b) for b in self.layer_biases]

        def get_gradient_funcs_of_activations() -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray, np.ndarray], np.ndarray]]:
            relu: Callable[[np.ndarray], np.ndarray] = lambda w: np.where(w <= 0, 0, 1)
            parametric_relu: Callable[[np.ndarray], np.ndarray] = lambda w: np.where(w <= 0, self.hidden_activation_args[0], 1)
            elu: Callable[[np.ndarray], np.ndarray] = lambda w: np.where(w <= 0, self.hidden_activation_args[0]*np.exp(w), 1)

            sigmoid: Callable[[np.ndarray], np.ndarray] = lambda a_L: a_L*(1-a_L)
            linear: Callable[[np.ndarray], np.ndarray] = lambda a_L: a_L

            if self.output_activation_name in {"sigmoid", "softmax", "linear"} and self.method in {"Binary Classification", "Multiclass Classification", "Regression"}:
                error_func = lambda a_L, y: a_L - y
            elif self.output_activation_name in {"relu", "parametric_relu", "elu"} and self.method == "Regression":
                error_func = lambda a_L, y: np.where(a_L < 0, self.hidden_activation_args[0]*(a_L-y), a_L-y)
            else:
                raise NotImplementedError(f"Error function for output activation function {self.output_activation_name} and method {self.method} is not implemented.")

            activation_name_to_gradient = {
                "relu": relu,
                "sigmoid": sigmoid,
                "parametric_relu": parametric_relu,
                "elu": elu,
                "linear": linear,
            }
            try:
                return activation_name_to_gradient[self.hidden_activation_name], error_func
            except KeyError:
                raise NotImplementedError(f"Gradient function for {self.hidden_activation_name} is not implemented.")

        grad_func_hidden, error_term = get_gradient_funcs_of_activations()
        
        for i in range(1, self.n_layers):
            a_L_minus_one = layer_outputs[-i-1]
            a_L = layer_outputs[-i]

            if i == 1:
                
                dError = error_term(a_L, y)

                dw = a_L_minus_one.T @ dError
                db = np.sum(dError, axis = 0, keepdims=True) # sum errors across batch
                
                # print(dw.shape) # (4,1) 4 nodes in last hidden layer, 1 output node
                # print(a_L_minus_one.T.shape) # (4, 500)
                # print(db.shape) # (1,1)

            else:
                dHidden_Layer_Activation = grad_func_hidden(weighted_sums[-i])

                # print(weighted_sums[-i+1].T.shape) # (10, 1764)
                # print(dError.shape) # (1764,10)

                dError =  (dError @ self.layer_weights[-i+1].T) * dHidden_Layer_Activation
            
                # print(self.layer_weights[-i+1].shape) # (4, 1)
                # print(dReLU.shape) # (500,4)
                # print(dError.shape) # (500,4)

                dw = a_L_minus_one.T @ dError
                db = np.sum(dError, axis = 0, keepdims=True)

                # print(db.shape) # (1,4)
            
            grads_w[-i] = dw
            grads_b[-i] = db
        
        return grads_w, grads_b

    def _update_params(self, grads_w: list[np.ndarray], grads_b: list[np.ndarray], alpha: float) -> tuple[list[np.ndarray], list[np.ndarray]]:
        assert len(self.layer_weights) == len(grads_w)
        assert len(self.layer_biases) == len(grads_b)

        reg_order = None
        reg_lambda = 0.0

        if self.regularization_setting:
            reg_order, reg_lambda = self.regularization_setting

        for i in range(len(grads_w)):
            reg_term = np.zeros_like(self.layer_weights[i])
            
            if reg_order == 2:
                reg_term = reg_lambda * self.layer_weights[i]
            elif reg_order == 1:
                reg_term = reg_lambda * np.sign(self.layer_weights[i])
                
            self.layer_weights[i] -= alpha * (grads_w[i] + reg_term)
            self.layer_biases[i] -= alpha*grads_b[i]
        
        return self.layer_weights, self.layer_biases
    
    @staticmethod
    def _check_input(input: pd.DataFrame | np.ndarray| spmatrix) -> np.ndarray:
        if isinstance(input, pd.DataFrame) or isinstance(input, pd.Series):
            corrected = input.to_numpy(copy=True, dtype=float)
        elif isinstance(input, np.ndarray):
            return input
        elif isinstance(input, spmatrix):
            return input.toarray()
        else:
            raise TypeError(f"X is unsupported type {type(input).__name__}, must be np.ndarray, pd.DataFrame or spmatrix")
        return corrected

    def fit(self, x: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> Self:
        """
        Fit the neural network to the training data.

        Parameters
        ----------
        x : np.ndarray | pd.DataFrame
            Input data matrix of shape (n_samples, n_features).
        y : np.ndarray | pd.Series
            Target values.
            - For Binary Classification: Shape (n_samples,) or (n_samples, 1).
            - For Multiclass Classification: One-hot encoded shape (n_samples, n_classes).
            - For Regression: Shape (n_samples,) or (n_samples, 1).

        Returns
        -------
        self : FeedForwardNeuralNetwork
            Returns the fitted instance of the model.
        """
        x = self._check_input(x)
        y = self._check_input(y)

        if len(y.shape) == 1:
            y = y.reshape(y.shape[0], 1)

        if y.shape[1] > y.shape[0]:
            y = y.T

        if y.shape[1] == 1 and self._estimator_type == "classifier":
            # for sklearn integration
            self.classes_ = np.unique(y)
        elif y.shape[1] > 1 and self._estimator_type == "classifier":
            self.classes_ = np.array(list(range(y.shape[1])))

        self.method = self._determine_method(y)

        self.activation_func_layers, self.activation_func_output = self._get_activation_funcs()
        self.error_func: Callable[[np.ndarray, np.ndarray], float] = self._get_error_func()

        self.size_of_layers: list[int] = [x.shape[1]] + self.size_of_hidden_layers + [self._get_outup_node_count(y)]

        self.layer_weights: list[np.ndarray] = [self._createParameterArray((self.size_of_layers[i], self.size_of_layers[i+1]), 0, 1) for i in range(0, self.n_layers-1)]
        self.layer_biases: list[np.ndarray] = [self._createParameterArray((1, self.size_of_layers[i+1]), 0, 1) for i in range(0, self.n_layers-1)]

        if self.batch_size != 0:
                if self.seed:
                    np.random.seed(self.seed)
                data_index = np.array(range(x.shape[0]))[:x.shape[0]-(x.shape[0] % self.batch_size)]
                np.random.shuffle(data_index)
                data_index = data_index.reshape(-1, self.batch_size)

        for epoch in range(self.epochs):
            if self.batch_size != 0:
                for indicies in data_index:
                    weighted_sums, layer_outputs = self._forwardPass(x[indicies, :])
                    grads_w, grads_b = self._gradients(y[indicies, :], weighted_sums, layer_outputs)
                    self.layer_weights, self.layer_biases = self._update_params(grads_w, grads_b, self.alpha)
            else:
                weighted_sums, layer_outputs = self._forwardPass(x)
                grads_w, grads_b = self._gradients(y, weighted_sums, layer_outputs)
                self.layer_weights, self.layer_biases = self._update_params(grads_w, grads_b, self.alpha)

            if epoch % 1 == 0 and self.verbose:
                weighted_sums, layer_outputs = self._forwardPass(x)
                final_output = layer_outputs[-1]
                loss = self._get_error_func()(y, final_output)
                print(f"Epoch {epoch}, Loss: {loss:.6f}")

        self.is_fitted_ = True
        return self
    
    def predict_proba(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for input samples.
        
        This method is only available for classification tasks.

        Parameters
        ----------
        x : np.ndarray | pd.DataFrame
            Input data matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Class probabilities.
            - Binary Classification: Returns shape (n_samples, 2), where column 0 
              is the probability of class 0 and column 1 is the probability of class 1.
            - Multiclass Classification: Returns shape (n_samples, n_classes).

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        ValueError
            If the model was initialized for Regression.
        """
        if not self.is_fitted_:
            raise NotFittedError("This model has not been fitted yet. Call 'fit' before predicting.")
        if self.method == "Regression":
            raise ValueError("Method predict_proba not supported for regression.")
        
        x = self._check_input(x)
        
        _, layer_outputs = self._forwardPass(x)

        pred_proba = layer_outputs[-1]
        # for sklearn integration
        if pred_proba.shape[1] == 1 and self.method == "Binary Classification":
            pred_proba = np.column_stack([1-pred_proba, pred_proba])
        
        return pred_proba
    
    def predict(self, x: np.ndarray | pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels or regression values for input samples.

        Parameters
        ----------
        x : np.ndarray | pd.DataFrame
            Input data matrix of shape (n_samples, n_features).
        threshold : float, default=0.5
            The threshold used for converting probabilities to class labels 
            in Binary Classification. Ignored for Multiclass or Regression.

        Returns
        -------
        np.ndarray
            Predicted targets.
            - Binary Classification: 0 or 1 labels.
            - Multiclass Classification: Index of the class with the highest probability.
            - Regression: Continuous values.
        """
        if self.method == "Binary Classification":
            return np.where(self.predict_proba(x)[:,1] < threshold, 0, 1)
        elif self.method == "Multiclass Classification":
            return np.argmax(self.predict_proba(x), axis=1)
        elif self.method == "Regression":
            x = self._check_input(x)
            output = self._forwardPass(x)[1][-1].flatten()
            return output
    
# Your solution goes here
from sklearn.exceptions import NotFittedError
from typing import Self

class Node():
    def __init__(self, feature: str | None = None, split_val: float | None = None):
        self.feature = feature
        self.split_value = split_val
        self.label = None
        self.left_child: Node | None = None
        self.right_child: Node | None = None
        self.is_leaf: bool = True
        self.n_samples: int = 0
    
class MyDecisionTreeClassifier():
    def __init__(self, 
                  criterion: Literal['gini', 'entropy'] = "gini",
                  max_depth: int | None = None, 
                  min_samples_split: float = 2, 
                  min_samples_leaf: float = 1, 
                  max_leaf_nodes: int | None = None, 
                  min_impurity_decrease: float = 0):
        """Implements Decision Tree Classifier based on Gini or Entropy impurity metrics.

        Parameters
        ----------
        criterion : Literal[&#39;gini&#39;, &#39;entropy&#39;], optional
            Impurity metric to use, by default "gini"
        max_depth : int | None, optional
            Maxmimal depth of tree, by default None
        min_samples_split : float, optional
            Minimum samples in a node to be able to split the node, by default 2
        min_samples_leaf : float, optional
            Minimum samples in a leaf node, by default 1
        max_leaf_nodes : int | None, optional
            Maximum number of leaf nodes in tree, by default None
        min_impurity_decrease : float, optional
            Minimum impurity decrease to continue splitting, by default 0
        """
        self._is_fitted: bool = False
        self._tree: Node | None = Node()
        self._n_features_in: int = None
        self._criterion = criterion
        self._max_depth = max_depth if isinstance(max_depth, int) else np.inf
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._max_leaf_nodes = max_leaf_nodes if isinstance(max_leaf_nodes, int) else np.inf
        self._min_impurity_decrease = min_impurity_decrease
        self._label_type = None
        self._n_leaves = 0
    
    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> Self:
        X = self.__correct_input__(X)
        
        self._label_type = y.dtype
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        self._tree = self._construct_tree(X, y, 0)
        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self._tree:
            raise NotFittedError("The model has not been fitted yet. Use fit method first.")
        X = self.__correct_input__(X)
    
        return np.array([self._predict(row, self._tree) for _, row in X.iterrows()])
    
    def prune(self, X: pd.DataFrame, y: np.ndarray, node: Node) -> tuple[float, Node]:
        ...

    def _prune(self, X: pd.DataFrame, y: np.ndarray, node: Node) -> tuple[float, Node]:
        ...

    def _predict(self, X: pd.Series, node: Node):
        if node.is_leaf:
            return node.label
        else:
            label = self._predict(X, node.left_child) if X[node.feature] < node.split_value else self._predict(X, node.right_child)
            return label
            
    def _construct_tree(self, X: pd.DataFrame, y: np.ndarray, depth: int):
        def create_leaf_node(X: pd.DataFrame, y: np.ndarray):
            leaf_node = Node()
            n = len(y)
            classes, counts = np.unique(y, return_counts=True)
            class_props = np.array([c/n for c in counts])
            leaf_node.label = classes[np.argmax(class_props)]
            leaf_node.n_samples = len(X)
            return leaf_node

        if len(X) < self._min_samples_split or \
            depth > self._max_depth or \
            len(np.unique(y)) == 1:
            return create_leaf_node(X,y)

        split_info = self._find_split(X, y)
        
        if split_info is None:
            return create_leaf_node(X,y)
        best_feature, best_split_val = split_info

        left_mask = X[best_feature] < best_split_val
        right_mask = X[best_feature] >= best_split_val

        self._n_leaves -= 1 # I am not a leaf node since I split
        self._n_leaves += 2 # My children are potentially leaves

        internal_node = Node(best_feature, best_split_val)
        internal_node.left_child = self._construct_tree(X.loc[left_mask, :], y[left_mask], depth + 1)
        internal_node.right_child = self._construct_tree(X.loc[right_mask, :], y[right_mask], depth + 1)
        internal_node.is_leaf = False
        internal_node.n_samples = len(X)

        return internal_node

    def _find_split(self, X_part: pd.DataFrame, y_part: np.ndarray) -> tuple[str, float]:
        best_impurity = np.inf
        best_feature = None
        best_split_value = None

        parent_impurity = impurity(y_part, self._criterion)

        features = list(X_part.columns)
        np.random.shuffle(features)
        
        for feature in features:
            local_X = X_part[feature]
            pos_spllits = np.sort(np.unique(local_X))
            if len(pos_spllits) == 1:
                continue
            for i, val_1 in enumerate(pos_spllits):
                if i+1 == len(pos_spllits):
                    break
                val_2 = pos_spllits[i+1]
                val = (val_1+val_2)/2
                regions = (y_part[local_X < val], y_part[local_X >= val])
                imp = weighted_impurity(regions, self._criterion)
                if imp < best_impurity:
                    best_impurity = imp
                    best_feature = feature
                    best_split_value = val
                    best_region = regions
        if  parent_impurity - best_impurity < self._min_impurity_decrease:
            return None
        return best_feature, best_split_value 
    
    def __correct_input__(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame) and isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        elif not isinstance(X, pd.DataFrame):
            raise TypeError(f"X is unsupported type {type(X).__name__}, has to be pd.DataFrame or np.ndarray or pd.Series.")
        return X

    def __sklearn_is_fitted__(self):
        return self._is_fitted