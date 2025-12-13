import os
from typing import Callable, Self, Literal, Any
from sklearn.exceptions import NotFittedError
import numpy as np
import pandas as pd
from util.general import check_input
from pydantic import validate_call
import time
import json
import shutil

def print_progress(self, i: int, max_i: int, interval: int) -> None:
        def format_time(seconds: float) -> str:
            minutes, sec = divmod(seconds, 60)
            hours, minutes = divmod(minutes, 60)
            if int(hours) > 0:
                return f"{str(int(hours)).rjust(2,'0')}h {str(int(minutes)).rjust(2,'0')}m {str(int(sec)).rjust(2,'0')}s"
            elif int(minutes) > 0:
                return f"{str(int(minutes)).rjust(2,'0')}m {str(int(sec)).rjust(2,'0')}s"
            else:
                return f"{str(int(sec)).rjust(2,'0')}s"
            
        if i % interval == 0:
            current_time = time.time()
            time_per_action = (current_time-self.start_time)/(i+1)
            remaining_time = time_per_action*(max_i-i+1)
            print(f"{i+1}/{max_i} epochs, Validation loss: {self.val_loss_[-1]:.6f}, Training loss {self.train_loss_[-1]:.6f}. Est. time remaining: {format_time(remaining_time):>11}",end="\r")

class FeedForwardNeuralNetwork():
    @validate_call
    def __init__(self, sizes_of_hidden_layers: list[int], epochs: int, learning_rate: float, batch_size: int = 0, regression: bool = False, optimizer: Literal["sgd", "adam"] = "adam",
                 hidden_activation_func: None | Literal["relu"] | tuple[Literal["parametric_relu", "elu"], float] = None, 
                 output_activation_func: None | Literal["sigmoid", "softmax", "linear"] = None, 
                 regularization_setting: None | tuple[int, float] = None, patience: int = 0,
                 random_state: None | int = None, verbose: bool = False, auto_save: tuple[bool, str, bool] | bool = (True, "ffnn_autosave", True)) -> None:
        """
        Initialize the Feed Forward Neural Network.

        Parameters
        ----------
        sizes_of_hidden_layers : list[int]
            A list where each integer represents the number of neurons in a specific hidden layer. 
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
        optimizer : Literal["sgd", "adam"], default="adam"
            The optimization algorithm used for training.
            - "sgd": Stochastic Gradient Descent.
            - "adam": Adam optimizer.
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
                * In Adam optimizer, L2 regularization is implemented as decoupled weight decay.
            - lambda (float): The regularization strength.
        patience : int, default=0
            Number of epochs with no improvement after which training will be stopped.
            If 0, will not stop early.
        random_state : None | int, default=None
            Seed for the random number generator to ensure reproducibility of weight initialization.
        verbose : bool, default=False
            If True, prints loss metrics every 100 epochs during training.
        auto_save : tuple[bool, str, bool] | bool, default=True
            If a boolean True is provided, it is interpreted as (True, "ffnn_autosave", True), if boolean False is interpreted as (False, "", True).
            Configuration for automatic model saving.
            - First element (bool): If True, the model is saved automatically during and after training.
            - Second element (str): Prefix for the saved model files.
            - Third element (bool): If True, the temporary files created during training are deleted after training finishes successfully.
        """
        self.sizes_of_hidden_layers = sizes_of_hidden_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.verbose = verbose
        self.optimizer = optimizer
        self.regularization_setting = regularization_setting
        self.batch_size = batch_size

        self.patience = patience
        self.regression = regression
        self.auto_save = auto_save
        self.hidden_activation_func = hidden_activation_func
        self.output_activation_func = output_activation_func
        
        

    def save(self, filename_prefix: str) -> None:
        """
        Save the trained neural network state to disk.

        This method serializes the object's attributes into two separate files:
        a JSON file for configuration/hyperparameters and a compressed NumPy
        archive (.npz) for weights, biases, and other array-like data. It
        automatically handles the flattening of list attributes containing
        NumPy arrays (specifically `layer_weights` and `layer_biases`).

        Parameters
        ----------
        filename_prefix : str
            The base path and filename for the saved files. For example,
            providing 'model' will generate 'model_config.json' and
            'model_weights.npz'.

        Returns
        -------
        None
        """
        all_attributes = self.__dict__
        
        config = {}
        arrays = {}
        
        for key, value in all_attributes.items():
            if isinstance(value, list) and all(isinstance(i, np.ndarray) for i in value):
                if key == "layer_weights":
                    for idx, arr in enumerate(value):
                        arrays[f'layer_weights_{idx}'] = arr
                elif key == "layer_biases":
                    for idx, arr in enumerate(value):
                        arrays[f'layer_biases_{idx}'] = arr
            elif isinstance(value, (list, np.ndarray)) and all(isinstance(i, (float)) for i in value):
                if isinstance(value, list):
                    arrays[key] = np.array(value)
                else:
                    arrays[key] = value
            elif isinstance(value, (int, float, str, list, dict, bool, tuple, type(None))):
                config[key] = value
            elif not isinstance(value, type(lambda: None)):
                print(f"Warning: Skipping attribute '{key}' of type {type(value)}")

        with open(f'{filename_prefix}_config.json', 'w') as f:
            json.dump(config, f, indent=4)
            
        np.savez(f'{filename_prefix}_weights.npz', **arrays)
        if self.verbose:
            print("\nSaved automatically.")

    @classmethod
    def load(cls, filename_prefix: str) -> Self:
        """
        Load a trained neural network state from disk.

        This class method reads the configuration JSON to instantiate the
        network with the correct architecture and hyperparameters, then
        loads the weights and biases from the NumPy archive into the
        instance. It also reconstructs necessary internal helper methods
        (activation and error functions).

        Parameters
        ----------
        filename_prefix : str
            The base path and filename used during saving.

        Returns
        -------
        NeuralNetwork
            An instance of the class with restored state, weights, and
            configuration.
        
        Raises
        ------
        FileNotFoundError
            If the configuration file is not found.
        KeyError
            If required keys are missing in the configuration file.
        """
        
        for exit_type in ["_final", "_interrupted"]:
            try:
                with open(f'{filename_prefix}{exit_type}_config.json', 'r') as f:
                    filename_prefix = f'{filename_prefix}{exit_type}'
                    break
            except FileNotFoundError:
                continue
        try:
            with open(f'{filename_prefix}_config.json', 'r') as f:
                config: dict[str, Any] = dict(json.load(f))
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file '{filename_prefix}_config.json' not found.")
        
        try:
            network = cls(config["size_of_hidden_layers"], config["epochs"], config["alpha"])
        except KeyError as e:
            raise KeyError(f"Missing required positional argument in configuration file: {e}")

        for key, value in config.items():
            setattr(network, key, value)
        
        network.activation_func_layers, network.activation_func_output = network._get_activation_funcs()
        network.error_func = network._get_error_func()

        data: dict[str, np.ndarray] = dict(np.load(f'{filename_prefix}_weights.npz'))

        network.layer_weights = [np.array([]) for i in range(0, network.n_layers-1)]
        network.layer_biases = [np.array([]) for i in range(0, network.n_layers-1)]
        
        for key, value in data.items():
            if key.startswith('layer_weights_'):
                idx = int(key.split('_')[-1])
                network.layer_weights[idx] = value
            elif key.startswith('layer_biases_'):
                idx = int(key.split('_')[-1])
                network.layer_biases[idx] = value
            else:
                setattr(network, key, value)
        
        if network.verbose:
            print("Model loaded successfully.")
        return network

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
                    "linear": linear
                }
                if self.hidden_activation_name in activation_name_to_activation_func and self.output_activation_name in activation_name_to_activation_func:
                    return activation_name_to_activation_func[self.hidden_activation_name], activation_name_to_activation_func[self.output_activation_name]
                else:
                    raise NotImplementedError(f"Wanted activation function is not in implemented functions {list(activation_name_to_activation_func.keys())}.")
        
    def _get_error_func(self) -> Callable[[np.ndarray, np.ndarray], float]:
        def BinaryCrossEntropy(y_real: np.ndarray, output: np.ndarray) -> float:
            return -np.mean(y_real * np.log(output + 1e-15) + (1 - y_real) * np.log(1 - output + 1e-15))
        
        def MSE(y_real: np.ndarray, output: np.ndarray) -> float:
            return np.mean((output-y_real)**2)
        
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
            
    def _createParameterArray(self, n: tuple, mean: float, std: float) -> np.ndarray:
        n_in = n[0]
        return np.random.normal(mean, np.sqrt(2/n_in), size=n) # to help with overflows

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

            sigmoid: Callable[[np.ndarray], np.ndarray] = lambda z: (s := 1/(1+np.exp(-z)))*(1-s)
            linear: Callable[[np.ndarray], np.ndarray] = lambda z: 1

            if self.output_activation_name in {"sigmoid", "softmax"} and self.method in {"Binary Classification", "Multiclass Classification", "Regression"}:
                error_func = lambda a_L, y: a_L - y
            elif self.output_activation_name in {"linear"} and self.method == "Regression":
                error_func = lambda a_L, y: 2*(a_L - y)
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

        batch_n = y.shape[0] 
        
        for i in range(1, self.n_layers):
            a_L_minus_one = layer_outputs[-i-1]
            a_L = layer_outputs[-i]

            if i == 1:
                
                dError = error_term(a_L, y)

                dw = a_L_minus_one.T @ dError / batch_n
                db = np.sum(dError, axis = 0, keepdims=True) / batch_n
                
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

                dw = a_L_minus_one.T @ dError / batch_n
                db = np.sum(dError, axis = 0, keepdims=True) / batch_n

                # print(db.shape) # (1,4)
            
            grads_w[-i] = dw
            grads_b[-i] = db
        
        return grads_w, grads_b
    
    def _regularization_term(self, param: np.ndarray) -> np.ndarray:
            reg_order = None
            reg_lambda = 0.0

            if self.regularization_setting:
                reg_order, reg_lambda = self.regularization_setting

            if reg_order == 2:
                return reg_lambda * param
            elif reg_order == 1:
                return reg_lambda * np.sign(param)
            else:
                return np.zeros_like(param)

    def _update_params(self, grads_w: list[np.ndarray], grads_b: list[np.ndarray], alpha: float, processed_batches: int) -> None:
        assert len(self.layer_weights) == len(grads_w)
        assert len(self.layer_biases) == len(grads_b)
        
        # Stochastic Gradient Descent update
        def sgd_update() -> None:
            for i in range(len(grads_w)):
                reg_term = self._regularization_term(self.layer_weights[i])
                self.layer_weights[i] -= alpha * (grads_w[i] + reg_term)
                self.layer_biases[i] -= alpha*grads_b[i]
            
        def momentum_update(grad: np.ndarray, m_t_minus_one: np.ndarray, beta: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
            m_t = beta * m_t_minus_one + (1 - beta) * grad
            return m_t
        
        def RMSprop_update(grad: np.ndarray, v_t_minus_one: np.ndarray, beta: float = 0.999, epsilon: float = 1e-15) -> tuple[np.ndarray, np.ndarray]:
            v_t = beta * v_t_minus_one + (1 - beta) * (grad ** 2)
            return v_t
        
        def adamW_update(processed_batches: int) -> None:
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-6
            
            for i in range(len(grads_w)):
                reg_term = self._regularization_term(self.layer_weights[i])

                self.m_w[i] = momentum_update(grads_w[i], self.m_w[i], beta1)
                self.v_w[i] = RMSprop_update(grads_w[i], self.v_w[i], beta2, epsilon)
                
                self.m_b[i] = momentum_update(grads_b[i], self.m_b[i], beta1)
                self.v_b[i] = RMSprop_update(grads_b[i], self.v_b[i], beta2, epsilon)
                m_w_corrected = self.m_w[i] / (1 - beta1 ** (processed_batches))
                v_w_corrected = self.v_w[i] / (1 - beta2 ** (processed_batches))
                m_b_corrected = self.m_b[i] / (1 - beta1 ** (processed_batches))
                v_b_corrected = self.v_b[i] / (1 - beta2 ** (processed_batches))

                self.layer_weights[i] -= self.learning_rate * (m_w_corrected / (np.sqrt(v_w_corrected) + epsilon) + reg_term) # decoupled weight decay
                self.layer_biases[i] -= self.learning_rate * m_b_corrected / (np.sqrt(v_b_corrected) + epsilon)

        if self.optimizer == "sgd":
            sgd_update()
        elif self.optimizer == "adam":
            adamW_update(processed_batches)
    
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
        self.is_fitted_: bool = False
        self.patience = self.patience if self.patience else self.epochs
        self._estimator_type = "classifier" if not self.regression else "regressor"
        self.save_auto = self.auto_save[0] if isinstance(self.auto_save, tuple) else self.auto_save
        self.save_prefix = self.auto_save[1] if isinstance(self.auto_save, tuple) else "ffnn_autosave"
        self.delete_temp_after_success = self.auto_save[2] if isinstance(self.auto_save, tuple) else True
        if isinstance(self.hidden_activation_func, tuple):
            self.hidden_activation_name = self.hidden_activation_func[0]  
            self.hidden_activation_args = self.hidden_activation_func[1:]
        else: 
            self.hidden_activation_name = self.hidden_activation_func
            self.hidden_activation_args = (0,)
        self.output_activation_name = self.output_activation_func
        if self.random_state:
            np.random.seed(self.random_state)
        self.n_layers = len(self.sizes_of_hidden_layers)+2

        x = check_input(x)
        y = check_input(y)

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

        self.size_of_layers: list[int] = [x.shape[1]] + self.sizes_of_hidden_layers + [self._get_outup_node_count(y)]

        validation_idx = np.random.choice(x.shape[0], size=int(0.1*x.shape[0]), replace=False)
        x_val = x[validation_idx, :]
        y_val = y[validation_idx, :]

        x = np.delete(x, validation_idx, axis=0)
        y = np.delete(y, validation_idx, axis=0)

        best_loss = float('inf')

        for _ in range(15): # get good starting parameters
            self.layer_weights: list[np.ndarray] = [self._createParameterArray((self.size_of_layers[i], self.size_of_layers[i+1]), 0, 1) for i in range(0, self.n_layers-1)]
            self.layer_biases: list[np.ndarray] = [self._createParameterArray((1, self.size_of_layers[i+1]), 0, 1) for i in range(0, self.n_layers-1)]

            loss = self.error_func(y, self._forwardPass(x)[1][-1])
            if loss < best_loss:
                best_loss = loss
                best_weights = self.layer_weights
                best_biases = self.layer_biases
        
        self.layer_weights = best_weights
        self.layer_biases = best_biases

        worse_counter = 0
        data_index = np.arange(x.shape[0])
        self.train_loss_: list[float] = list()
        self.val_loss_: list[float] = list()

        if self.optimizer == "adam":
            self.m_w: list[np.ndarray] = [np.zeros_like(w) for w in self.layer_weights]
            self.v_w: list[np.ndarray] = [np.zeros_like(w) for w in self.layer_weights]
            self.m_b: list[np.ndarray] = [np.zeros_like(b) for b in self.layer_biases]
            self.v_b: list[np.ndarray] = [np.zeros_like(b) for b in self.layer_biases]


        def exit_handler(best_weights: list[np.ndarray], best_biases: list[np.ndarray], best_epoch: int, exit_type: str = "normal") -> None:
            self.layer_weights = best_weights
            self.layer_biases = best_biases
            epoch = self.current_epoch
            val_loss = self.val_loss_[best_epoch]
            train_loss = self.train_loss_[best_epoch]

            if exit_type == "normal":
                self.training_epochs = epoch + 1
                self.time_taken_ = time.time() - self.start_time
                if self.verbose:
                    print(f"\nTraining stopped at epoch {epoch+1} with best validation loss {val_loss:.6f} and best training loss {train_loss:.6f} from epoch {best_epoch}.")
                if self.save_auto:
                    os.makedirs("model_saves", exist_ok=True)
                    self.save(f"model_saves/{self.save_prefix}_val_loss_{str(round(val_loss, 6)).split('.')[1]}_final")
                    if self.is_fitted_ and self.delete_temp_after_success:
                        shutil.rmtree(temp_folder, ignore_errors=True)
                        if self.verbose:
                            print("Temporary files deleted after successful training.")
            elif exit_type == "interrupt":
                if self.save_auto:
                    os.makedirs("model_saves", exist_ok=True)
                    self.save(f"model_saves/{self.save_prefix}_val_loss_{str(round(loss, 6)).split('.')[1]}_interrupted")
                    if self.verbose:
                        print(f"Training interrupted at epoch {epoch}. Model saved.")
            self.is_fitted_ = True
            self.exit_type_ = exit_type
        
        temp_folder = f"temp/ffnn_{np.random.randint(0, 1_000_000)}"

        processed_batches = 0
        def process_batch(x: np.ndarray, y: np.ndarray) -> None:
            weighted_sums, layer_outputs = self._forwardPass(x)
            grads_w, grads_b = self._gradients(y, weighted_sums, layer_outputs)
            self._update_params(grads_w, grads_b, self.learning_rate, processed_batches)

        self.start_time = time.time()
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            try:
                if self.batch_size != 0:
                    np.random.shuffle(data_index)
                    
                    for batch_start_i in range(0, x.shape[0], self.batch_size):
                        indicies = data_index[batch_start_i:batch_start_i + self.batch_size]
                        processed_batches += 1
                        process_batch(x[indicies, :], y[indicies, :])
                else:
                    processed_batches += 1
                    process_batch(x, y)

                error_func = self._get_error_func()
                _, validation_outputs = self._forwardPass(x_val)
                validation_output = validation_outputs[-1]
                loss = error_func(y_val, validation_output)
                best_loss = min(loss, best_loss) if epoch > 0 else loss

                self.val_loss_.append(loss)
                _, train_outputs = self._forwardPass(x)
                self.train_loss_.append(error_func(y, train_outputs[-1]))

                if loss > best_loss:
                    worse_counter += 1
                else:
                    worse_counter = 0
                    best_weights = [w.copy() for w in self.layer_weights]
                    best_biases = [b.copy() for b in self.layer_biases]
                    best_epoch = epoch
                if worse_counter == self.patience:
                    break

                if self.verbose:
                    print_progress(self, self.current_epoch, self.epochs, 1)
                if self.current_epoch % 100 == 0 and self.save_auto:
                    os.makedirs(temp_folder, exist_ok=True)
                    self.save(f"{temp_folder}/{self.save_prefix}_epoch_{self.current_epoch}")
            except KeyboardInterrupt:
                exit_handler(best_weights, best_biases, best_epoch, exit_type="interrupt")
                print("WARNING: Training interrupted by user.")
                return self
            
        exit_handler(best_weights, best_biases, best_epoch, exit_type="normal")
        return self
    
    def score(self, x: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> float:
        """
        Evaluate the model's performance on the provided data.

        Parameters
        ----------
        x : np.ndarray | pd.DataFrame
            Input data matrix of shape (n_samples, n_features).
        y : np.ndarray | pd.Series
            True target values.
            - For Binary Classification: Shape (n_samples,) or (n_samples, 1).
            - For Multiclass Classification: One-hot encoded shape (n_samples, n_classes).
            - For Regression: Shape (n_samples,) or (n_samples, 1).

        Returns
        -------
        float
            Performance metric.
            - For Classification: Accuracy score.
            - For Regression: R² score.
        """
        def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            return np.mean(y_true == y_pred)

        def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - ss_res / ss_tot
        
        x = check_input(x)
        y = check_input(y)
        y_pred = self.predict(x)
        
        if self.method in {"Binary Classification", "Multiclass Classification"}:
            if len(y.shape) == 2 and y.shape[1] > 1:
                y_true = np.argmax(y, axis=1)
            else:
                y_true = y.flatten()
            return accuracy_score(y_true, y_pred)
        elif self.method == "Regression":
            return r2_score(y, y_pred)
    
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
        
        x = check_input(x)
        
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
            x = check_input(x)
            output = self._forwardPass(x)[1][-1].flatten()
            return output
    
    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            Only for compatibility with sklearn, has no effect.

        Returns
        -------
        dict
            Parameter names mapped to their values.
        """
        params = {
                 "sizes_of_hidden_layers": self.sizes_of_hidden_layers,
                 "epochs": self.epochs,
                 "learning_rate": self.learning_rate,
                 "random_state": self.random_state,
                 "verbose": self.verbose,
                 "optimizer": self.optimizer,
                 "regularization_setting": self.regularization_setting,
                 "batch_size": self.batch_size,
                 "patience": self.patience,
                 "regression": self.regression,
                 "auto_save": self.auto_save,
                 "hidden_activation_func": self.hidden_activation_func,
                 "output_activation_func": self.output_activation_func
        }
        return params
    
    def set_params(self, **params: Any) -> Self:
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        FeedForwardNeuralNetwork
            Returns self.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"WARNING: Invalid parameter '{key}' for estimator {self.__class__.__name__}.")
        return self