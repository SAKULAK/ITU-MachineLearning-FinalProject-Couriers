from sklearn.exceptions import NotFittedError
from typing import Self, Literal
from collections import Counter
import numpy as np
import pandas as pd


def impurity(classes: np.ndarray | pd.Series, measure: Literal["gini", "entropy"]) -> float:
    """Calculates region impurity.

    Implements gini and entropy impurity measures.

    Parameters
    ----------
    classes : np.ndarray | pd.Series
        1D array of datapoint classes in region.
    measure : Literal["gini", "entropy"]
        Impurity measure to use.

    Returns
    -------
    float
        Impurity score

    Raises
    ------
    ValueError
        Wrong measure.
    """
    n = len(classes)
    class_counts = Counter(classes)
    class_ps = np.array([count/n for c, count in class_counts.items()])

    if measure == "gini":
        impurity = (class_ps*(1-class_ps)).sum()
    elif measure == "entropy":
        impurity = -1*(class_ps*(np.log2(class_ps))).sum()
    else:
        raise ValueError("'measure' must be 'gini' or 'entropy'")

    return impurity

def weighted_impurity(regions: tuple[np.ndarray | pd.Series, np.ndarray | pd.Series], impurity_measure: Literal["gini", "entropy"]) -> float:
    """Calculates weighted impurity for a given split.

    Parameters
    ----------
    regions : tuple[np.ndarray | pd.Series, np.ndarray | pd.Series]
        Classes in regions
    impurity_measure : Literal[&quot;gini&quot;, &quot;entropy&quot;]
        Impurity measure to use.

    Returns
    -------
    float
        Weighted impurity score.
    """
    R1, R2 = regions
    n = len(R1) + len(R2)
    impuri = len(R1)/n*impurity(R1, impurity_measure) + len(R2)/n*impurity(R2, impurity_measure)
    return impuri

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