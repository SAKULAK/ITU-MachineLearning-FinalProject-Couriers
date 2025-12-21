import numpy as np
from ..util.general import check_input
import heapq


class Node:
    def __init__(self, instances, prediction):
        self.is_leaf = True 
        self.instances = instances # Indices of data points in the node 
        self.prediction = prediction 

    def split(self, feature, threshold, left, right):
        self.is_leaf = False  # Becomes a decision node
        self.feature = feature  # Splitting feature
        self.threshold = threshold  # Threshold value for splitting
        self.left = left  
        self.right = right  

class DecisionTree:
    def __init__(self, criterion='gini', max_depth=None, min_to_split=2, min_samples_leaf = 1, max_leaves=None, min_impurity_decrease = 0.0):
        self.max_depth = max_depth
        self.min_to_split = min_to_split
        self.max_leaves = max_leaves
        self.min_impurity_decrease = min_impurity_decrease
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        
    def fit(self, X, y):
        X = check_input(X)
        y = check_input(y)
        self._data = X
        self._labels = y
        self._total_n = len(self._labels)
        if self.criterion == 'gini': self.criterion = self._gini_impurity
        if self.criterion == 'entropy': self.criterion = self._entropy 
        self.is_fitted_ = False

        if np.issubdtype(y.dtype, np.number):
            self._is_regression = True
            self.criterion = self._mse
        else:
            self._is_regression = False

        if not self._is_regression:
            self.classes_ = np.unique(y)

        #arange to get all the ids of the data for the root node
        self._root = self._make_leaf(np.arange(len(self._data)))
        if self.max_leaves is None:
            self._recursive_split(self._root, 0)
        else:
            self._closed_split()
        self.is_fitted_ = True
        return self

    def _find_best_split(self, node):

        instances = node.instances
        X = self._data
        y = self._labels
        n_samples = len(instances)
        n_features = X.shape[1]

        parent_impurity = self.criterion(instances)
        best_decrease = -1.0
        best_feature = None
        best_threshold =None
        best_left = None
        best_right = None

        if isinstance(self.min_samples_leaf, float):
            if not (0 < self.min_samples_leaf <= 1):
                raise ValueError("min_samples_leaf as float must be in (0, 1]")
            min_leaf = int(np.ceil(self.min_samples_leaf * n_samples))

        else:
            min_leaf = self.min_samples_leaf

        for feature in range(n_features):

            sorted_idx = instances[np.argsort(X[instances, feature])]
            X_sorted = X[sorted_idx, feature]

            for i in range(1, n_samples):

                left = sorted_idx[:i]
                right = sorted_idx[i:]

                if i < min_leaf or (n_samples - i) < min_leaf: continue

                if X_sorted[i] == X_sorted[i - 1]:continue

                #midpoint at the start
                threshold = (X_sorted[i] + X_sorted[i - 1]) / 2.0

                left_imp = self.criterion(left)
                right_imp = self.criterion(right)

                weighted_child_impurity = (len(left) / n_samples) * left_imp + (len(right) / n_samples) * right_imp
                node_weight = n_samples / self._total_n
                decrease = node_weight * (parent_impurity - weighted_child_impurity)

                if (decrease > best_decrease or
                (decrease == best_decrease and feature < best_feature) or
                (decrease == best_decrease and feature == best_feature and threshold < best_threshold)):
                    
                    best_decrease = decrease
                    best_feature = feature
                    best_threshold = threshold
                    best_left = left
                    best_right = right

        if best_feature is None:
            return None

        if best_decrease < self.min_impurity_decrease:
            return None
        
        return best_decrease, best_feature, best_threshold, best_left, best_right
    

    
    def _recursive_split(self,node, depth):
        if not self._can_split(node, depth): return
        best_split = self._find_best_split(node)
        _, feature, threshold, left, right = best_split if best_split is not None else (None, None, None, None, None)
        if left is None or right is None: return
        node.split(feature, threshold, self._make_leaf(left), self._make_leaf(right))
        self._recursive_split(node.left, depth + 1)
        self._recursive_split(node.right, depth + 1)


    def _make_heap_element(self, node, index, depth):
        split = self._find_best_split(node)
        if split is None: return None
        impurity_decrease, feature, threshold, left, right = split

        # (priority-impurit_decrease, index, depth, node, feature, threshold, left, right)
        return (-impurity_decrease, index, depth, node, feature, threshold,left,right)


        
    def _closed_split(self): 
        heap = [self._make_heap_element(self._root, 0, 0)]
        heap = [h for h in heap if h is not None]
        heapq.heapify(heap)

        for i in range (self.max_leaves-1):
            if not heap: break
            heap_element = heapq.heappop(heap)
            priority, index, depth, node, feature, threshold, left, right = heap_element
            node.split(feature, threshold, self._make_leaf(left), self._make_leaf(right))

            # from binary heap / binary tree array representation
            # left child index = 2 * i + 1
            # right child index = 2 * i + 2
                # index:   0
                    #     / \
                    #    1   2
                    #   / \ / \
                    #  3  4 5  6

            
            left_elem = self._make_heap_element(node.left, 2 * i + 1, depth + 1)
            right_elem = self._make_heap_element(node.right, 2 * i + 2, depth + 1)
            if left_elem: heapq.heappush(heap, left_elem)
            if right_elem: heapq.heappush(heap, right_elem)


    def predict(self, new_data):
        new_data = check_input(new_data)
        if not self.is_fitted_: raise RuntimeError("This DecisionTree instance is not fitted yet. Call 'fit' first.")
        data_size = len(new_data) 
        results = np.zeros(data_size)
        for i in range(data_size):
            node = self._root
            while not node.is_leaf:
                if new_data[i][node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            results[i] = node.prediction
        return results

    
    # for checking conditions before split
    def _can_split(self, node, depth):
        reached_max_depth = self.max_depth is None or depth < self.max_depth 
        reached_min_split = len(node.instances) >= self.min_to_split
        if self._is_regression: return reached_max_depth and reached_min_split
        #Check if the node contains more than one unique lable. Do NOT split if all the labels are equal to the majority class.
        not_a_single_lable = len(np.unique(self._labels[node.instances])) > 1 
        return reached_max_depth and reached_min_split and not_a_single_lable


    def _make_leaf(self, instances: np.array) ->Node:

        if self._is_regression:
            prediction = np.mean(self._labels[instances])

        else:
            labels = self._labels[instances]
            idx = np.searchsorted(self.classes_ , labels)
            counts = np.bincount(idx, minlength=len(self.classes_))
            prediction = self.classes_[counts.argmax()]

        return Node(instances, prediction)

    def get_params(self, deep=True):
        return {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_to_split": self.min_to_split,
            "max_leaves": self.max_leaves,
            "min_impurity_decrease": self.min_impurity_decrease,
            "min_samples_leaf": self.min_samples_leaf
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def _gini_impurity(self, instances: np.array) -> float:
        if len(instances) == 0: return 0.0

        labels = self._labels[instances]
        _, counts = np.unique(labels, return_counts=True)
        p = counts / len(instances)
        return 1.0 - np.sum(p ** 2)

    def _entropy(self, instances: np.array) -> float:
        if len(instances) == 0: return 0.0

        labels = self._labels[instances]
        _, counts = np.unique(labels, return_counts=True)
        p = counts / len(instances)
        p = p[p > 0]

        return -np.sum(p * np.log2(p))
        
    def _mse(self, instances):
        if len(instances) == 0: return 0.0
        y = self._labels[instances]
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)
