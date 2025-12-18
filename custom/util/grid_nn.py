from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from models.feedforward_neural_network import FeedForwardNeuralNetwork


# first creating a warpper which will bring our nn to the format that scikit gridseacrch requires
class FFNNRegressor(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        sizes_of_hidden_layers=(64,),
        epochs=50,
        learning_rate=0.001,
        batch_size=32,
        optimizer="adam",
        regularization_setting=None,
        hidden_activation_func="relu",
        patience=0,
        random_state=42
    ):
        self.sizes_of_hidden_layers = sizes_of_hidden_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.regularization_setting = regularization_setting
        self.hidden_activation_func = hidden_activation_func
        self.patience = patience
        self.random_state = random_state


    def fit(self, X, y):
        self.model_ = FeedForwardNeuralNetwork(
        sizes_of_hidden_layers=list(self.sizes_of_hidden_layers),
        epochs=self.epochs,
        learning_rate=self.learning_rate,
        batch_size=self.batch_size,
        optimizer=self.optimizer,
        regularization_setting=self.regularization_setting,
        hidden_activation_func=self.hidden_activation_func,
        output_activation_func="linear",
        regression=True,
        patience=self.patience,
        random_state=self.random_state,
        verbose=False
        )

        self.model_.fit(X, y)
        return self

    def predict(self, X):
        preds = self.model_.predict(X)
        return preds.ravel()
    
    # scoring returns the negative mse, sklearn alr does this apparently
    # def score(self, X, y):
        # preds = self.predict(X)
        # return -np.mean((y - preds) ** 2)




param_grid = {
    "sizes_of_hidden_layers": [(32,), (64,), (64, 32)],
    "learning_rate": [0.01, 0.001],
    "epochs": [50, 100],
    "batch_size": [16, 32],
    "optimizer": ["adam", "sgd"]
}

grid = GridSearchCV(
    estimator=FFNNRegressor(),
    param_grid=param_grid,
    scoring="neg_mean_squared_error",
    cv=5,           #5 fold split
    n_jobs=-1
)

grid.fit(X_train, y_train)

# will print a list of scores
risk_scores = grid.predict(X_test)
print(risk_scores)

# will print the dictionary of the best model
print(grid.best_params_)