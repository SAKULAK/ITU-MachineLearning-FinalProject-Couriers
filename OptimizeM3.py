from custom.util.data_manipulation import load_and_process_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from scipy.stats import randint, uniform, loguniform
import numpy as np
import gc
import json

N_ITER = 25
verbosity = 3
random_state = None

_, _, numeric_columns, categorical_columns, df = load_and_process_data(r'data/claims_train.csv', get_dummies=True, scaler = "standard", return_full_df=True)

del _
gc.collect()

high_risk_idx = list(df[df['ClaimNb'] > 1].index)*10
middle_risk_idx = df[df['ClaimNb'] == 1].index
low_risk_idx = df[df['ClaimNb'] == 0].sample(n=len(middle_risk_idx)+len(high_risk_idx)).index
subset_idx = middle_risk_idx.union(low_risk_idx).union(high_risk_idx)
df_subset = df.loc[subset_idx]
df_subset.sort_index(inplace=True)
df_subset.reset_index(inplace=True, drop=True)
subset_X = df_subset.loc[:, numeric_columns + categorical_columns]
subset_y = df_subset["Risk"]

X_train, X_test, y_train, y_test = train_test_split(subset_X, subset_y, test_size=0.2, random_state=random_state)

try:
    with open("best_random_hyperparameters.json", "r") as f:
        best_params_dict = json.load(f)
    print("Loaded existing best random hyperparameters.")
except FileNotFoundError:
    best_params_dict = None
    print("No existing hyperparameter file found. Proceeding with optimization.")

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

if best_params_dict is None:
    print("Starting Randomized Search for Hyperparameter Optimization...")
    rf_dist = {
        'n_estimators': randint(100, 1000),
        'max_depth': [None] + list(range(10, 50)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': [1.0, 'sqrt', 'log2']
    }

    gb_dist = {
        'n_estimators': randint(100, 1000),
        'learning_rate': loguniform(0.001, 0.2),
        'max_depth': randint(3, 10),
        'subsample': uniform(0.5, 0.5),
        'min_samples_split': randint(2, 20),
        'loss': ['squared_error', 'huber']
    }
    mlp_dist = {
        'mlp__hidden_layer_sizes': [
            (50,), (100,), (50, 50), (100, 50), (100, 100), (200, 100), (200, 200, 100, 50), (300, 200, 100, 50), (50, 50, 50), (10, 10, 10)
        ],
        'mlp__activation': ['relu'],
        'mlp__solver': ['adam'],
        'mlp__alpha': loguniform(1e-5, 1e-1),
        'mlp__learning_rate_init': loguniform(1e-4, 1e-1)
    }

    print("Tuning Random Forest...")
    rs_rf = RandomizedSearchCV(
        RandomForestRegressor(random_state = random_state),
        param_distributions=rf_dist,
        n_iter=N_ITER,
        cv=5,
        verbose=verbosity,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        random_state = random_state
    )
    rs_rf.fit(X_train, y_train)

    print("Best RF Params:", rs_rf.best_params_)

    print("Tuning Gradient Boosting...")
    rs_gb = RandomizedSearchCV(
        GradientBoostingRegressor(random_state = random_state),
        param_distributions=gb_dist,
        n_iter=N_ITER,
        cv=5,
        verbose=verbosity,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        random_state = random_state
    )
    rs_gb.fit(X_train, y_train)

    print("Best GB Params:", rs_gb.best_params_)

    print("Tuning MLP (Pipeline)...")
    mlp_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(random_state = random_state, max_iter=1500))
    ])

    rs_mlp = RandomizedSearchCV(
        mlp_pipe,
        param_distributions=mlp_dist,
        n_iter=N_ITER,
        cv=5,
        verbose=verbosity,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        random_state = random_state
    )
    rs_mlp.fit(X_train, y_train)

    print("Best MLP Params:", rs_mlp.best_params_)

    best_params_dict = {
        "RandomForest": rs_rf.best_params_,
        "GradientBoosting": rs_gb.best_params_,
        "MLP": rs_mlp.best_params_
    }

    filename = "best_random_hyperparameters.json"

    with open(filename, "w") as f:
        json.dump(best_params_dict, f, cls=NumpyEncoder, indent=4)

    print(f"Successfully saved parameters to {filename}")

def generate_fine_grid(best_params, spread=0.2, num_steps=3):
    new_grid = {}
    
    for key, value in best_params.items():
        
        if isinstance(value, int) and not isinstance(value, bool):
            lower = int(value * (1 - spread))
            upper = int(value * (1 + spread))
            
            lower = max(1, lower) 
            
            if lower == upper:
                values = [lower]
            else:
                step = max(1, (upper - lower) // (num_steps * 2))
                values = list(range(lower, upper + 1, step))
                
            if value not in values:
                values.append(value)
            
            new_grid[key] = sorted(list(set(values)))
            
        elif isinstance(value, float):
            lower = value * (1 - spread)
            upper = value * (1 + spread)
        
            values = np.linspace(lower, upper, num_steps * 2 + 1).tolist()
            new_grid[key] = values
        else:
            new_grid[key] = [value]
            
    return new_grid

print("Generating fine-tuned grids...")

fine_grid_rf = generate_fine_grid(best_params_dict["RandomForest"], spread=0.2, num_steps=1)
fine_grid_gb = generate_fine_grid(best_params_dict["GradientBoosting"], spread=0.2, num_steps=1)
fine_grid_mlp = generate_fine_grid(best_params_dict["MLP"], spread=0.2, num_steps=1)

print("New RF Grid:", fine_grid_rf)
print("New GB Grid:", fine_grid_gb)
print("New MLP Grid:", fine_grid_mlp)

print("Running Fine-Tuning GridSearch for RF...")
gs_rf = GridSearchCV(RandomForestRegressor(random_state = random_state), fine_grid_rf, cv=2, n_jobs=-1, verbose=verbosity)
gs_rf.fit(X_train, y_train)
print("Best RF Params after Fine-Tuning:", gs_rf.best_params_)

print("Running Fine-Tuning GridSearch for GB...")
gs_gb = GridSearchCV(GradientBoostingRegressor(random_state = random_state), fine_grid_gb, cv=2, n_jobs=-1, verbose=verbosity)
gs_gb.fit(X_train, y_train)
print("Best GB Params after Fine-Tuning:", gs_gb.best_params_)

# For MLP (Pipeline)
gs_mlp = GridSearchCV(mlp_pipe, fine_grid_mlp, cv=2, n_jobs=-1, verbose=verbosity)
gs_mlp.fit(X_train, y_train)
print("Best MLP Params after Fine-Tuning:", gs_mlp.best_params_)

print("Optimization Complete.")

best_params_dict = {
    "RandomForest": gs_rf.best_params_,
    "GradientBoosting": gs_gb.best_params_,
    "MLP": gs_mlp.best_params_
}

filename = "best_hyperparameters.json"

with open(filename, "w") as f:
    json.dump(best_params_dict, f, cls=NumpyEncoder, indent=4)


print("Building Final Stacking Ensemble...")

best_rf_model = gs_rf.best_estimator_
best_gb_model = gs_gb.best_estimator_
best_mlp_model = gs_mlp.best_estimator_

stacking_model = StackingRegressor(
    estimators=[
        ('rf', best_rf_model),
        ('gb', best_gb_model),
        ('mlp', best_mlp_model)
    ],
    final_estimator=Ridge(),
    cv=5,
    n_jobs=-1,
    verbose=verbosity
)

stacking_model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error

y_pred = stacking_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Final Ensemble Test MSE: {mse}")