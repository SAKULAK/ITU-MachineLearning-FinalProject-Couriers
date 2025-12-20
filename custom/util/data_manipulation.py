from typing import Literal
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer

def load_and_process_data(filepath: str, get_dummies: bool, scaler: Literal["standard", "quantile"] | None, return_full_df: bool = False) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]] | pd.DataFrame:

    scalar_mapping: dict[str, object] = {
        "standard": StandardScaler(),
        "quantile": QuantileTransformer(output_distribution='normal')
    }

    df = pd.read_csv(filepath)
    df.loc[df["Exposure"]>1, "Exposure"] = 1
    df = df.loc[df["VehAge"] < 30, :]

    df["DrivAgeCat"] = "Normal"
    df.loc[df["DrivAge"] < 25, "DrivAgeCat"] = "Young"
    df.loc[df["DrivAge"] > 70, "DrivAgeCat"] = "Old"
    df.reset_index(inplace=True, drop=True)

    df['BonusMalusBin'] = pd.cut(df['BonusMalus'], bins=[0, 75, 100, 150, 1000], labels=['Low', "Medium", 'Higher', 'High'])
    df['VehAgeBin'] = pd.cut(df['VehAge'], bins=[0, 1, 7, 13, 20, 30], labels=['New', "Average", 'Older', "Old", 'VeryOld'])

    K = 0.12
    Z = df["Exposure"] / (df["Exposure"] + K) # Credibility factor, if Exposure is high, Z -> 1, else Z -> 0
    avg_freq = df["ClaimNb"].sum() / df["Exposure"].sum()

    df["Risk"] = Z * (df["ClaimNb"] / df["Exposure"]) + (1 - Z) * avg_freq # Bayesian credibility formula, if Z -> 1, use individual frequency, else pull outliers to avg frequency, so
                                                                            # if Exposure is low, use avg frequency because bad luck could have caused high ClaimNb
    df["Risk"] = np.log(df["Risk"] + 1)

    numeric_columns = ["BonusMalus", "Density", "DrivAge", "VehAge"]
    categorical_columns = ["VehBrand", "VehPower", "DrivAgeCat", "BonusMalusBin", "Region", "Area", "VehGas"]
    X = df[numeric_columns + categorical_columns]
    X = X.astype({col: float for col in numeric_columns})

    if scaler is not None:
        scaler_instance = scalar_mapping[scaler]
        X.loc[:, numeric_columns] = scaler_instance.fit_transform(X[numeric_columns])    

    if get_dummies:
        X = pd.get_dummies(X, columns=categorical_columns)

    y = df['Risk']
    if return_full_df:
        return pd.concat([df[["ClaimNb", "Exposure"]], y, X], axis=1)
    if get_dummies:
        return X, y, numeric_columns, df.columns[len(numeric_columns):].tolist()
    else:
        return X, y, numeric_columns, categorical_columns

def split_data(X: pd.DataFrame, y: pd.Series, test_size: float, random_state: int|None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, X_test, y_train, y_test =  train_test_split(X.to_numpy(dtype=float), y.to_numpy(dtype=float).ravel(), test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test