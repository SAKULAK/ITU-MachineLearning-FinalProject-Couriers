import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_process_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.loc[df["Exposure"]>1, "Exposure"] = 1
    df = df.loc[df["VehAge"] < 30, :]

    df["AgeCategory"] = "Normal"
    df.loc[df["DrivAge"] < 25, "AgeCategory"] = "Young"
    df.loc[df["DrivAge"] > 70, "AgeCategory"] = "Old"

    # Interaction term?
    df["Interaction1"] = df["BonusMalus"]/ df["DrivAge"]

    # Create column ExposureDays as (Exposure * 365) % 366
    df['ExposureDays'] = (df['Exposure'] * 365).round() % 366

    # Create TimeBetweenClaims column as ExposureDays/ClaimNb
    df['TimeBetweenClaims'] = df['ExposureDays'] / (df['ClaimNb']+1)

    df["Risk"] = 1-((df["TimeBetweenClaims"]-df["ExposureDays"])/df["ExposureDays"])-1
    df['Risk'] /= df["Exposure"]

    for i, claimNb in enumerate(sorted(df["ClaimNb"].unique())):
        df.loc[df["ClaimNb"] == claimNb, "Risk"] += ((claimNb**2)*(df["Exposure"].max()-df.loc[df["ClaimNb"] == claimNb, "Exposure"]))

    df["Risk"] /= df["Risk"].max()
    return df

def split_data(numeric_features: list[str], categorical_features: list[str], target: str, df: pd.DataFrame, test_size: float, random_state: int|None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    types = {feature: float for feature in numeric_features}
    for feature in categorical_features:
        types[feature] = str
    df = df.astype(types)

    X = df[numeric_features+categorical_features]
    y = df[[target]]

    X.loc[:, numeric_features] = StandardScaler().fit_transform(X[numeric_features])

    X_encoded = pd.get_dummies(X, columns=categorical_features)
    X_train, X_test, y_train, y_test =  train_test_split(X_encoded.to_numpy(dtype=float), y.to_numpy(dtype=float).ravel(), test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test