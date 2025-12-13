import pandas as pd
import numpy as np
from scipy.sparse import spmatrix

def check_input(input: pd.DataFrame | np.ndarray| spmatrix) -> np.ndarray:
        if isinstance(input, pd.DataFrame) or isinstance(input, pd.Series):
            corrected = input.to_numpy(copy=True, dtype=float)
        elif isinstance(input, np.ndarray):
            return input
        elif isinstance(input, spmatrix):
            return input.toarray()
        else:
            raise TypeError(f"X is unsupported type {type(input).__name__}, must be np.ndarray, pd.DataFrame or spmatrix")
        return corrected  