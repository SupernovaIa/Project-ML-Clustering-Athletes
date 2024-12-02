# Data processing  
# -----------------------------------------------------------------------  
import pandas as pd  
import numpy as np  

# Data scaling and preprocessing  
# -----------------------------------------------------------------------  
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler  


def scale_df(df, cols, method="robust", include_others=False):
    """
    Scale selected columns of a DataFrame using specified scaling method.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        cols (list): List of columns to scale.
        method (str): Scaling method, one of ["minmax", "robust", "standard"]. Defaults to "robust".
        include_others (bool): If True, include non-scaled columns in the output. Defaults to False.
    
    Returns:
        pd.DataFrame: DataFrame with scaled columns (and optionally unscaled columns).
        scaler: Scaler object used for scaling.
    """
    if method not in ["minmax", "robust", "standard"]:
        raise ValueError(f"Invalid method '{method}'. Choose from ['minmax', 'robust', 'standard'].")
    
    if not all(col in df.columns for col in cols):
        missing = [col for col in cols if col not in df.columns]
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    
    # Select the scaler
    scaler = {
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
        "standard": StandardScaler()
    }[method]
    
    # Scale the selected columns
    scaled_data = scaler.fit_transform(df[cols])
    df_scaled = pd.DataFrame(scaled_data, columns=cols, index=df.index)
    
    # Include unscaled columns if requested
    if include_others:
        unscaled_cols = df.drop(columns=cols)
        df_scaled = pd.concat([df_scaled, unscaled_cols], axis=1)
    
    return df_scaled, scaler