import pandas as pd

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')
    return df

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df