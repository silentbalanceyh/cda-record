import pandas as pd

def get_kind(x: pd.Series, diff_limit: int = 8):
    x = x.astype('str')
    x = x.str.extract(r'(^(\-|)(?=.*\d)\d*(?:\.\d*)?$)')[0]
    x.dropna(inplace=True)
    if x.nunique() > diff_limit:
        kind = 'numeric'
    else:
        kind = 'categorical'
    return kind