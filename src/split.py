import pandas as pd

def split_df(df,n=0.2):
    df=df.sample(frac=1)
    df.Label=df.Label.astype('category').cat.codes
    df_train=df.iloc[int(len(df)*n):,:]
    df_test=df.iloc[:int(len(df)*n),:]

    return df_train,df_test