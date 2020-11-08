# importing libraries
import pandas as pd

# custom libraries
import config

def cataract_or_not(txt):
    if "cataract" in txt:
        return 1
    else:
        return 0

def downsample(df):
    df = pd.concat([
        df.query('cataract==1'),
        df.query('cataract==0').sample(sum(df['cataract']), 
                                       random_state=42)
    ])
    return df


def preprocess_me(path):
    """ path: dataframe"""
    df=pd.read_csv(path)
    df['left_eye_cataract']=df["Left-Diagnostic Keywords"].apply(lambda x:cataract_or_not(x))
    df['right_eye_cataract']=df["Right-Diagnostic Keywords"].apply(lambda x:cataract_or_not(x))
    left_df=df.loc[:,['Left-Fundus','left_eye_cataract']].rename(columns={'left_eye_cataract':'cataract'})
    left_df['Path']=config.oc_img_path+"/"+left_df['Left-Fundus']
    left_df=left_df.drop(['Left-Fundus'],1)

    right_df=df.loc[:,['Right-Fundus','right_eye_cataract']].rename(columns={'right_eye_cataract':'cataract'})
    right_df['Path']=config.oc_img_path+"/"+right_df['Right-Fundus']
    right_df=right_df.drop(['Right-Fundus'],1)

    ## if you have data with little unbalance skip this step
    left_df = downsample(left_df)
    right_df = downsample(right_df)

    train_df = pd.concat([left_df, right_df], ignore_index=True)

    # shuffle 
    train_df=train_df.sample(frac=1.0)

    return train_df



