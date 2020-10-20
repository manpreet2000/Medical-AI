import config
import glob
import pandas as pd
import os
import numpy as np
import cv2

def load_images_in_df():
    data_map = []
    
    for sub_dir_path in glob.glob(config.DATA_PATH+"*"):
        if os.path.isdir(sub_dir_path):
            dirname = sub_dir_path.split("/")[-1]
            for filename in os.listdir(sub_dir_path):
                image_path = sub_dir_path + "/" + filename
                data_map.extend([dirname, image_path])
        else:
            print("This is not a dir:", sub_dir_path)
            
            
    df = pd.DataFrame({"dirname" : data_map[::2],
                    "path" : data_map[1::2]})
    #print(df.head())
    df_imgs = df[~df['path'].str.contains("mask")]
    df_masks = df[df['path'].str.contains("mask")]

    # Data sorting
    imgs = sorted(df_imgs["path"].values, key=lambda x : int(x[config.BASE_LEN:-config.END_IMG_LEN]))
    masks = sorted(df_masks["path"].values, key=lambda x : int(x[config.BASE_LEN:-config.END_MASK_LEN]))


    # Final dataframe
    df = pd.DataFrame({"patient": df_imgs.dirname.values,
                        "image_path": imgs,
                    "mask_path": masks})


    # Adding A/B column for diagnosis
    def positiv_negativ_diagnosis(mask_path):
        value = np.max(cv2.imread(mask_path))
        if value > 0 : return 1
        else: return 0

    df["diagnosis"] = df["mask_path"].apply(lambda m: positiv_negativ_diagnosis(m))


    return df

# if __name__ == "__main__":
#     load_images_in_df()