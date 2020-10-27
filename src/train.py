import tensorflow as tf
import numpy as np
import pandas as pd
import split
import class_imbalance
import matplotlib.pyplot as plt
IMG_DIR="/home/manpreet/codes/medical/pnumonia/data/Xray/xray/train"

def train_generator(df,image_dir,x_col,y_col,shape,shuffle=True,batch_size=1):
    print("Image Generator Loading !!")
    image_generator=tf.keras.preprocessing.image.ImageDataGenerator(samplewise_center=True,
    samplewise_std_normalization=True)

    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_col,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            target_size=shape)
    return generator

def val_generator(df,image_dir,x_col,y_col,shape,shuffle=True,batch_size=1):
    print("Valid Image Generator Loading !!")
    image_generator=tf.keras.preprocessing.image.ImageDataGenerator(featurewise_center=True,
        featurewise_std_normalization= True)

    generator = image_generator.flow_from_dataframe(
            dataframe=df,
            directory=image_dir,
            x_col=x_col,
            y_col=y_col,
            class_mode="raw",
            batch_size=batch_size,
            shuffle=shuffle,
            target_size=shape)
    return generator





df=pd.read_csv("data/metadata.csv").drop(["Unnamed: 0","Dataset_type","Label_2_Virus_category","Label_1_Virus_category"],1)
df_train,df_test=split.split_df(df)
train_gen=train_generator(df=df_train,image_dir=IMG_DIR,x_col="X_ray_image_name",y_col="Label",shape=(128,128))
valid_gen=val_generator(df=df_test,image_dir=IMG_DIR,x_col="X_ray_image_name",y_col="Label",shape=(128,128))

## plotting time
x,y=train_gen.__getitem__(0)
plt.imshow(x[0]);
plt.show()

base_model=tf.keras.applications.densenet.DenseNet121(include_top=False)
x=base_model.output
x=tf.keras.layers.GlobalAveragePooling2D()(x)
pred=tf.keras.layers.Dense(1,activation='sigmoid')(x)
model = tf.keras.models.Model(inputs=base_model.input, outputs=pred)
pos_weights,neg_weights=class_imbalance.get_class_freq(train_gen.labels)
model.compile(optimizer='adam', loss=class_imbalance.get_weighted_loss(pos_weights, neg_weights))
print("-"*80)
print("")
print("Ready !!")


model.fit_generator(train_gen, 
                              validation_data=valid_gen,
                              steps_per_epoch=100, 
                              validation_steps=25, 
                              epochs = 1)
model.save_weights('/home/manpreet/codes/medical/pnumonia/weights/my_model_weights.h5')