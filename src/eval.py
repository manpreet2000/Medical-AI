import tensorflow as tf
import argparse
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import models

parser = argparse.ArgumentParser()
parser.add_argument("path", help="path of image",
                    type=str)
args = parser.parse_args()

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


model.load_weights("/weights/model_weights1.h5")

print("")
print("Loading Image")
img=tf.keras.preprocessing.image.load_img(args.path,target_size=(150, 150))
img=tf.keras.preprocessing.image.img_to_array(img)
img=np.expand_dims(img,axis=0)
print("Predicting ")
result=model.predict(img)
if result[0][0]<0.50:
    result="Normal"
else:
    result="Pnemonia"

print("*"*50)
print("Result is...")
print(result)