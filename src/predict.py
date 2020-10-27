#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:45:05 2020

@author: sudhanshukumar
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers
from tensorflow.keras import models

class pnemonia_class:
    def __init__(self,filename):
        self.filename =filename


    def predictpnemonia(self):
        # load model

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


        model.load_weights("./weights/model_weights1.h5")

        # summarize model
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (150 , 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        if result[0][0] >= 0.50:
            prediction = 'Pnemonia'
            return [{ "image" : prediction}]
            
        else:
            prediction = 'Normal'
            return [{ "image" : prediction}]
            


