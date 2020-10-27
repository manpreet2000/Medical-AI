import pandas as pd
import numpy as np
import tensorflow.keras.backend as tf
import tensorflow
def get_class_freq(label):
    """ Returns positive class frequency and 
    negative class freq """
    N=len(label)
    wp=np.sum(label)/N
    wn=1-wp
    return wp,wn

def get_weighted_loss(wp,wn,ep=1e-7):

    def weighted_loss(y_true,y_pred):
        
        y_pred=tensorflow.dtypes.cast(y_pred,dtype=tensorflow.float32)
        y_true=tensorflow.dtypes.cast(y_true,dtype=tensorflow.float32)
        pos_loss= -1*tf.mean( wp * y_true * tf.log(y_pred+ep) )
        neg_loss= -1*tf.mean( wn * (1 - y_true) * tf.log(1 - y_pred+ep) )
        loss=pos_loss+neg_loss
        return loss
    return weighted_loss