#Affine Data

import numpy as np
import glob
import os
import tensorflow as tf
import pandas as pd
import glob

import matplotlib.pyplot as plt
import SimpleITK as sitk

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

datapath = ('/arc/project/ex-rtam-1/Output_ants_Affine/DataSamples/All/')
patients = os.listdir(datapath)
labels_df = pd.read_csv('/arc/project/ex-rtam-1/Data_Index/Data_Index.csv', index_col = 0 )

labelset = []

for i in patients:
  label = labels_df.loc[i, 'Group']
  if label == 'AD':  # use `==` instead of `is` to compare strings
    labelset.append(0.)
  elif label == 'CN':
    labelset.append(1.)
  elif label == 'MCI':
    labelset.append(2.)
  else:
      raise "Oops, unknown label" 

labelset = to_categorical(labelset)


FullDataSet = []

for patient in patients:
  a = sitk.ReadImage(datapath + patient)
  b = sitk.GetArrayFromImage(a)
  c = np.reshape(b, (182, 218, 182, 1))
  FullDataSet.append(c)

FullDataSet = np.array(FullDataSet)

X_train, X_valid, y_train, y_valid = train_test_split(FullDataSet, labelset, train_size=0.80, random_state=42 )

## 3D CNN
CNN_model = tf.keras.Sequential(
  [
      #tf.keras.layers.Reshape([189, 233, 197, 1], input_shape=[189, 233, 197]), 
      tf.keras.layers.Input(shape =( 182, 218, 182, 1) ),                       
      tf.keras.layers.Conv3D(kernel_size=(7, 7, 7), filters=32, activation='relu',
                          padding='same', strides=(3, 3, 3)),
      #tf.keras.layers.BatchNormalization(),
      tf.keras.layers.MaxPool3D(pool_size=(3, 3, 3), padding='same'),
      tf.keras.layers.Dropout(0.20),
      
      tf.keras.layers.Conv3D(kernel_size=(5, 5, 5), filters=64, activation='relu',
                          padding='same', strides=(3, 3, 3)),
      #tf.keras.layers.BatchNormalization(),
      tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='same'),
      tf.keras.layers.Dropout(0.20),

      tf.keras.layers.Conv3D(kernel_size=(3, 3, 3), filters=128, activation='relu',
                          padding='same', strides=(1, 1, 1)),
      #tf.keras.layers.BatchNormalization(),
      tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='same'),
      tf.keras.layers.Dropout(0.20), 

      # last activation could be either sigmoid or softmax, need to look into this more. Sig for binary output, Soft for multi output 
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(256, activation='relu'),   
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dropout(0.20),
      tf.keras.layers.Dense(3, activation='softmax')
  ])

# Compile the model
CNN_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# print model layers
CNN_model.summary()

CNN_history = CNN_model.fit(X_train, y_train, epochs=300, validation_data=( X_valid, y_valid ))

#ploting all the results
plt.plot(CNN_history.history['accuracy'])
plt.plot(CNN_history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('/scratch/ex-rtam-1/Affine/AffineAllAcc.pdf')
print( "Training Accuracy is " + str(np.mean(CNN_history.history['accuracy'])))
print( "Validation Accuracy is " + str(np.mean(CNN_history.history['val_accuracy'])))
plt.clf()

# Plot training & validation loss values
plt.plot(CNN_history.history['loss'])
plt.plot(CNN_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('/scratch/ex-rtam-1/Affine/AffineAllLoss.pdf')
print( "Training Loss is " + str(np.mean(CNN_history.history['loss'])))
print( "Validation Loss is " + str(np.mean(CNN_history.history['val_loss'])))
plt.clf()

#model.save('/scratch/ex-rtam-1/Affine/AffineAllModel.h5')