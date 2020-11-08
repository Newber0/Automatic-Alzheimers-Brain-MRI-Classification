# Overview of Methods, from Data Acquisition to Output of Trained Model

The purpose of this project is to compare the effects of different methods of image registration, or spatial normalization, on the ability of a neural network to correctly classify 3D Brain MRI data as Alzhiemer's Disease (AD), Mild Cognitive Injury (MCI), or Control (CN). This project will make use of the Advanced Normalization Tool package, or [ANTs](https://antspy.readthedocs.io/en/latest/) the Robust Brain Extraction package, or [ROBEX](https://www.nitrc.org/projects/robex), and [Keras](https://keras.io/).

All code associated with this project can be found in this repository separated by function, but this document will act as an overarching summary of all work completed. While the data will not be provided here, it can be acquired from the [ADNI Database](http://adni.loni.usc.edu/) with permission. Upon download of the data a CSV with identifying information for each image is also made avaliable. This can be found [here](https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/Data_Index.csv) in the case of this project.

[Obtain data from ADNI Website](#Obtain_data_from_ADNI_Website)

[Preprocessing of File Structure](#Preprocessing_of_File_Structure)

[Brain Extraction using ROBEX](#Brain_Extraction_using_ROBEX)

[Registration to Template and Segmentation into GM, WM, and CSF using ANTs](#Registration_to_Template_and_Segmentation_into_GM,_WM,_and_CSF_using_ANTs)

[Separation of Data prior to CNN Entry](#Separation_of_Data_prior_to_CNN_Entry)

[Construction of CNN](#Construction_of_CNN)

[Summary of Output](#Summary_of_Output)


# <a name="Obtain_data_from_ADNI_Website"></a>Obtain data from ADNI Website

We obtained our data from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) database. We selected T1 weighted 3T images including AD, MCI, and CN labelled data. All available data in this category was taken from the ADNI Database. A CSV file containing all identifying information for each image was provided upon download of the data. The data was downloaded from different sources, so these files were combined into a master excel file which can be found [here](https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/Data_Index.csv). 

# <a name="Preprocessing_of_File_Structure"></a>Preprocessing of File Structure

The downloaded file structure was organized by date, patient, and preprocessing types. All data was removed and placed into one file to reduce complicated file parsing. 

Each file contained all identifying information for that file including the file structure as well as a unique image ID. All files were renamed to include only the image ID. This was accomplished with the simple line of code below.

```
import os

filepath = '/Data_Directory/'
# '/Data_Directory/' is wherever the data is stored.

for f in os.listdir(filepath):

  # splitting filename into components
  file_name, file_ext = os.path.splitext(f)
  ADN, REDUNDANT, ID = file_name.split('I')

  # mild formatting to ensure names match up
  ID = ID.strip()
  file_ext = file_ext.strip()
  
  # Renaming of files
  new_name = ('{}{}'.format(ID,file_ext))
  os.rename((filepath + f), (filepath + new_name)) 
)
```
For example, this code renamed all files from something like

ADNI_136_S_0579_MR_SmartBrain_br_raw_20080121170059746_1_S44769_I87960.nii 

to 

87960.nii.

All files contain two capital I characters, first as part of 'ADNI' and second preceeding the Image ID, therefore the filename is split by this and renaming is simple

# <a name="Brain_Extraction_using_ROBEX"></a>Brain Extraction using ROBEX and further Preprocessing

The tool used for this step was the Robust Brain Extraction package was used which can be found [here](https://www.nitrc.org/projects/robex). This removes the tissue surrounding and including the skull. Isolating exclusively brain tissue in an effective manner dramatically increase the accuracy of the neural network. The code for this process can be found below.
```
cd /Working_Dir/

for f in /ROBEX_InputData/* ; do
	./ROBEX/runROBEX.sh /InputData/$f /ROBEX_OutputData/$f ;
	done
```
This is not run using python for simplicities sake on our end. The UBC ARC Sockeye system was used for all computation of this project and compatibility issues dictate we do this.

# <a name="Registration_to_Template_and_Segmentation_into_GM,_WM,_and_CSF_using_ANTs"></a>Registration to Template and Segmentation into GM, WM, and CSF using ANTs using ANTs

Registration of the image to normalised space was carried out using the ANTsPy package (Python Version of ANTs). Here is the [Documentation](https://antspy.readthedocs.io/en/latest/) and [GitHub download](https://github.com/ANTsX/ANTsPy) package. This is a multistage process requiring registration to a template using the various methods we are interested in, and segmentation of these images into grey matter (GM), white matter (WM), and Cerebrospinal Fluid (CSF). First we will focus on Registration.

Registration methods that we tested include Translation, Affine, ElasticSyN, SyNRA, and SyNAggro. These methods range from least to most aggressive, roughly in this order. The code for each of these registrations can be found in this repository, however the difference is minimal for our purposes so Affine will be used as an Example here. 

```
import ants
import os

# this links the list of files when running the full dataset through
link = '/ROBEX_Output/'

# Skull stripped data output by ROBEX
mylist = os.listdir('/ROBEX_Output/')

# fixed image is the template to which the registration will be fitted found in the downloaded ANTs package
fixed = ants.image_read('/ANTsPy-master/data/mni.nii.gz')

# This runs the entire dataset through registration
for i in mylist:
  # registraction
  moving = ants.image_read(link+i)
  mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='Affine' )
  mywarpedimage = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=mytx['fwdtransforms'])
  mywarpedimage.to_file('/Output_ants_Affine/reg_ants/'+i)
  
  #segmentation
  mask=ants.get_mask(mywarpedimage)
  img_seg = ants.atropos(a=mywarpedimage, m='[0.2,1x1x1]', c='[2,0]', i='kmeans[3]', x=mask)
  gm =img_seg['probabilityimages'][1]
  gm.to_file('/Output_ants_Affine/greymatter/'+i)
  continue;
```
This results in two outputs, the Raw Registered data, and the segmentation of that Raw data into GM, WM, and CSF. The literature states that grey matter presents signs of MCI and AD therefore we are only interested in the GM mask as an output, the other two are unnecessary.

# <a name="Separation_of_Data_prior_to_CNN_Entry"></a>Separation of Data prior to CNN Entry

The original scope of this project was to classify MRI data as either AD or CN, but when the data was downloaded there was a high rate of images with MCI and the decision to include this class was made. This complicated the process because binary classification is far easier than multiclass classification. As a result we decided to run multiple networks for each registration method, 3 binary networks, (comparing AD to CN, MCI to CN, and AD to MCI) and one multiclass network (AD vs CN vs MCI). Before building the network, the data was split into 3 files containing AD, MCI, and CN respectively, and then from these files, copies were taken to create the following 4 files; ADvsCN, MCIvsCN, ADvsMCI, and ADvsMCIvsCN. Each network could then sample from the correct file instead of having to parse through the full set for each network. Additionally, to ensure consistency, 100 samples from each class were taken. Below is an example of the code for Affine that achieved this.

```
import os
import numpy as np
import pandas as pd
import glob
import shutil

# CSV file will be used to match image ID to one of the three labels
index = '/Data_Index.csv'
df = pd.read_excel(index)

# identifying Image ID for AD
AD = df[(df.Group == 'AD')]
AD = (AD['Image Data ID'])
AD = list(AD)
AD = [str(i) for i in AD]

# identifying Image ID for MCI
MCI = df[(df.Group == 'MCI') ]
MCI = (MCI['Image Data ID'])
MCI = list(MCI)
MCI = [str(i) for i in MCI]

# identifying Image ID for CN
CN = df[(df.Group == 'CN')]
CN = (CN['Image Data ID'])
CN = list(CN)
CN = [str(i) for i in CN]

## Move files into appropriate directory for sorting
for i in AD:
  shutil.move('/Output_ants_Affine/Greymatter/' + i + '.nii', '/Output_ants_Affine/AD/')
  continue;

for i in CN:
  shutil.move('/Output_ants_Affine/Greymatter/' + i + '.nii', '/Output_ants_Affine/CN/')
  continue;

for i in MCI:
  shutil.move('/Output_ants_Affine/Greymatter/' + i + '.nii', '/Output_ants_Affine/MCI/')
  continue;
```
This sorts all available Affine greymatter files into three files, AD, MCI, and CN. Next 100 samples from each are taken and placed into the files that will be used for training and testing the networks. These include ADvsCN, MCIvsCN, and ADvsMCIvsCN.

```
# Moving AD files
cd /Output_ants_Affine/AD/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Affine/DataSamples/ADvsCN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Affine/DataSamples/ADvsMCI/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Affine/DataSamples/ADvsMCIvsCN/

# Moving CN files
cd /Output_ants_Affine/CN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Affine/DataSamples/ADvsCN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Affine/DataSamples/MCIvsCN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Affine/DataSamples/ADvsMCIvsCN/

#Moving MCI files
cd /Output_ants_Affine/MCI/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Affine/DataSamples/ADvsMCI/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Affine/DataSamples/MCIvsCN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Affine/DataSamples/ADvsMCIvsCN/
```
The above code is a shell script used for the copying of the files without removing the source files so that the same files can be used in multiple comparisons.

Next is the construction and implementation of the neural network. In this case a Convolutional Neural Network.

# <a name="Construction_of_CNN"></a>Construction of CNN

We chose to construct a 3D CNN as this is suited to both the features we wish to identify and the data we have. 3D MRI data is made up of multiple slices of 2D data which can be interpreted as 3d data. As the features that are characteristic of AD and MCI will be found across multiple slices it is important that the kernels be 3D as well. Typically with machine vision, a 3d image is 4 dimensional, 3 spatial dimensions and one colour channel dimension. An RGB image has three colour channels, therefore an RGB image that is 5 by 10 by 15 pixels would have a dimensionality of 5,10,15,3. In the case of a binary image there is no colour channel dimension, therefore a dummy dimension has to be added in order for successful input into a 3D CNN. 

It is important to note that the purpose of this project is to compare registration methods, therefore no major fine tuning of the network took place, and a rather simplistic network was used to accentuate the differences.

We chose to construct one network for each comparison, for each registration method, meaning 5 times 4 networks for a total of 20 networks. All draw from different sources and output to different source, but the general code is the same. The exception are the multiclass networks, which had to be changed to accept multiple classes. Here we will show the ADvsCN and ADvsCNvsMCI networks for the Affine registration method, but all other networks can be found in this repository as well. Here we utilize the Keras package found [here](https://keras.io/). Also note that an additional model with unregistered data was used as well. This would act as a baseline.

```
#Affine Data ADvsCN CNN
# Importing packages
import numpy as np
import os
import tensorflow as tf
import pandas as pd

import matplotlib.pyplot as plt
import SimpleITK as sitk

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

# Setting up Datapaths and CSV file
datapath = ('/Output_ants_Affine/DataSamples/ADvsCN/')
patients = os.listdir(datapath)
labels_df = pd.read_csv('/Data_Index/Data_Index.csv', index_col = 0 )

# Setting up Label data for intry into CNN
labelset = []

for i in patients:
  label = labels_df.loc[i, 'Group']
  if label == 'AD':  # use `==` instead of `is` to compare strings
    labelset.append(0.)
  elif label == 'CN':
    labelset.append(1.)
  else:
      raise "Oops, unknown label" 

labelset = tuple(labelset)
labelset = np.array(labelset)

# Setting up dataseet for entry to CNN
FullDataSet = []

# Important to note that here a dummy dimension is added to make the data compatible with the 3D CNN architecture
for patient in patients:
  a = sitk.ReadImage(datapath + patient)
  b = sitk.GetArrayFromImage(a)
  c = np.reshape(b, (182, 218, 182, 1))
  FullDataSet.append(c)

FullDataSet = np.array(FullDataSet)

#splitting training and testing data
X_train, X_valid, y_train, y_valid = train_test_split(FullDataSet, labelset, train_size=0.80, random_state=42 )

## 3D CNN Architecture
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

      # Because this is a binary classification we use a sigmoid activation to the final layer, and the final layer is a single node 
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(256, activation='relu'),   
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dropout(0.20),
      tf.keras.layers.Dense(1, activation='sigmoid')
  ])

# Compile the model, compiled with binary crossentropy due to binary classification
CNN_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

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
plt.savefig('/Affine/AffineADvsCNAcc2.pdf')
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
plt.savefig('/Affine/AffineADvsCNLoss2.pdf')
print( "Training Loss is " + str(np.mean(CNN_history.history['loss'])))
print( "Validation Loss is " + str(np.mean(CNN_history.history['val_loss'])))
plt.clf()

```
Next is the network classifying one of all three classes. The only major changes are the introduction of an additional label, the changing of the final layer to have 3 Nodes and a softmax activation function, and a loss function of categorical crossentropy.

```
#Affine Data ADvsCNvsMCI CNN

#importing of packages
import numpy as np
import os
import tensorflow as tf
import pandas as pd

import matplotlib.pyplot as plt
import SimpleITK as sitk

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

# Setting up datapaths
datapath = ('/Output_ants_Affine/DataSamples/All/')
patients = os.listdir(datapath)
labels_df = pd.read_csv('/Data_Index/Data_Index.csv', index_col = 0 )

Setting up label data
labelset = []

# Here is a change compared to the binary model, 3 classes not 2
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

#setting up the dataset, including dummy dimension as previously mentioned
FullDataSet = []

for patient in patients:
  a = sitk.ReadImage(datapath + patient)
  b = sitk.GetArrayFromImage(a)
  c = np.reshape(b, (182, 218, 182, 1))
  FullDataSet.append(c)

FullDataSet = np.array(FullDataSet)

#splitting into training and testing
X_train, X_valid, y_train, y_valid = train_test_split(FullDataSet, labelset, train_size=0.80, random_state=42 )

## 3D CNN architecture
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
plt.savefig('/Affine/AffineAllAcc.pdf')
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
plt.savefig('/Affine/AffineAllLoss.pdf')
print( "Training Loss is " + str(np.mean(CNN_history.history['loss'])))
print( "Validation Loss is " + str(np.mean(CNN_history.history['val_loss'])))
plt.clf()
```
Batch Normalization was found to be ineffective, but was kept in the code for transparency.

All CNNs were run and the results are discussed below.
# <a name="Summary_of_Output"></a>Summary of Output

Registration had a significant positive effect on the ability of a CNN to classify 3D Brain MRI data as AD, MCI, or CN. Model accuracy and loss were used as metrics too determine success for all models. Graphs displaying this information can be found in this repository [here](). Between the different classifications, the highest accuracy and lowest loss was achieved by ADvsCN, followed by ADvsMCI. MCIvsCN performed poorly and ADvsMCIvsCN performed the worst. Below the results for Affine registration associated with these methods are shown.

ADvsCN Affine Accuracy

<img src="https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/CNN_Results/ADvsCN/Affine/AffineADvsCNAcc2-1.jpg" width="40%"> 

ADvsCN Affine Loss

<img src="https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/CNN_Results/ADvsCN/Affine/AffineADvsCNLoss2-1.jpg" width="40%">

ADvsMCI Affine Accuracy

<img src="https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/CNN_Results/ADvsMCI/Affine/AffineADvsMCIAcc-1.jpg" width="40%">

ADvsMCI Affine Loss

<img src="https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/CNN_Results/ADvsMCI/Affine/AffineADvsMCILoss-1.jpg" width="40%">

MCIvsCN Affine Accuracy

<img src="https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/CNN_Results/MCIvsCN/Affine/AffineMCIvsCNAcc-1.jpg" width="40%"> 

MCIvsCN Affine Loss

<img src="https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/CNN_Results/MCIvsCN/Affine/AffineMCIvsCNLoss-1.jpg" width="40%">

ADvsMCIvsCN Affine Accuracy

<img src="https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/CNN_Results/All/Affine/AffineAllAcc-1.jpg" width="40%">

ADvsMCIvsCN Affine Loss

<img src="https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/CNN_Results/All/Affine/AffineAllLoss-1.jpg" width="40%">


It is possible that signs of AD are strong enough that the network can accurately identify these features. This would explain why the ADvsCN and ADvsMCI models work well but the MCIvsCN does not. The multiclass model performing poorly was expected, as the architecture was not tuned for such a model. From this point only the ADvsCN information will be shown when comparing different registration methods.

The less aggressive methods, Translation and Affine show the worst results:

ADvsCN Translation Accuracy

<img src="https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/CNN_Results/All/Affine/AffineAllAcc-1.jpg" width="40%">

ADvsCN Translation Loss

<img src="https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/CNN_Results/All/Affine/AffineAllLoss-1.jpg" width="40%">

ADvsCN Affine Accuracy

<img src="https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/CNN_Results/ADvsCN/Affine/AffineADvsCNAcc2-1.jpg" width="40%">

ADvsCN Affine Loss

<img src="https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/CNN_Results/ADvsCN/Affine/AffineADvsCNLoss2-1.jpg" width="40%">

The more agressive models showed better success with the most successful being SyNRA. Note that the most aggressive model, SyNAggro was not as effective, meaning the features of interest were normalized to the point of being unrecognizable.

ADvsCN ElasticSyN Accuracy

<img src="https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/CNN_Results/ADvsCN/SyNElastic/ElasticSyNADvsCNAcc2-1.jpg" width="40%">

ADvsCN ElasticSyN Loss

<img src="https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/CNN_Results/ADvsCN/SyNElastic/ElasticSyNADvsCNLoss2-1.jpg" width="40%">

ADvsCN SyNRA Accuracy

<img src="https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/CNN_Results/ADvsCN/SynRA/SyNRAADvsCNAcc2-1.jpg" width="40%">

ADvsCN SyNRA Loss

<img src="https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/CNN_Results/ADvsCN/SynRA/SyNRAADvsCNLoss2-1.jpg" width="40%">

ADvsCN SyNAggro Accuracy

<img src="https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/CNN_Results/ADvsCN/SyNAggro/SyNAggroADvsCNAcc2-1.jpg" width="40%">

ADvsCN SyNAggro Loss

<img src="https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/CNN_Results/ADvsCN/SyNAggro/SyNAggroADvsCNLoss2-1.jpg" width="40%">

While is difficult to visually see the difference between these last three methods, based on average accuracy and loss over train time, SyNRA produces the best results. In addition. While no models achieved particularly low losses, this provides an effective look into the difference between registration methods an their effect in training a neural network
