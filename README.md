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

The original scope of this project was to classify MRI data as either AD or CN, but when the data was downloaded there was a high rate of images with MCI and the decision to include this class was made. This complicated the process because binary classification is far easier than multiclass classification. As a result we decided to run multiple networks for each registration method, 3 binary networks, (comparing AD to CN, MCI to CN, and AD to MCI) and one multiclass network (AD vs CN vs MCI). Before building the network, the data was split into 3 files containing AD, MCI, and CN respectively, and then from these files, copies were taken to create the following 4 files; ADvsCN, MCIvsCN, ADvsMCI, and ADvsMCIvsCN. Each network could then sample from the correct file instead of having to parse through the full set for each network. Additionally, to ensure consistency, 100 samples from each class were taken. Below is the code that achieved this.

```

```

# <a name="Construction_of_CNN"></a>Construction of CNN

# <a name="Summary_of_Output"></a>Summary of Output

