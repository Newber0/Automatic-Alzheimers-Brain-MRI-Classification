# Overview of Methods, from Data Acquisition to Output of Trained Model

All code associated with this project can be found in this repository separated by function, but this document will act as an overarching summary of all work completed. While the data will not be provided here, it can be acquired from the [ADNI Database](http://adni.loni.usc.edu/) with permission. Upon download of the data a CSV with identifying information for each image is also made avaliable. This can be found [here](https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/Data_Index.csv) in the case of this project.

[Obtain data from ADNI Website](#Obtain_data_from_ADNI_Website)

[Preprocessing of File Structure](#Preprocessing_of_File_Structure)

[Brain Extraction using ROBEX](#Brain_Extraction_using_ROBEX)

[Registration to Template using ANTs](#Registration_to_Template_using_ANTs)

[Segmentation into GM, WM, and CSF using ANTs](#Segmentation_into_GM,_WM,_and_CSF_using_ANTs)

[Separation of Data prior to CNN Entry](#Separation_of_Data_prior_to_CNN_Entry)

[Construction of CNN](#Construction_of_CNN)

[Summary of Output](#Summary_of_Output)


# <a name="Obtain_data_from_ADNI_Website"></a>Obtain data from ADNI Website

We obtained our data from the Alzheimer’s Disease Neuroimaging Initiative (ADNI) database. We selected T1 weighted 3T images including AD, MCI, and CN labelled data. All available data in this category was taken from the ADNI Database. A CSV file containing all identifying information for each image was provided upon download of the data. The data was downloaded from different sources, so these files were combined into a master excel file which can be found [here](https://github.com/Newber0/Automatic-Alzheimers-Brain-MRI-Classification/blob/main/Data_Index.csv). 

Once downloaded the data was organized by date, then 

# <a name="Preprocessing_of_File_Structure"></a>Preprocessing of File Structure

The data was organized by date, patient, and preprocessing types. The naming convention for these files included all this information as well. All data was removed and placed into one file to reduce complicated file parsing. 

Each file contained all identifying information for that file including the file structure as well as a unique image ID. All files were renamed to include only the image ID. This was accomplished with the simple line of code below.

```
import os

filepath = '/Data_Directory/'
# '/Data_Directory/' is wherever the data is stored.

for f in os.listdir(filepath):
    file_name, file_ext = os.path.splitext(f)
    ADN, REDUNDANT, ID = file_name.split('I')

    ID = ID.strip()
    file_ext = file_ext.strip()

    new_name = ('{}{}'.format(ID,file_ext))
    os.rename((filepath + f), (filepath + new_name))
```

# <a name="Brain_Extraction_using_ROBEX"></a>Brain Extraction using ROBEX and further Preprocessing

The tool used for this step was the Robust Brain Extraction package was used which can be found [here](https://www.nitrc.org/projects/robex). This removes the tissue surrounding and including the skull. Isolating exclusively brain tissue in an effective manner dramatically increase the accuracy of the neural network. The code for this process can be found below.
```

```
Next the images were reoriented to the standard position. This was accomplished using the fsl package found [here](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL), and the following code accomplishes this.

# <a name="Registration_to_Template_using_ANTs"></a>Registration to Template using ANTs

# <a name="Segmentation_into_GM,_WM,_and_CSF_using_ANTs"></a>Segmentation into GM, WM, and CSF using ANTs

# <a name="Separation_of_Data_prior_to_CNN_Entry"></a>Separation of Data prior to CNN Entry

# <a name="Construction_of_CNN"></a>Construction of CNN

# <a name="Summary_of_Output"></a>Summary of Output

