import os

## Renaming of Files

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
