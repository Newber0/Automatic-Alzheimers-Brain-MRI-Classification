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
  shutil.move('/Output_ants_Translation/Greymatter/' + i + '.nii', '/Output_ants_Translation/AD/')
  continue;

for i in CN:
  shutil.move('/Output_ants_Translation/Greymatter/' + i + '.nii', '/Output_ants_Translation/CN/')
  continue;

for i in MCI:
  shutil.move('/Output_ants_Translation/Greymatter/' + i + '.nii', '/Output_ants_Translation/MCI/')
  continue;
