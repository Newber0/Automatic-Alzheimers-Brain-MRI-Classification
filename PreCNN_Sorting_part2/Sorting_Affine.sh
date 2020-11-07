cd /Output_ants_Affine/AD/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Affine/DataSamples/ADvsCN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Affine/DataSamples/ADvsMCI/
cd /Output_ants_Affine/CN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Affine/DataSamples/ADvsCN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Affine/DataSamples/MCIvsCN/
cd /Output_ants_Affine/MCI/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Affine/DataSamples/ADvsMCI/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Affine/DataSamples/MCIvsCN/
