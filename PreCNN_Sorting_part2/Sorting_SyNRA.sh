  
cd /Output_ants_SyNRA/AD/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_SyNRA/DataSamples/ADvsCN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_SyNRA/DataSamples/ADvsMCI/
cd /Output_ants_SyNRA/CN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_SyNRA/DataSamples/ADvsCN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_SyNRA/DataSamples/MCIvsCN/
cd /Output_ants_SyNRA/MCI/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_SyNRA/DataSamples/ADvsMCI/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_SyNRA/DataSamples/MCIvsCN/
