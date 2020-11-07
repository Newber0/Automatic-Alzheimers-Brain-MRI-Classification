# Moving AD files
cd /Output_ants_Translation/AD/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Translation/Datasamples/ADvsCN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Translation/Datasamples/ADvsMCI/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Translation/Datasamples/ADvsMCIvsCN/

# Moving CN files
cd /Output_ants_Translation/CN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Translation/Datasamples/ADvsCN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Translation/Datasamples/MCIvsCN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Translation/Datasamples/ADvsMCIvsCN/
# Moving MCI files
cd /Output_ants_Translation/MCI/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Translation/Datasamples/ADvsMCI/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Translation/Datasamples/MCIvsCN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Translation/Datasamples/ADvsMCIvsCN/
