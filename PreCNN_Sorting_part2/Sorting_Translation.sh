# Moving AD files
cd /Output_ants_Translation/AD/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Translation/ADvsCN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Translation/ADvsMCI/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Translation/ADvsMCIvsCN/

# Moving CN files
cd /Output_ants_Translation/CN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Translation/ADvsCN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Translation/MCIvsCN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Translation/ADvsMCIvsCN/
# Moving MCI files
cd /Output_ants_Translation/MCI/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Translation/ADvsMCI/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Translation/MCIvsCN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_Translation/ADvsMCIvsCN/
