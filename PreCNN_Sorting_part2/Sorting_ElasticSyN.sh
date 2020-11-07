# Moving AD files
cd /Output_ants_ElasticSyN/AD/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_ElasticSyN/Datasamples/ADvsCN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_ElasticSyN/Datasamples/ADvsMCI/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_ElasticSyN/Datasamples/ADvsMCIvsCN/

# Moving CN files
cd /Output_ants_ElasticSyN/CN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_ElasticSyN/Datasamples/ADvsCN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_ElasticSyN/Datasamples/MCIvsCN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_ElasticSyN/Datasamples/ADvsMCIvsCN/

# Moving MCI files
cd /Output_ants_ElasticSyN/MCI/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_ElasticSyN/Datasamples/ADvsMCI/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_ElasticSyN/Datasamples/MCIvsCN/
find . -maxdepth 1 -type f | head -100 | xargs cp -t /Output_ants_ElasticSyN/Datasamples/ADvsMCIvsCN/
