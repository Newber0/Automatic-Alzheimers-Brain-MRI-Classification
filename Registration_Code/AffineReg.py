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
