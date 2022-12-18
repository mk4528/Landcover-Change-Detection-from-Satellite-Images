# Downloading dynamic world labels corresponding to each NLCD label from Google Earth Engine python API

#### Get_Coords_of_NLCD.ipynb
- It is made to run on Google Colabolatory
  https://colab.research.google.com/drive/1yG2grTW85cl2E2nsRiaMoEC3pmAIoPkP?usp=sharing
- You need to upload [training_set_naip_nlcd_both.csv](https://github.com/calebrob6/dfc2021-msd-baseline/blob/master/data/splits/training_set_naip_nlcd_both.csv) to Google drive mounted by Google Colab
- The output is list `coors` formatted as coors.txt which is to be saved by pickle. This contains coordinates of NLCD labels. We are going to download dynamic world labels for rectangular regions of interest with these coordinates using the next notebook.

####  DL_DW_to_GDrive_bandonly_or_probbands.ipynb
- It is made to run on your local environment to export dynamic world labels from GEE to Google Drive using GEE python API
- It uses coors.txt outputted from the above notebook as input
- Though submitting tasks might not take for a long time, it takes hours to export all 2250 dynamic world labels from GEE