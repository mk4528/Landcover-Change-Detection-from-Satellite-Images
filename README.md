## Intro

A repo developed in Keras with a purpose to train image segmentation models on NAIP/NLCD pairs

### Prerequisites

1. Environment: tensorflow
2. GPU-enabled device
3. Disk: minimum 10GB
4. Memory: minimum 16GB

### Capabilities

* Models: FCN, U-Net
* Labeling: hard-labeling, Soft-labeling
* Data: NAIP/NLCD pairs

## Steps to Reproduce

### 1. Download .tif Images

Configure:
* `downloadtif.training_both`: path to csv file
* `downloadtif.img_year`: 2013 or 2017
* `downloadtif.num`: number of .tif images to download
* `downloadtif.naip_output_dir`: output directory to save NAIP .tif images
* `downloadtif.nlcd_output_dir`: output directory to save NLCD .tif images

Run:
```
python downloadtif.py
```

### 2. Create Training Slices

Configure:
* comment out `create_slices.rerun()` if do not want to re-create all directories and sub-directories
* `utils.naip_train2013`, `utils.nlcd_train2013`: directories to save patches of NAIP/NLCD images
* `utils.train_naip_nlcd_2013`: path to csv
* `utils.train_stride`: stride to slide through NAIP/NLCD images
* `utils.train_sample`: how many slices in total to create
* `utils.train_input_size`: size of smaller patches

Run:
```
python create_slices.py
```

### 2. (Optional) Create NLCD Soft Labels

Configure:
* `utils.nlcd_train2013_soft`: directory to save patches of modified NLCD images - different from the patches of original NLCD images
* make sure `utils.nlcd_train2013` is where you saved previous patches of NLCD images

Run:
```
python convert_to_softlabel.py
```

### 3. Train

Configure:
* `train.py` L50-52: directories to save checkpoints
* `create_training_dataset.py`: comment or un-comment based on whether to train using soft-labeled NLCD patches or hard-labeled NLCD patches

Run:
```
python train.py
```

