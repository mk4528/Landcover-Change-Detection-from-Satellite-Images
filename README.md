### 1. Create Training Slices

Modify `train_sample` in `utils.py` to specify training sample size, make sure the directories such as `naip_2017_tifs` contain the right .tif images. Then

```
python create_slices.py
```

which will create `train_sample` number of slices in the directory `utils.naip_train2013` and `utils.nlcd_train2013`

### 2. (Optional) Create NLCD Soft Labels

```
python convert_to_softlabel.py
```

It will read in the slices of NLCD labels, convert them to soft labels, and save to PNG images. 
Warning: it will take a relatively long time to run (8 mins for 2000 samples)



