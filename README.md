# Automatic Landcover Change Detection and Classification from Satellite Images

## Project Objective

Establish a semantic segmentation model in a specific study area describing land cover change between two consecutive time points (such as landcover in year 2022 vs. year 2017). Detect the loss or gain of “tree canopy” land cover class.


## Training and Evaluation

In the following order, you can run codes for training and evaluating models, including the ensemble model, in `Main_verXX.ipynb` on Google Colab with an A-100 40GB GPU. If you want to run them on another environment including GPU type, you need to modify codes in the 4th and 5th cells mainly (also one in the 7th cell if needed).

1. Change the path of `MAIN_DIRECTORY` in the 5th cell

2. Save the training_set_naip_nlcd_both.csv under your `MAIN_DIRECTORY`

3. Make your `test_hr_geojson` folder under your `MAIN_DIRECTORY` and save the 8 geojson files in the `test_hr_geojson` folder on this GitHub in your `test_hr_geojson` folder

4. Set `RENEW_TRAIN_DATASET_FLAG`, `RENEW_TEST_DATASET_FLAG`, and `CODALAB_NAIP_DOWNLOAD_FLAG = True` in the 5th cell at the first running (After the first running, you can choose them as you want)

5. Do a. or b.
    a. If you want to train and evaluate a single model not an ensemble model, including the chained model, set `TRAIN_ENSEMBLE_FLAG = False`

    b. If you want to train an ensemble model, set `TRAIN_ENSEMBLE_FLAG = True`, `ENSEMBLE_ID = 0`, and `EVALUATE_ENSEMBLE_FLAG = False`

6. Modify other parameters in the 5th and 7th cells as you want (You do not need to set proper NUM_ENSEMBLE when training an ensemble model)

7. Run all the codes

8. (Only for ensemble model) Repeat 4 (if needed) and 7 for `i = 1` to `(M - 1)` with `ENSEMBLE_ID = i`. M denotes the number of ensemble models you want to train

9. (Only for ensemble model) Set `TRAIN_ENSEMBLE_FLAG = False`, `EVALUATE_ENSEMBLE_FLAG = True`, and `NUM_ENSEMBLE = M`

10. (Only for ensemble model) Run all the codes

You can get result images in your `OUTPUT_IMAGE_DIRECTORY` and a zip file for submission to Codalab to calculate test IoUs in your `CODALAB_SUBMISSION_ZIP_DIRECTORY`. The result files are named `OUTPUT_FILES_NAME` for single models and `OUTPUT_FILES_NAME + '_ENS_EVAL_M' + str(NUM_ENSEMBLE)` for ensemble models. Also, `OUTPUT_FILES_NAME + '_ENS_EVAL_M' + str(NUM_ENSEMBLE) + '_ex'` denotes the files are for the model of `ENSEMBLE_ID = 0`

## Incorporating Dynamic World Labels

The code resides in `./DW` directory. The instruction to run the codes are as follows: 

1. Train 5-layer FCN model on each data below augmented w.r.t. brightness (Input: Dynamic World labels (probability bands), coors.txt, training_set_naip_nlcd_2017.csv, NAIP test images, high-resolution test labels, Output: torch weights of trained model as .pt file)

    1. `./DW/DAug_NLCDonly_git.ipynb`: NLCD 2016
    2. `./DW/DAug_DWonly_softlabel_git.ipynb`: DW2017 (hard label)
    3. `./DW/DAug_DWonly_probs_git.ipynb`: DW2017 (soft label)

The links need to be modified first. You need Dynamic World labels (probability bands) downloaded and `coors.txt` outputted by the other notebooks for Google Earth Engine. They can be run using T4, P100, V100 or A100 GPU with high memory setting on Google Colab

2. Make submission zip file to Codalab (Input: 5-layer FCN model trained (not limited to above), Output: Submission zip file to Codalab)

The notebook is `./DW/model2Codalab_git.ipynb`

You need to modify the links to the model you trained. You can specify `SPL_WH`, i.e. the input height/width of the model (it does not have to be always 388. This is important because the input size of U-net is restricted to be multiples of eight). You can also change the model class. This can be run using T4, P100, V100 or A100 GPU with high memory setting on Google Colab

## Contributors

### Authors (Team Captain: Masataka Koga):
+ Ashkan Bozorgzad (ab5243)
+ Hari Prasad Renganathan (hr2514)
+ Karveandhan Palanisamy (kp2941)
+ Masataka Koga (mk4528)
+ Yewen Zhou (yz4175)
+ Yuki Ikeda (yi2220)

###  Sponsor/Mentor:
- Dr. Saba Rahimi from J.P. Morgan

###  CA:
- Katie Jooyoung Kim

###  Instructor:
- Vivian S. Zhang
