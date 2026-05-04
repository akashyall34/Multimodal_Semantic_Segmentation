
# README Template

This template should be filled out by the student and included as `README.txt` in their submission. All sections must be completed. Delete any instructions in parentheses.

------------------------------------------------------------

## 1. Project Overview

Project Title: Multimodal Semantic Segmentation

Model Type:
Transformer

Objective:
Classification

Dataset Used:
MCubeS dataset | Link: https://drive.google.com/file/d/14egTCyC0Pampb7imrXVwaDRffHN7FZxh/view

Expected test evaluation for sanity check: mIoU = 25.99% 

------------------------------------------------------------

## 2. Repository Structure

List the structure of your project directory below. Add short descriptions if needed.

```
akash_project2_code/
  train.ipynb                         # notebook to run training, optuna-led hyperparameter tuning, and testing
  multimodal_dataset/                 # dataset containing all modalities and there is a script text file with prewritten train/val/test split
    SSGT4MS
    SS
    polL_dolp
    polL_color
    polL_aolp_sin
    polL_aolp_cos
    NIR_warped_mask
    NIR_warped
    list_folder
    GT
  quadwaterfall/                      # folder containing train.py, architecture.py, etc.
    optuna_results/                   # folder which contains saved results from optuna-led hyperparameter tuning
    checkpoints/                      # folder containing epoch 50, epoch 100, and best model .pt files
    test_integration.py               # file required to run a quick end to end pipeline check test to ensure data goes through architecture and is good for training
    train.py                          # file required to perform training as it contains metric calculation and args for training function
    test.py                           # file required to run evaluation of best model checkpoint on test set
    hyperparameter_search.py          # file required to run optuna-led hyperparameter tuning as it contains hyperparameter search implementation
    dataset.py                        # file required for data loaders of dataset to model for training
    config.py                         # file required containing quantitative information to perform augmentations, set training parameters, etc.
    architecture.py                   # file containing QuadWaterfall architecture implementation influenced from FuseForm and WTPose paper architectures
  README_akash.txt                    # file containing instructions about code directory, files, how to run, etc.
```

------------------------------------------------------------

## 3. Dataset (Choose ONE of the 3 options)

### OPTION A — PUBLIC DATASET SPLITS
Dataset Link:
https://drive.google.com/file/d/14egTCyC0Pampb7imrXVwaDRffHN7FZxh/view

For ease of use, here is the Box link as well: https://usf.box.com/s/i9qukqzyjtg1wa4074qbeih5bprigwgv

Where to place the downloaded dataset:
```
akash_project2_code/
  multimodal_dataset/ # this entire folder is available at the box link or if you download from the google drive link just few lines above and then unzip, you can place in the root directory of the code
```

The files in quadwaterfall/ directory take care of reading the predefined splits so all you need is to place the multimodal_dataset/ folder in the root directory of the code.

------------------------------------------------------------

## 4. Model Checkpoint

Box Link to Best Model Checkpoint:
https://usf.box.com/s/i9qukqzyjtg1wa4074qbeih5bprigwgv

Give access to:
yusun@usf.edu, kandiyana@usf.edu

Where to place the checkpoint after downloading:
```
akash_project2_code/
  quadwaterfall/
    checkpoints/
      best_model.pt   # the .pt file accessible from the box link must be placed in this path: akash_project2_code/quadwaterfall/checkpoints/
```

------------------------------------------------------------

## 5. Requirements (Dependencies)

Python Version:
3.12 # default version available in google colab

How to install all dependencies (e.g. requirements.txt):

Not necessary at all because the train.ipynb uses default python version available in colab and there are "install..." commands in the cells
So all that matters for requirements is to just open train.ipynb file and follow markdown 1's instructions and run all cells subsequently

------------------------------------------------------------

## 6. Running the Test Script

Command to run testing:
```
!python test.py --checkpoint ./checkpoints/best_model.pt \
               --data-root ../multimodal_dataset \
               --device cuda \
               --batch-size 8
```

------------------------------------------------------------

## 7. Running the Training Script

Command to run training:
```
!python train.py \
  --batch-size 8 \
  --num-epochs 100 \
  --lr-init 1.0e-06 \
  --lr-max 1.4e-05 \
  --lr-min 8.69e-10 \
  --warmup-epochs 15 \
  --weight-decay 5.164e-03 \
  --poly-power 0.972299 \
  --dropout-p 0.079453 \
  --device cuda
```

------------------------------------------------------------

## 8. Submission Checklist

- [A] Dataset provided using Option A, B, or C and placed correctly.
- [X] Model checkpoint linked and instructions for placement included.
- [X] `requirements.txt` generated and Python version specified.
- [X] Test command works.
- [X] Train command works.

------------------------------------------------------------
