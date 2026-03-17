# U-Net Based Image Segmentation for Plant Species Recognition
## Introduction

This is the implementation of the image segmentation used in my master's thesis titled "Applying Semantic Segmentation and Structure-from-Motion to Monitor Flora in Cliff Ecosystems" [https://studenttheses.uu.nl/handle/20.500.12932/50960]. The segmentation model is based on research by Badrouss et al. [https://link.springer.com/article/10.1007/s40808-024-02222-w] using a U-Net architecture with a ResNet50 backbone.

## Dataset

For this research, two datasets were created of drone images recorded in Meia Velha and Cabo Espichel in Portugal. The former contains images of Opuntius ficus-indica and their corresponding image masks, while the latter contains images of Euphorbia pedroi.

Cabo Espichel dataset: [https://drive.proton.me/urls/YJHKEW73FW#zpGglzov4Aje]  
Meia Velha dataset: [https://drive.proton.me/urls/QNPYYG85X0#gXBzc3WM7Wxq]

## Prerequisites

To run this program, it is necessary to first install the following software:
1. Visual Studio Code [https://code.visualstudio.com/]
2. In Visual Studio Code, install the Python for Visual Studio Code extension [https://marketplace.visualstudio.com/items?itemName=ms-python.python]
3. Anaconda or Miniconda [https://docs.conda.io/projects/conda/en/stable/user-guide/install/windows.html] 
4. CUDA [https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/]. We used CUDA 11.8.

## Setup

This project uses a Conda environment called `pytorch-unet`. A Conda environment is an isolated workspace that contains the correct Python version and allr equired packages for this project. Follow the following steps to activate it using the terminal inside Visual Studio Code.

- Open the project in VS Code  
  1. Start Visual Studio Code
  2. Click File -> Open Folder
  3. Select the folder containing this project
- Open the VS Code Terminal  
  1. In the top menu, click Terminal -> New Terminal
  2. A terminal panel will open at the bottom of the VS Code window

  You should now see a command line where you can type commands
- Create Conda Environment (First time only)  
  The repository contains a file called `environment.yml`. This file lists all required Python packages. Run the following command in the terminal:
  ```
  conda env create -f environment.yml
  ```
- Activate the Conda Environment  
  ```
  conda activate pytorch-unet
  ```
  If the environment activates successfully, you should see (pytorch-unet) appear at the beginning of the command line, for example:
  ```
  (pytorch-unet) C:\Users\YourName\project-folder>
  ```
  This indicates that the environment is active.  
  If VS Code asks you to select a Python Interpreter, choose the one that contains `pytorch-unet`.

## Folder Structure

```none
├── data
│   ├── dataset_name (used for training the model, images and masks should contain one mask for each image, where both have the same file name)
│   │   ├── images
│   │   ├── masks
│   ├── inference_images (contains only images for inference -> images you want to get the predictions from)
│   ├── temp (contains folders, do not touch/use this folder)
├── output
│   ├── dataset_name
│   │   ├── checkpoints (Trained model weights)
│   │   ├── iou_rankings (Raking of the best and worst performing tiles, useful for visualizing which tiles the network performs better/worse on)
│   │   ├── loss_curves (Train and validation loss graph)
│   │   ├── model_metrics (Metrics like IoU, precision, recall)
│   ├── inference
│   │   ├── checkpoint_name
│   │   │   ├── masks (output as binary mask)
│   │   │   ├── probs (output as float probabilities)
├── scripts
├── config.py
```

When training the network, the dataset automatically gets restructured to fit the programs needs. If you wish to reset this restructuring you can do this with: 
```
bash reset_data.sh
```

All settings are found in config.py

## General Use

You normally run three separate steps. The first step is model training, where the network learns how to make predictions based on the training data. The second step is the test loop, where the performance of the trained model is evaluated based on unseen data. The final step is the inference, where you feed the trained model images that you want predictions of. In `train_unet.sh` you can tell the program to run the train and/or test loops with `--train` and `--test`.

## Model Training

To train the model, start by opening `train_unet.sh` and replace `example_training_data` with the name of your dataset folder. It is recommended to go through the settings in config.py under TrainConfig to review whether the settings are correct. Some important settings will be highlighted here:
- `augmentation_method` allows you to chose between 3 augmentation setups, or to do no data augmentation at all. The default selection was our best performing version.
- `use_test_split` is used to chose between what sources to use for the training/test split. `none` simply uses all images for training, it should only be used when you don't run the test loop. `directory` allows you to appoint a separate folder with test data. All images in your dataset folder under data will be used as training data, images in the appointed folder are used as test data. `csv` finds a csv for your current dataset, if none exists it will create a new one. `force` creates a new test split csv, whether one already exists or not.
- `test_split_method` determines whether the train/test split is made according to the spatial clustering of images in the dataset or if it is done randomly.
- `pretraining` allows you to pass a checkpoint (trained model weights) to the network for pre-training. It overwrites load_IMAGENET1K_V1, which are a separate set of pre-trained weights. 

You can then run the script using:
```
bash train_unet.sh
```

This first pre-processes your dataset and makes sure that the images and masks are separated into tiles. It then creates a train/test split if this is required.  
If `--train` is enabled, it goes through the training loop until the training is finished. It then saves the checkpoint and loss curves in `output`.
If `--test` is enabled, it immediatly enters the test loop to evaluate the model's performance. This outputs the best and worst tile predictions and the model's performance metrics to `output`.   
Note that it is possible to run either only `--train` or `--test`. If you only want to run `--test`, make sure `test_only_checkpoint` is correctly specified in `config.py`.  
  
All output is saved using the same naming convention: `<file_name>_<dataset_name>_<num_epochs>_<augmentation_method>_<pre_training>`.

## Inference

When you have a trained model, you can run inference on the images in `data/inference_images` by running the following command:
```
bash run_inference.sh
```

This outputs a binary mask and a probability mask for each image.
