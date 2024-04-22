# DL-individual

## Goal
This project aims to attempt to remove 'good lighting' as a requirement for body-based computer vision gaming, particularly on regular laptops where the user does not have access to special cameras. I present a sports-based charades game that goes beyond the need for lighting. 

## Project Structure

### `data`
Contains the raw dataframe for the images as well as the processed/grouped images and the normalized DataFrame used for classification.

### `app`
Includes scripts to build and run the user interface of the application. Please not that because I have two very large model weights, I could not upload those .pth files to github. The demo won't evaluate without them, but it is possible to run the scripts and save them on your own local machine.

### `notebooks`
This directory houses Jupyter notebooks for:
- Classical SVM approach for classification of the keypoints
- Manually converting the DataFrame into COCO Format acceptable for training Faster RCNN Detectron2
- Naive approach without training for the Detectron2 FasterRCNN
- Building the NN out for classification
- Notebook for the Detectron2 pose detection
- Notebook for processing the points for my attempt at comparing each point to point in classification
- Notebook for preprocessing the keypoints prior to putting them in for classification.

### `final models`
This directory contains the weights for the GAN and classification models. I could not include the pose detection weights in github given their large size.

### `scripts`
Scripts in this folder include:
- `GAN.py`: Implements the GAN model.
- `poseclassification.py`: Script to take in the normalized keypoints and classify them as one of the available sports in the group of classes.
- `posedetection.py`: Script to train FasterRCNN to detect the keypoints in the image.

### `requirements.txt`
Lists all the necessary Python packages and libraries required to run the project.


