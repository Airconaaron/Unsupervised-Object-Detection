# Introduction
This repo is a collection of code used for a Capstone Project at Yale-NUS College.
The objective of this project was to create a methodology for an unsupervised object detection algorith via reinforcement learning. 
Unsupervised here is used to describe that neither the bounding boxes nor classes labels from Pascal VOC data were used in training our algorithm. 
The only labelled data used were the images and classes from ImageNet in order to train our confidence classifier.

# Basics


# Installation
* First download Conda from https://anaconda.org/
* Create our environment from 
``` 
conda env create -f environment.yml
```
* Go into the scripts folder and change the directories to point to your locations for Pascal VOC data
* Run scripts via `python image_zooms_testing`

# Acknowledgements
Code was extensively modified from [this repo](https://imatge-upc.github.io/detection-2016-nipsws/). 