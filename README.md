# vrnn2018
Project for Visual Recognition: Neural Networks class (https://www.mimuw.edu.pl/~bilinski/VRNN2018/).

Design doc [here](https://docs.google.com/document/d/1PpHswgc0P_6O-V_I3gins0K2ckUbiehw3eqtKNxmkE4/edit?usp=sharing).

## Prerequisites
### Python packages
* TensorFlow 1.6.0 for GPU
* PIL
* numpy
* progress
* tkinter
* keras
* matplotlib
### Data
* [Car dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). Download the [train images](http://imagenet.stanford.edu/internal/car196/cars_train.tgz) and place the `cars_train` directory in `data`.
* (Optional) [Cityscapes dataset](https://www.cityscapes-dataset.com/downloads/). You need to register and request access to the dataset, then download files `gtFine_trainvaltest.zip` and `leftImg8bit_trainvaltest.zip (11GB)`. Place directories `gtFine_trainvaltest` and `leftImg8bit_trainvaltest` to `data` directory.

Your `data` directory should look like this:
```
 ~/Desktop/VRNN/vrnn2018$ ls data 
cars_train  cars_train_labels  gtFine_trainvaltest  leftImg8bit_trainvaltest
```

## Execution
All Python scripts should be executed with the root repository directory as the working directory.

## Directory structure
* `segmentation_network` - code of the segmentation network and learning scripts
  * `learning_scripts`
    * `cars_fit.py` - trains the segmentation network using `cars_train` as input set and `cars_train_labels` as ground truths. Note that only those inputs are used, for which there is a labels file.
    * `cars_snowball.py` - runs in a loop, takes those images from `cars_train` which don't have labels yet. For each image, marks it on the image and displays the prediction. The user can accept (`y`), reject (`n`) or correct (`c`) the prediction. If you choose to correct the prediction, you will be presented with a very simple image editor (pen, eraser, change size), then you will be asked whether you want to save the edited labels.
    
    It is strongly recommended to run this script in PyCharm so that matplotlib windows don't pop up.
    * `cars_snowball_labels_editor.py`
    * `cityscapes_fit.py` - fits the network on Cityscapes dataset (reduced to only two labels: car & non-car).
    * `predictor.py` - takes a path to an image as argument and displays the labels (car & non-car)
    * `generate_cut.py` - for all label images in `data/cars_train_labels`, cuts the corresponding input image from `data/cars_train` and writes the effect to `data/cars_train_cut`. If directory `data/cars_train_cut` does not exist, creates it.
  * `cnn.py` - implementation of UNet for segmentation
  * `constants.py` - constants for UNet
* `segmentation_tmp` - trained network checkpoint. Please be careful when commiting any changes to this dir.

## Labels format
Label files are, like images, stored in .jpg files. Pixels containing cars should be black, pixels containing anything else should be white.

## Segmentation
For segmentation we are using UNet architecture. I have marked 50 first images from Cars dataset manually in gimp (using intelligent selection (`I`)), and 450 further random images using the snowball script.
