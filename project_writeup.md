# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/original_dataset.png "Original Data Distribution"
[image3]: ./examples/tenprcnt_elimination.png "10% 0 steering angle data allowed"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

For learning, I used a 3 layer convlution with leaky relu activations, followed by three layers for fully connected layers. Also, first two layers of the model contains normalization and cropping functions.

I have chosen leaky relu activation considering the fact that changes between postive to negative steering angle values might lead some nodes to go offline and not learn. This thought was validated when both training and validation accuracy improved after switching to leaky relu activation.

![alt text][image1]

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers to apply regularization to help reducing overfitting (model.py lines 124, 126). 

Also dataset is divided into 80% training and 20% validation split in order to track training accuracy vis a vis validation accuracy. Finally, generator shuffles the data when creating batches for training which should also help reducing overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 158).

#### 4. Appropriate training data

During first trials model always appeared to under predict the absolut value of the steering angle. I attributed this fact to the overrepresentation of smaller (especially 0 angle) steering angle values in the data.

![alt text][image2]

For this reason I decided to eliminate approximately 10% of the data points that contained 0 steering angle value. This resulted in a much more balanced training data distribution and improved the ability of the model to turn sharp corners. On the other the hand the straight line performance of the model has decreased, resulting in vehicle to swerve from side to side although still within the limits of the drivable portion of the road. 

![alt text][image3]

Furthermore I have added flipped versions of each image and steering angle value to the dataset in order to compansate the counter clockwise bias of the track. 

Another data collection method I followed was to drive in the reversed direction, however this has not improved the performance beyond flipping the images. Thus I decided to ditch it altogether to keep the training time managable.
