# **Behavioral Cloning** 

## Below is an overview of my solution for the Udacity Self Driveing Car Engineer Term 1 Project: Behavioral Cloning Project - P3.

---

**Behavioral Cloning Project**

The steps completed for this project are as follows:
* Use the simulator to collect data of good driving behavior
* Build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
* See the first recorded successful lap here: https://www.youtube.com/watch?v=pwmWeQYJzgk


[//]: # (Image References)

[image1]: ./examples/center_2017_04_16_11_31_21_488.jpg "Example output from training"
[image2]: ./examples/center_2017_04_16_11_31_21_488_crop.jpg "Cropped"
[image3]: ./examples/center_2017_04_16_18_50_25_362.jpg "Recovery Image 1"
[image4]: ./examples/center_2017_04_16_18_50_27_240.jpg "Recovery Image 2"
[image5]: ./examples/center_2017_04_16_18_50_28_037.jpg "Recovery Image 3"
[image6]: ./examples/center_2017_04_16_11_31_21_488.jpg "Normal Image"
[image7]: ./examples/center_2017_04_16_11_31_21_488_horizflip.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results (this file)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I tried a number of different models over the course of this project.  Starting with a simple flatten+dense 2 layer setup, I found that the driving performance was pretty poor.  Adding a single convolution layer improved the loss rate a bit, but didn't do much to the driving performance.  I then tried creating a more robust model similar to LeNet.  That took me down a path of very long training times, and trying to get the AWS GPU instance up and running so that I could actually work through the data.  I finally settled on an NVidia-like approach, with no pooling layers, but I kept a dropout and a softmax activation from our earlier exercises.  

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layers in order to reduce overfitting (model.py line 106).  In addition, the model was trained and validated on different data sets to ensure that the model was not overfitting (code line 30).  Lastly, the training data was shuffled to ensure that the model would not become overfit on the order of the images (line 80)

After training was complete, the model was tested by running it through the simulator and ensuring that the vehicle could stay on the track (see included video of successful track lap).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 115).  It worked to minimize the mean squared error loss, as defined in the model.compile() statement.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I first started by recording a full lap of the track, driving the center of the roadway.  This provided a marginal result, with the car clearly not knowing how to fix a problem once it occured.  I tried to record a lap swerving back and forth, but this introduced the behavior of swerving, rather than just fixing an indavertant drift.  I then started over, recording a full lap in the center of the track, followed by a series of recordings which started after the car was already heading off-track, recording only the steps needed to get back on-track.  Retraining against this data greatly improved the results.

This is the video recording of this training seesion, recorded by hand to show the scrolling angle information on the right:  https://www.youtube.com/watch?v=pwmWeQYJzgk

I then attempted to apply this model to the second track, with very poor results.  I duplicated my training recordings on the second track, with the intent of generalizing the model across both tracks.  However, I found that the different tracks did not work well together, resulting in a model which couldn't navigate either.  On further research, I realized that the turning angles are so different that co-training was causing the models to confuse how much turning should be expected.  I later attempted to solve this in a few ways I've documented below.

For details about how I created the training dataset from the images recorded during training driving sessions, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start small, and build layers until mean squared error loss was effectively minimized.  My first model was little more than a single convolution layer with a flatten and dense step to produce an output.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set and a low mean squared error on the validation set, suggesting underfitting.  By adding more convolution layers, I was able to improve the training and validation mse loss.

I then decided to try a convolution neural network model similar to the LeNet model, since the task is still working on image data, even though LeNet was originally tuned to classification tasks.  What I found was no real improvement in performance, but a significant increae in training time.

I then simplified the model again, towards an Nvidia model consisting of convolution layers with relu and 2x2 strides, and dense layers.  This seemed to improve training time significantly, and the driving performance seemed a bit better than before.  At this point, the car was making it all the way around the first track.

I tried this model against the second track, but the car only made it halfway up the first incline.  I suspected that the model may have been overfitting to the first track, so I decided to generalize it.

To combat the overfitting, I modified the model to include a dropout layer with 50% retention.  This seemed to help the loss, but the performance of the driving was no better.  I then introduced left/right flipping of each frame, to prevent the model from over-assuming a left-hand bias.  I then spent a good amount of time recording driving laps and corrective clips on the second track, and retrained the model on the combined dataset.  loss on this combined set reduced from 0.2560 to 0.0901 before increasing again, finishing at roughly 0.0930.  Testing this against the simulator resulted in very poor performance on both tracks.

I then took a step back from the second track and focused on just the first track.  I re-recorded all of the training data so I had a clean set, and retrained the model for 4 epochs (loss seemed to peak at this point during earlier training runs).  Thie test of this training run is the one I have included with the project, though it is not as clean as the original run; I'm not clear what changed between the two to introduce the stop and go behavior, or the increased swerving comparing to the initial runs.

At the end of the process, the vehicle is able to drive autonomously around track 1 without leaving the road.  Working on track 2 will take more time.

#### 2. Final Model Architecture

The final model includes (from line 92-113)
Cropping2D: to remove the top and bottom of the image, as well as a single pixel from both sides to account for a bug in Keras 1.2.1
Lambda: to normalize the data using the normalization function provided in the lectures
Convolution2D: 24 depth, 5x5 filter size, 2x2 subsampling size with RELU activation to provide first layer modeling
Convolution2D: 32 depth, 5x5 filter size, 2x2 subsampling size with RELU activation to provide second layer modeling
Convolution2D: 48 depth, 3x3 filter size with RELU activation to provide third layer modeling
Convolution2D: 64 depth, 3x3 filter size with RELU activation to provide fourth layer modeling
Convolution2D: 64 depth, 3x3 filter size with RELU activation to provide fifth layer modeling
Dropout: 50% hold rate to reduce overfitting
Flatten: to flatten
Dense: a Fully-connected layer of size 120
Dense: a Fully-connected layer of size 5
Softmax: A softmax activation
Dense: a Fully-connected layer of size 1 to finish the model

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct itself after having drifted from the center. These images show what a recovery looks like, starting from the left edge of a curve and recentering:

![alt text][image3]
![alt text][image4]
![alt text][image5]

I drove the track again in reverse, to provide a different perspective on the track, and to provide some additional right-handed turning input.

Then I repeated this process on track two in order to get more data points.  These datapoints, as mentioned above, seem to confuse the model, likely due to the significant streeing angle difference needed, as well as the addition of height.  The occlution of the road during downhill sections, as well as the road being cut off by the cropping during uphill sections likely impacted the performance when training against those images.

To augment the data sat, I also flipped images and angles thinking that this would help reduce overfitting to a left-turn bias. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 20,934 data points (without the track 2 input, 65,772 with the track 2 input). I then preprocessed this data by cropping the top and bottom to remove unneeded data, and I normalized it using the lambda layer described above.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

This training data was used for training the model. The ideal number of epochs was smaller than expected, requireing only 3 epochs before the loss leveled off or began to increase.  

#### 4. Track #2

After it became clear that track #2 presented a somewhat different problem than Track #1, I re-recorded training data for Track 2 into a folder by itself.  Keeping the Track 2 training data apart from the Track 1 data would, in theory, create a model which should work for just Track #2.  Having trained the model against just Track 2 data, however, I discovered that this was not the case.  Using the same model and roughly the same training data creation methods as for Track 1, the Track 2 model failed terribly.  Adding more training data helped, but the car continued to run directly into the street signs within feet of the start line.

Because of this poor performance, I looked back at the model itself, and decided to do two things: remove the cropping (in case it was cutting off important road information), and add two more convolutional layers, along with a pooling layer.  If I could build a model which would work on Track #2, I could then determine a good method for merging the two together.