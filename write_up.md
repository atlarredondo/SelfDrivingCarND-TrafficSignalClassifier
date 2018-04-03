# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image0]: https://github.com/atlarredondo/SelfDrivingCarND-TrafficSignalClassifier/tree/master/writeup_imgs/vis_train.png "Visualization"
[image1]: .https://github.com/atlarredondo/SelfDrivingCarND-TrafficSignalClassifier/tree/master/writeup_imgs/og.png "Orginial Image"
[image2]: https://github.com/atlarredondo/SelfDrivingCarND-TrafficSignalClassifier/tree/master/writeup_imgs/grey_norm.png "Grayscaling"
[image2]: https://github.com/atlarredondo/SelfDrivingCarND-TrafficSignalClassifier/tree/master/writeup_imgs/grey_norm_flipped.png "Grayscaling Flipped"
[image4]: https://github.com/atlarredondo/SelfDrivingCarND-TrafficSignalClassifier/tree/master/new_images/stop.jpg "Traffic Sign 1"
[image5]: https://github.com/atlarredondo/SelfDrivingCarND-TrafficSignalClassifier/tree/master/new_images/100.jpg "Traffic Sign 2"
[image6]: https://github.com/atlarredondo/SelfDrivingCarND-TrafficSignalClassifier/tree/master/new_images/construction.jpg "Traffic Sign 3"
[image7]: https://github.com/atlarredondo/SelfDrivingCarND-TrafficSignalClassifier/tree/master/new_images/right.jpg "Traffic Sign 4"
[image8]: https://github.com/atlarredondo/SelfDrivingCarND-TrafficSignalClassifier/tree/master/new_images/yield_to_cross.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.
I explored the frequency on the labels to understand which images were going to be the easiest and harder to classify. I expected an even distribution of images however, there are some signals that are orders of magnitude more frequent that others. This raises some concerns because the model might be bias to classify the signals that are more frequent. In addition, some of the signals that are similar(such as the speed limit ones) have also an uneven distribution, which might cause some misclassification. 

![alt text][image0]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

At the beginning I trained the Neural Network the the raw training set and a normalized version of the training set. 
The normalize version of the training set did not give me better accuracy so I exclude it. 

Since normalizing the dataset did not worked, I converted the dataset to grayscale. This gave me a lot better results and increased the validation accuracy significatively.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image1]

![alt text][image2]


After trying the grayscale dataset, I normalize it by subtracting and dividing it by 128. I added normalization in order to stabilize the feature distribution in the neural network and help it learn faster.

Finally, I tried to add enhanced data to the training dataset by flipping a subset of the images by 180 degrees. However, this did not help the neural network accuracy and I decided to not use it in my final implementation.

Here is an example of the enhanced image:
![alt text][image3]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale Image image   				| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Max pooling	      	| 2x2 stride,  outputs 14x14x10 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| Inception Layer:		|      						         			|
| 	Convolution 1x1	-> Convolution 3x3	| Both: 1x1 stride, same padding, outputs 10x10x8|
| 	Convolution 1x1 -> Convolution 5x5 | Both 1x1 stride, same padding, outputs 10x10x8|
| 	Convolution 1x1		| Both 1x1 stride, same padding, outputs 10x10x8|
| 	Max pooling -> Convolution 1x1  | Max pooling: 3x3, Both 1x1 stride, same padding, outputs 10x10x8  |  
|	Concat all previous layers|  output 10x10x32|
| Convolution 1x1					| 1x1 stride, valid padding, outputs 10x10x16 (to reduce number of weights )|
| Fully Connected layer	      	| Input 1600,  outputs 400 			|
| RELU      	| 	|
| Fully Connected layer	      	| Input 400,  outputs 84 			|
| RELU	      	| 			|
| Fully Connected layer	      	| Input 84,  outputs 43 			|
| SoftMax	      	| outputs 43 			|


Note: all convolutional layers are followed by a relu activation function.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam Optimizer with a learning rate of 0.001. 
Also, I set the batch size to 64 since I noticed that it reached a decent validation accuracy faster without overfitting. 
Finally, to give the model some more time to learn I set the number of epochs to 15.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Validation Accuracy = 0.967
* Train Accuracy = 0.995
* Test Accuracy = 0.938

Iterative process:
I started with the LeNet network architecture and I adapted it to the images with 3 color channels.
At first I decided to play around with adding more convolutional layers and increase the feature map. However, this only helped up to certain point and I was not able to reach more than 0.9 accuracy on the validation set.

I thought that the NN was not able to learn useful features because of the massive number of parameters and ended underfitting.
After this, I decided to reduce the number of convolutional layers to two and focus on the data preprocessing. 
I tried to normalized the raw color dataset but that made the accuracy wost. In addition, I enhanced the training dataset with a subset of images that were rotated 90 degrees. This made the learning slower on every epoch, and did not give me a better accuracy.

Later, I decided to go back to the model architecture and decided to try an Inception module. The Inception module that was implemented consisted of 4 different convolution blocks that were concatenated at the end.
This is the high level description of the Inception module blocks:
1. Conv 1x1 -> Conv 3x3 32 filters
2. Conv 1x1 -> 5x5 32 filters
2. Conv 1x1 32 filters
3. MaxPool 3x3 -> Conv 1x1 32 filters
Final concatenated layer was  10x10x128

This improve the validation accuracy considerably. However, the number of parameters that were passed into the first fully connected layer was around ~12k, which lead to underfitting and slow learning.

I tried to spread out the number of parameters, by adding additional fully connected layers that will decrease the feature representation at each layer, but that did not helped. 
To regularize the network I added dropout to the layers with the most parameters. Adding dropout helped on accuracy, however it was not reducing the total number of parameters or preventing underfitting.

Finally, I tried to add an additional 1x1 convolutional layer to reduce the number of filters from 128 to 32 and added additional dropout layers to some fully connected layers. This was successful. The accuracy increased to over 0.93 at times.

At that moment, I understood that the massive amount of parameters was the problem, so I decided to decrease the inception blocks filters to 8 and the final 1x1 filter to 16. With this architecture I was able to get from 0.93-0.94 with the raw dataset. 

As a final step I went back to the data preprocessing and converted the images to grayscale and normalize it. This was the best model and gave me accuracy of 0.958 on the validation set. I tried to enhanced this dataset again with a subset of rotated images, but that actually harmed the model accuracy.

If a well known architecture was chosen:
I decided to try an inception module from the beginning because I wanted to experiment with its powerful ability to combine features. Also, I found interesting that a network can learn better just by parallelizing the network computational graph.
I believed that an inception module was going to be helpful on the traffic sign classifier because of its previous success on similar classification problems.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The 100 km/hr sign might be hard to classify due to the slight distortion on the shape of the picture which makes the sign look like an oval. In addition, the sign is not perfectly centered in the image.

I think that the Stop and Turn right signs are the easiest to classify due to their unique sign shapes and content.

The yield sign also can be easily classified because of its flipped triangular shape, however there might be signs with similar shapes that can cause the model to misclassify the image to the others. 

Finally, the road work sign can be misclassified because of the human figure in the image and its similarities with other signs such as the children crossing and the pedestrian signs. 


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| 100 km/h     			| 50km/h										|
| Road Work			| Bumpy road											|
| Turn right ahead     	| Turn right ahead					 				|
| Yield		| Yield      							|

The model was able to classify 3 out of the 5 images, giving us an accuracy of 60% which is not bad for images that it has not seen before. It is interesting to see how the 100km/hr sign got misclassified with the 50km/h, both of these signs have pretty much the same representation with the exception of the speed digits. As a follow up step, it will be great to add a variety of enhanced images for the categories in which the model have a harder time differentiating. 

The stop and turn right sign were correctly classified as expected and I believe it's due to its uniqueness.
On the other hand, road work was misclassified to Bumpy road which looks so similar at first sight. Again, this seems to be a misclassification due to the strong similarity of the image with another class. 

The accuracy is not even close to the accuracy of the testing set(93.8%), however in this case we had an small amount of images and I believe this exercise can give us a good insight into the images that are harder to classify.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 45th cell of the Ipython notebook.

For the first image the model was able to classify the Stop sign correctly with a probability of 9.57. The next top probabilities were Priority road and Turn right. I believe that the reason Priority road signal pop out is because of the solid red colors on the sign, however I am not completely sure how the stop sign features are related to the turn right ahead features.
Correct Label: Stop, 14

| Probability         	|     Prediction				|
|:---------------------:|:---------------------------------------------:|
| 9.57 | Stop |
| 6.04 | Priority road |
| 4.62 | Turn right ahead |
| 3.69 | Speed limit (80km/h) |
| 3.55 | No entry |


The second image got misclassified by the model however, the first top probability is also a speed limit sign which demonstrate that the model was able to capture the speed limit features. The next top probabilities also share similar characteristics such as circular shapes and numbers. Finally, the model was able to identify that the image was similar to the 100km/h sign and it appeared as the fifth biggest probability. 

Correct Label: Speed limit (100km/h), 7

| Probability         	|     Prediction				|
|:---------------------:|:---------------------------------------------:|
| 5.39 | Speed limit (50km/h) |
| 3.87 | No passing for vehicles over 3.5 metric tons |
| 3.38 | Roundabout mandatory |
| 3.20 | Go straight or left |
| 2.19 | Speed limit (100km/h) |


The model misclassified the Road work sign, and gave a Bumpy road top prediction instead. The second prediction was the Road work sign. I think in this case the model was not able to differentiate correctly between the Bumpy Road and the Road work because of the similarities in its features. As an improvement, I would like to have more examples from either new images or enhanced images that can allow the model to differentiate from these hard signals.

Correct Label: Road work, 25

| Probability         	|     Prediction				|
|:---------------------:|:---------------------------------------------:|
| 10.10 | Bumpy road |
| 8.74 | Road work |
| 4.12 | Bicycles crossing |
| 2.46 | Traffic signals |
| 1.39 | Road narrows on the right |

The Turn Right signal was correctly classified by the model with a 9.86 probability. In this case the model did not had a problem classifying this image and the other 4 probabilities are 3 to 5 times smaller than the top probability. I this is due to the uniqueness of the Turn Right signal, such as the shapes and contents. 

Correct Label: Turn right ahead, 33

| Probability         	|     Prediction				|
|:---------------------:|:---------------------------------------------:|
| 9.86 | Turn right ahead |
| 3.14 | Yield |
| 2.16 | Double curve |
| 2.15 | Keep left |
| 1.36 | Speed limit (50km/h) |

The model also did a great job on classifying the Yield sign without a problem. In addition, the other top 4 probabilities are significantly smaller. I believe this is due to the unique inverse triangular shape of the yield sign and the lack of content in the signal
Correct Label: Yield, 13

| Probability         	|     Prediction				|
|:---------------------:|:---------------------------------------------:|
| 10.91 | Yield |
| 3.13 | Speed limit (50km/h) |
| 2.36 | No vehicles |
| 2.33 | Speed limit (30km/h) |
| 2.21 | Priority road |
