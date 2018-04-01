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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

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

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

At the beginning I trained the Neural Network the the raw training set and a normalized version of the training set. 
The normalize version of the training set did not give me better accuracy so I exclude it. 

Since normalizing the dataset did not worked, I converted the dataset to greyscale. This gave me a lot better results and increased the validation accuracy significatevely.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]


After trying the greyscale dataset, I normalize it by substracting and diving it by 128. I added normalization in order to stabilize the feature distribution in the neural network and help it learn faster.

Finally, I tried to add enhaced data to the training dataset by flipping a subset of the images by 180 degrees. However, this did not help the neural network accuracy and I decided to not use it in my final implementation.

Here is an example of an original image and an augmented image:

![alt text][image3]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grescale Image image   				| 
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
Also, I set the batch size to 64 since I moticed that it reached a decent validaiton accuracy faster without overfitting. 
Finally, to give the model some more time to learn I set the number of epochs to 15.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.91
* validation set accuracy of 0.958
* test set accuracy of 0.993

Iterative process:
I started with the LeNet network architecture and I addapted it to the images with 3 color channels.
At first I decided to play around with adding more convolutional layers and increase the feature map. However, this only helped up to certain point and I was not able to reach more than 0.9 accuracy on the validation set.

I thought that the NN was not able to learn useful features because of the massive number of parameters and ended underfitting.
After this, I decided to reduce the number of convolutional layers to two and focus on the data preprocessing. 
I tried to normalized the raw color dataset but that made the accuracy wost. In addition, I enhanced the training dataset with a subset of images that were rotated 90 degrees. This made the learning slower on every epoch, and did not give me a better accuracy.

Later, I decided to go back to the model arquitecture and decided to try an Inception module. The Inception module that was implemented consisted of 4 different convolution blocks that were concatenated at the end.
This is the high level decription of the Inception module blocks:
1. Conv 1x1 -> Conv 3x3 32 filters
2. Conv 1x1 -> 5x5 32 filters
2. Conv 1x1 32 filters
3. MaxPool 3x3 -> Conv 1x1 32 filters
Final concatinated layer was  10x10x128

This improve the validation accuracy considerably. However, the number of parameters that were passed into the first fully conected layer was around ~12k, which lead to underfitting and slow learning.

I tried to spread out the number of parameters, by adding additional fully connected layers that will decrease the feture representation at each layer, but that did not helped. 
To regularize the network I added dropout to the layers with the most parameters. Adding dropout helped on accuracy, however it was not reducing the total number of parameters or preventing underfitting.

Finally, I tried to add an additional 1x1 convolutional layer to reduce the number of filters from 128 to 32 and added additional dropout layers to some fully connected layers. This was successful. The accuracy increased to over 0.93 at times.

At that moment, I undestood that the masive ammount of parameters was the problem, so I decided to decrease the inception blocks filters to 8 and the final 1x1 filter to 16. With this architecture I was able to get from 0.93-0.94 with the raw dataset. 

As a final step I went back to the data preprocessing and converted the images to greyscale and normalize it. This was the best model and gave me accuracy of 0.958 on the validation set. I tried to enhacend this dataset again with a subset of rotated images, but that actually harmed the model accuracy.
If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
I decided to try an inception module from the beginning because I wanted to experiment with its powerful ability to combine features. Also, I found interesting that a network can learn better just by parallelizing the network computational graph.
I believed that an inception module was going to be hlpful on the traffic sign classifier because of its previous success on similar classification problems.

* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
