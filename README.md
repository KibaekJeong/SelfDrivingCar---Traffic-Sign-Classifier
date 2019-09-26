# **Traffic Sign Recognition**

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

[image1]: ./output/explore_data.png "Visualization"
[image2]: ./output/graynormal.png "Grayscaling/normalizing"
[image3]: ./output/new_data.png "New Data"
[image4]: ./output/softmax0.png "Traffic Sign 1"
[image5]: ./output/softmax1.png "Traffic Sign 2"
[image6]: ./output/softmax2.png "Traffic Sign 3"
[image7]: ./output/softmax3.png "Traffic Sign 4"
[image8]: ./output/softmax4.png "Traffic Sign 5"
[image9]: ./output/featuremap.png "Feature Map"
[image10]: ./output/featuremap1.png "Feature Map1"

---
### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing the image data


As a first step, I decided to convert the images to grayscale. Although humans use color to recognize different traffic sign in real world, colors can vary depending on weather and condition of the traffic sign. Therefore, instead of using colors for verification, using shapes and contents of the traffic sign to identifying class would be more accurate.

In addition to gray scaling, normalization is applied to all the image data. Normalizing image changes range of pixel intensity values. It helps reducing poor contrast due to glare, noise in image, etc.

Here is an example of an original image and an augmented image:
![alt text][image2]


#### 2. Model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 GRAYSCALE image   							|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x32 				|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 12x12x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 6x6x64 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 4x4x128 	|
| RELU					|												|
| Flatten					|Input 4x4x128, Output 512												|
| Fully connected		| Input 512, Output 300        									|
| Dropout					|												|
| Fully connected		| Input 300, Output 150        									|
| Dropout					|												|
| Fully connected		| Input 150, Output 43        									|


#### 3. Type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

Optimizer : Adam optimizer
Batch Size: 128
Epochs : 50
Learning rate: 0.0006

#### 4. Discussion of the results on the training, validation and test sets.

First model architecture was adapted from LeNet architecture, which was one of the winnning architecture of image processing using Convolutional Neural Network. From there, I have tried adding another convolutional neural network layer and fully connected layer. By adding one more convolutional layer, I was able to achieve higher validation accuracy. However, adding more fully connected layer did not improve the model at all. So I decided to only add convolutional layer. Also, adding pooling layer and dropout layers improved model by preventing overfitting. When overfitted, model was only able to achieve accuracy of 7 percent. Also, when learning rate was higher, accuracy was fluctuating frequently for each epoch. So learning rate was reduced and number of epochs were increased. Lastly, I saved model only when validation accuracy is higher than 0.93 and highest among all epochs.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.975
* test set accuracy of 0.953


### Test a Model on New Images

#### 1. Obtain German traffic signs from the web and provide them in the report.

Here are German traffic signs that I found on the web:

![alt text][image3]

Among all of the sign photos, some of the signs are tilted in angle. Also, some of the traffic signs are covered with other objects or another traffic sign. These photos may be more difficult to classify as it is not in full shape or image of traffic sign is incomplete.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

Here are the results of the prediction:

| Prediction                    | Actual                        | Result    |

| ------------------------------------------------------------------------ |

| Dangerous curve to the right  | Speed limit (60km/h)          | Not Match |

| Traffic signals               | Traffic signals               | Match     |

| Go straight or left           | Go straight or left           | Match     |

| Speed limit (30km/h)          | Speed limit (30km/h)          | Match     |

| Road narrows on the right     | Road narrows on the right     | Match     |

| Road work                     | Road work                     | Match     |

| Turn right ahead              | Ahead only                    | Not Match |

| Speed limit (20km/h)          | Speed limit (20km/h)          | Match     |

| No entry                      | No entry                      | Match     |

| Pedestrians                   | Pedestrians                   | Match     |

| Children crossing             | Children crossing             | Match     |

| Priority road                 | Priority road                 | Match     |

| Speed limit (30km/h)          | Speed limit (30km/h)          | Match     |

| Go straight or left           | Go straight or left           | Match     |

| Bicycles crossing             | Road work                     | Not Match |

| Children crossing             | Children crossing             | Match     |

| Keep left                     | Keep left                     | Match     |

| Turn right ahead              | Turn right ahead              | Match     |

| No passing for vehicles over  | Speed limit (60km/h)          | Not Match |

| Road work                     | Road work                     | Match     |

|------------------------------------------------------------------------|

The model was able to correctly guess 17 of the 20 traffic signs, which gives an accuracy of 85%. This compares favorably to the accuracy on the test set of 95.3%.

#### 3. Softmax probabilities for each prediction.

Here are the softmax probabilities for each prediction.
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]

Example image with top five softmax probabilities of prediction is shown in the image.

### Visualize the Neural Network's State

 Following Figures demonstrate feature maps of first convolutional layer of our model on two of the new images found on the web.

![alt text][image9]
![alt text][image10]

As we can see from the above diagram, feature maps show general shape of the traffic signs. It contains circular or straight edges of traffic signs. Also, inner shape such as number 60 can be seen from first image feature map.
These feature maps shows how our model detects and classifies traffic signs.
