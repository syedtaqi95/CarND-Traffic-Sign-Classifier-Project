# Traffic Sign Recognition Project

The goal of this project is to build a traffic sign recognition classifier using deep learning techniques.

Find my project on [Github](https://github.com/syedtaqi95/CarND-Traffic-Sign-Classifier-Project).

 The steps of this project are the following:

- Load the data set (see below for links to the project data set)
- Explore, summarize and visualize the data set
- Design, train and test a model architecture
- Use the model to make predictions on new images
- Analyze the softmax probabilities of the new images
- Summarize the results with a written report

[//]: # (Image References)
[dataset_histogram]: ./writeup_images/dataset_histogram.png "dataset_histogram"
[example_images]: ./writeup_images/example_images.png "example_images"
[grayscale]: ./writeup_images/grayscale.png "grayscale"
[local_hist_eq]: ./writeup_images/local_hist_eq.png "local_hist_eq"
[normalised]: ./writeup_images/normalised.png "normalised"
[newtest_images]: ./writeup_images/newtest_images.png "newtest_images"
[road_narrows]: ./writeup_images/road_narrows.png "road_narrows"

---

## Rubric Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

### **Files Submitted**

#### 1. The project submission includes all required files.

- Ipython notebook with code: see *Traffic_Sign_Classifier.ipynb*.
- HTML output of the code: see *Traffic_Sign_Classifier.html*
- A writeup report (either pdf or markdown): you're reading it!

### **Dataset Exploration**

#### 1. The submission includes a basic summary of the data set.

I used the *numpy* library to calculate the statistics of the datasets:

- Number of training examples = 34799
- Number of validation examples = 4410
- Number of testing examples = 12630
- Image data shape = (32, 32, 3)
- Number of classes = 43

#### 2. The submission includes an exploratory visualization on the dataset.

I visualised the training, validation and test datasets using a probability distribution histogram, as below:

![dataset_histogram]

As the histogram shows, most classes have roughly equal probability distributions in each of the test sets. Some classes (such as n=20,21,22) have slightly higher probability densities in the validation test set compared to the training and test sets.

I also plotted a random set of images from the images, example below:

![example_images]

### **Design and Test a Model Architecture**

#### 1. The submission describes the preprocessing techniques used and why these techniques were chosen.

First, I shuffled the training dataset using the *sklearn* library, because it improves the detection for all classes equally. Since I implemented batching in my training pipeline, I would like to have all classes represented roughly equally in each training batch.

Second, I converted the images to grayscale as they improved detection accuracy as per the research paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks" by Pierre Sermanet and Yann LeCun. Some example images are below:

![grayscale]

Third, I applied local histogram equalisation using the *skimage* library. Since the training images are real-world photos taken in different lighting conditions, this technique improves the contrast of the images and hence improves the visibility of the traffic signs. Some example images are below:

![local_hist_eq]

Fourth and last, I applied normalisation to the training images. Normalisation is used so the image pixels have zero mean and equal variance. I approximated this step by using the following formula:

```python
pixel_value = pixel_value / 255
```

Some example images are below:

![normalised]

#### 2. The submission provides details of the characteristics and qualities of the architecture, including the type of model used, the number of layers, and the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged.

I implemented two popularly used Convolutional Neural Networks (CNNs) in this project:
- LeNet-5
- VGGNet-16

**LeNet-5 Model:**

I implemented the same model as the one used in the LeNet-5 lab.

| Layer  | Description |
| :-----:|:------------|
| Input | The LeNet architecture accepts a 32x32xC image as input, where C is the number of colour channels (1 in this case) |
| Convolutional | 1x1 stride, valid padding, output 28x28x6 |
| Activation | ReLU |
| Max Pooling | 2x2 stride, valid padding, output 14x14x6 |
| Convolutional | 1x1 stride, valid padding, output 10x10x16 |
| Activation | ReLU |
| Max Pooling | 2x2 stride, valid padding, output 5x5x16 |
| Flatten | Output 400 |
| Fully connected | Output 120 |
| Activation | ReLU |
| Fully connected | Output 84 |
| Activation | ReLU |
| Fully connected | Output 43 |

**VGGNet-16 Model:**

I implemented a modified version of the VGGNet-16 model as I reduced the number of layers to 12 to aid computation speed. 

| Layer  | Description |
| :-----:|:------------|
| Input |  VGGNet also accepts a 32x32xC input, where C is the number of colour channels (1 in this case)|
| Convolutional | 1x1 stride, same padding, output 32x32x32 |
| Activation | ReLU |
| Convolutional | 1x1 stride, same padding, output 32x32x32 |
| Activation | ReLU |
| Max Pooling | 2x2 stride, valid padding, output 16x16x32, dropout |
| Convolutional | 1x1 stride, same padding, output 16x16x64 |
| Activation | ReLU |
| Convolutional | 1x1 stride, same padding, output 16x16x64 |
| Activation | ReLU |
| Max Pooling | 2x2 stride, valid padding, output 8x8x64, dropout |
| Convolutional | 1x1 stride, same padding, output 8x8x128 |
| Activation | ReLU |
| Convolutional | 1x1 stride, same padding, output 8x8x128 |
| Activation | ReLU |
| Max Pooling | 2x2 stride, valid padding, output 4x4x128, dropout |
| Flatten | Output 2048 |
| Fully connected | Output 128 |
| Activation | ReLU, dropout |
| Fully connected | Output 128 |
| Activation | ReLU, dropout |
| Fully connected | Output 43 |

#### 3. The submission describes how the model was trained by discussing what optimizer was used, batch size, number of epochs and values for hyperparameters.

I used the following parameters to train both models:

```python
EPOCHS = 30
BATCH_SIZE = 64
rate = 0.001
```
I used the Adam optimiser for this project as it is more efficient compared to other stochastic optimisation methods. Research has shown that it minimises the loss function more quickly compared to other methods such as stochastic gradient descent.

#### 4. The submission describes the approach to finding a solution. Accuracy on the validation set is 0.93 or greater.

I first used LeNet-5 and achieved the following results:
- Training accuracy = 99.718%
- Validation accuracy = 93.787%
- Test accuracy = 91.346%

The discrepancy between the training and validation accuracy can be attributed to overfitting. While the results were not bad (and good enough for the project criteria), I was keen to improve my results! After a bit of research, I found the VGGNet-16 model which performed better than LeNet-5 for image recognition applications. 

I modified the VGGNet model to reduce the number of layers to 12 to improve computation speed. I also implemented dropout in this model, which helped to create redundancy and reduce reliance on one set of neurons. I played around with the batch size, epochs and learning rate, and found the values mentioned above as a good balance between training time and accuracy. One last change I made was to save the model only if it had a better validation accuracy than that in previous epochs.

Once I implemented these changes, I achieved the following results:
- Training accuracy = 99.181%
- Validation accuracy = 98.435%
- Test accuracy = 96.318%

Much better!

### **Test a Model on New Images**

#### 1. The submission includes five new German Traffic signs found on the web, and the images are visualized. Discussion is made as to particular qualities of the images or traffic signs in the images that are of interest, such as whether they would be difficult for the model to classify.

Here are 5 German traffic sign images I found online:

![newtest_images]

The *Road work* image might be difficult to classify due to the low contrast of the image (hence why I implemented local histogram equalisation!). The *Speed limit (50km/h)* sign is slightly blurred so presents a challenge as well.

#### 2. The submission documents the performance of the model when tested on the captured images. The performance on the new images is compared to the accuracy results of the test set.

Here are the results of the prediction:

| Image | Prediction |
|:-----:|:----------:|
| Road work | Road work |
| Priority road | Priority road |
| Speed limit (50km/h) | Speed limit (50km/h) |
| Road narrows on the right | Road narrows on the right |
| Stop | Stop |

Which gives an accuracy of 100%! This is better than the previously achieved test accuracy of 96.318%.

#### 3. The top five softmax probabilities of the predictions on the captured images are outputted. The submission discusses how certain or uncertain the model is of its predictions.

I visualised the top 5 softmax probabilities in the Ipython notebook in the relevant section.

The prediction softmax probabilities of 4 out of the 5 images were almost 1, showing that the network was very confident of its predictions. The only image which had a slightly lower prediction softmax probability (around 0.9) was the *Road narrows on the right* image which had a non-trivial probability for *Pedestrians*, as below:

![road_narrows]

---

## Conclusion

While I initially struggled to get up and running on this project, since I had no previous experience in machine learning or Tensorflow, it was a great learning experience to see the techniques improve my model (or sometimes not!). I tried to implement the techniques taught in the lectures and others I found online. I also felt that this project provided a good context to the kinds of challenges self-driving car engineers face in their day-to-day roles.

