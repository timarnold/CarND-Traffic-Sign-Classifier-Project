# Traffic Sign Recognition

## Writeup Template

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[preprocessing]: ./examples/preprocessing.png "Preprocessing"
[exploratory]: ./examples/exploratory_visualization.png "Exploratory Visualization"
[lenet]: ./examples/lenet.png "LeNet-5 Architecture"
[results]: ./examples/results.png "Training Results"
[web_images]: ./examples/web_images.png "Traffic Sign Images from Web"
[web-1]: ./examples/web-1.png "Web Image #1: Traffic Sign"
[web-2]: ./examples/web-2.png "Web Image #2: Duck Crossing"
[web-3]: ./examples/web-3.png "Web Image #3: Caution"
[web-4]: ./examples/web-4.png "Web Image #4: Double Curve"
[web-5]: ./examples/web-5.png "Web Image #5: Traffic Sign"

## Rubric Points

### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/timcamber/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration, Preprocessing

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the the IPython notebook.

I used basic NumPy methods to calculate summary statistics of the traffic signs data set:

* The size of the training set is 34,799 images
* The size of the validation set is 4,410 images
* The size of the test set is 12,630 images
* The shape of a traffic sign image is (32, 32, 3), i.e., 32x32 with 3 color channels
* The number of unique classes/labels in the data set is 43.

#### 2. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

We performed preprocessing steps before the exploratory visualization of the dataset, because I thought it would be valuable to include data before and after preprocessing in my exploratory visualization.

Below the section titled "Preprocessing Dataset" in my IPython notebook is the method I use to preprocess my dataset (`preprocess_image`). This preprocessing adjusts the brightness of the pixels to normalize the histogram of pixel brightness values using OpenCV's `equalizeHist` function. This helps normalize the images' brightness throughout the dataset, and improve image contrast. I also converted all images to grayscale. Using grayscale images or three-channel color images didn't seem to have a huge effect on my results, but grayscale images provided marginally better accuracy on the validation set. It also reduces the input size and thus marginally improves the training speed.

I added a code cell (under "Visualize Preprocessing" in the IPython notebook) to visualize the effects that my preprocessing steps have on the dataset. The code block can select a random image (which I used during development of my notebook) or a specific image.

There are three histogram plots that show a histogram of the pixel brightness values in the image. The first row is an unmodified image. The second row is the same image after applying OpenCV's `equalizeHist` function. The third row is the result of scaling the intermediate value so that the pixel values all fall between -0.5 and 0.5 instead of 0 to 255. Shifting input values to this range seemed to be a widely suggested tip for improving neural net performance, and improved my validation accuracy.

Finally, a before and after image is shown of the subject image.

![Figure showing preprocessing steps][preprocessing]

#### 3. Include an exploratory visualization of the dataset and identify where the code is in your code file.

For my exploratory visualization, I plotted a histogram of the sign classes and showed samples from each class, along with the human-readable name of the class. These results can be seen in the IPython notebook under the section title "Include an exploratory visualization of the dataset". The histogram and samples from two classes can be seen in the figure below.

![Figure showing general exploration of the dataset][exploratory]

This visualization shows a few things. First, that our dataset is highly unbalanced. That is, the number of signs of each sign class type can differ dramatically between classes. This might affect how well our classifier works for different sign types. Ideally, we would add some preprocessing to balance our dataset, perhaps in an input data augmentation step.

We additionally see some examples of the data we'll be working with. We note the sometimes dramatic improvements our preprocessing steps have on the data. Also, we can see some of the artifacts of the more problematic input data (e.g. blurred). This information could be used if we decided to augment our dataset. E.g., if many blurred images appear, we could decide to augment our dataset to include more blurred images.

In this experiment, I ran out of time to include image augmentation. Since I was already getting pretty good results without augmentation, I decided not to devote extra time to augmentation. I did explore options for augmentation, which could include rotation, scaling, skewing, brightness variations, simulated motion blur, and, for some sign types, reflection.

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets.

Training, validation, and testing data was stored in variables and preprocessed in the IPython notebook under the heading "Preprocess Dataset". Label values (`y_train` etc.) were changed to [one-hot encoding](https://en.wikipedia.org/wiki/One-hot) for easier input into TensorFlow. Training, validation, and testing input data were preprocessed in the same fashion. Since Udacity was kind enough to pre-split training and validation data, we did not have to select a subset of our training dataset to use for validation. As described above, the training dataset contained 34,799 images and the validation dataset contained 4,410 images.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

After experimenting with different model architectures, I eventually created some basic helper methods for common actions like generating convolutional layers, max pooling layers, and fully connected layers.

My main inspiration and references for different models were the Udacity lectures that described the LeNet-5 model and its associated presentation in the literature ([LeCun 1998](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)), another paper describing a multi-scale convolutional network used for the specific traffic sign classification problem [Sermanet 2011](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), and the various architectures described in [Stanford CS231n lectures](https://cs231n.github.io) [here](https://cs231n.github.io/convolutional-networks/#case), which included LeNet, AlexNet, VGGNet, and others. I actually initially tried a larger variety of architectures, but when I ran into issues I reverted to the basic LeNet example and focused on making improvements outside the net architecture (preprocessing, regularization, etc.). After mamking preprocessing and regularization improvements, I was getting sufficiently high validation accuracies that I didn't find a great need to explore more complex architectures.

A visualization of the LeNet-5 architecture can be seen below.

![Figure showing LeNet-5 architecture][lenet]

I used the same basic structure as LeNet-5 (two convolutional layers, two max pooling layers, two max pooling layers, and two fully connected layers), with the change of increasing the number of convolutional feature maps, adding Dropout [Srivastava 2014](http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf) and weight regularization.

In addition to increasing the number of feature maps and adding Dropout and weight regularization, I also used a `tanh` activation instead of `relu`. I did this to attempt to address issues of my optimizer getting stuck in its initial state due, I believe, to issues with near-zero initial conditions. I addressed these issues by using an [Xavier initialization](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf), which has built-in support for `tanh` activations in TensorFlow.

The code for my final model is located a few code cells below the header "Model Architecture" in my IPython notebook, defined in the function `model_pass`.

My final model consisted of the following layers:

| Layer                 |     Description                                |
|-----------------------|------------------------------------------------|
| Input                 | 32x32x3 RGB image                              |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x128   |
| TANH                  |                                                |
| Max pooling           | 2x2 stride,  outputs 14x14x128                 |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x256   |
| TANH                  |                                                |
| Max pooling           | 2x2 stride,  outputs 5x5x256                   |
| Fully connected       | 6400 flattened to 120                          |
| TANH                  |                                                |
| Dropout               | 75% keep probability                           |
| Fully connected       | 120 to 84                                      |
| TANH                  |                                                |
| Dropout               | 75% keep probability                           |
| Fully connected       | 84 to 43 (number of classes)                   |
| Softmax               |                                                |

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located under the heading "Train, Validate and Test the Model" in the IPython notebook. I trained for 100 epochs, with a batch size of 1024, learning rate of 0.001, Dropout keep probability of 75%, and weight regularization parameter (beta) of 0.001. These hyperparameters were tweaked ad hoc by small amounts to try to improve validation accuracy.

To train, the training dataset was shuffled and separated into batches. The batches cross entropy was calculated and the loss was calculated as the mean of the softmax cross entropy plus the L2 loss of the weights of the fully connected layers. When evaluating the accuracy and loss of the model, keep probability was set to 100% so neurons were not dropped from the model during accuracy checking. The training and validation accuracies and losses were retained during training for later visualiation, and the model parameters were saved.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Results from training can be seen below. The plot below shows training and validation accuracy and loss during training (for 100 epochs), as well as a text output of the testing accuracy and loss after training, which was 95.5%. Final validation accuracy was 97.3%.

![alt text][results]

My final model results were:
* training set accuracy of 99.4%
* validation set accuracy of 97.3%
* test set accuracy of 95.5%

As described above, I started out dramatically modifying net architectures, as that seemed to be the most interesting problem when I started. I ran into issues because of bugs, and, I suspect, because I didn't do any preprocessing or regularization at first. After some failure trying different architectures, I added preprocessing and regularization, which dramatically improved my result with the initial LeNet-5 architecture.

I then tweaked the LeNet-5 architecture, primarily by adding feature maps and regularization strategies (dropout and weight regularization). Regularization strategies were especially useful when I was seeing a big difference between training and validation accuracies, indicating potential overfitting to the training set. These improvements were substantial enough that I was satisfied with the model's performance, and I focused on other aspects of the project. I did a little bit of tweaking of hyperparameters, but didn't find that they substantially altered my validation accuracy.

I tried Dropout keep probabilities equal to and lower than 50% as well, but found that these values dramatically reduced the overall accuracy of my validation set.

The LeNet-5 architecture, and similar architectures I tried (e.g. [Sermanet 2011](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), which was specifically designed for this data set), all seemed to be potentially useful because they processed similar or identical data to our dataset. Convolutional layers work well on image-based data like our dataset.

I found that my model worked well, but was not "state-of-the-art". Adding data augmentation, and perhaps multi-scale strategies (as in [Sermanet 2011](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf); this is where output of convolutional layers are directly added to the first fully connected layer to incorporate those weights in the final classifier, letting the classifier use different scale features before they are further processed by later convolutional and subsampling layers) would likely improve my accuracy.
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Below are five German traffic signs that I found [on the web](http://finde-das-bild.de/bildersuche?keys=verkehrsschild), before and after being processed by my preprocessing pipeline. Images were downloaded from the web and resized to 32x32x3.

![Traffic sign images from the Internet][web_images]

Four of the five images were signs that appeared in my training set (traffic signs, caution, double curve, and 20 km/h speed limit). One image (duck crossing) did not appear in my training set. I would expect my classifier to get good accuracy on the objects that were in the training set, and poor accuracy for signs not in the training set.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Results from my classifier acting on the images can be seen in the Jupyter notebook under the heading "Predict the Sign Type for Each Image". Additionally, the top 8 classifications for each image can be seen in the figure below.

![Internet image classification][web-1]
![Internet image classification][web-2]
![Internet image classification][web-3]
![Internet image classification][web-4]
![Internet image classification][web-5]

Here are the results of the prediction:

| Image                 |     Prediction               |
|-----------------------|------------------------------|
| Traffic signals       | Traffic signals (14.9%)      |
| Duck crossing         | Beware of ice/snow (11.5%)   |
| General caution       | General caution (14.9%)      |
| Double curve          | Double curve (3.2%)          |
| Speed limit (20 km/h) | Speed limit (60 km/h) (2.3%) |

Since I only have 5 Internet images, I am simply doing the math by hand. My classifier got 3/5 images correct. One image (duck crossing, I believe) doesn't actually correspond to a sign in my dataset, and so my model got 3/4 images correct for the images that existed in my dataset. This means the accuracy is either 60% or 75%, depending on how we decide to handle sign types that don't exist in one of the 43 classes in the training dataset.

Surprisingly (because my model overall has a high accuracy and this particular image is a very clear one), my model got the 20 km/h sign wrong. All its predictions were tied for a very low probability, but not even the top 8 included the 20 km/h sign class. One possible explanation is that the 20 km/h sign class is very under-represented in our training set (there are 7x as many 120 km/h signs, for example).