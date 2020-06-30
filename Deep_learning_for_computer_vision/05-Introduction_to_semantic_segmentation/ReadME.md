### **An overview of semantic image segmentation**
---
<br/>
<p align="center">
<img src="https://glassboxmedicine.files.wordpress.com/2020/01/coco-task-examples-1.png?w=616" width="60%"/>


<br/>

---
### **What can you do with images?**
---

* Classification (Image label)
* Semantic segmentation (pixel-wise label)
* Localization (bounding box)
* Object detection (multiple bounding boxes)
* Instance segmentation (multiple segments)
* Image captioning

<br/>

---
### **Introduction to image segmentation**
---
More specifically, the goal of semantic image segmentation is to label each pixel of an image with a corresponding class of what is being represented. Because we're predicting for every pixel in the image, this task is commonly referred to as dense prediction.

Here we are not assigning a label to each image that we did in classification, instead we labeling a each pixel in the input image with perticular category.

Pixel labels for training images must be known to train for semantic segmentation

if thier are 2 classes present in the image, then we have c+1 i.e 3 classes which include background.

<p align="center">
<img src="https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-17-at-7.42.16-PM.png" width="60%"/>


One important thing to note is that `we're not separating instances of the same class` (For example, it doesn't distinguish between persons in image they all belong to one class, person) ; we only care about the category of each pixel. In other words, if you have two objects of the same category in your input image, the segmentation map does not inherently distinguish these as separate objects. There exists a different class of models, known as instance segmentation models, which do distinguish between separate objects of the same class.

<p align="center">
<img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2019/03/Screenshot-from-2019-03-28-11-45-55.png" width="60%"/>

### **`How to create a training data ?`** <br/>

`Answer` : we have to label each pixel in the image manually with help of some tool.

Checkout some tool 
* [Manage and annotate training data by Brain Builder](https://www.youtube.com/watch?v=tYqnsp-OcLQ)

* [semantic-segmentation-using-deep-learning-tutorial-how-can-i-create-my-own-datasets](https://in.mathworks.com/matlabcentral/answers/380272-in-the-semantic-segmentation-using-deep-learning-tutorial-how-can-i-create-my-own-datasets)



**Segmentation models are useful for a variety of tasks, including:**

* Autonomous vehicles
    
    We need to equip cars with the necessary perception to understand their environment so that self-driving cars can safely integrate into our existing roads.

  <img src="https://www.jeremyjordan.me/content/images/2018/05/deeplabcityscape.gif" width="70%"/><p>A real-time segmented road scene for autonomous driving.</p>

* Medical image diagnostics
  
  Machines can augment analysis performed by radiologists, greatly reducing the time required to run diagnositic tests.

  <img src="https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-23-at-7.17.43-PM.png" width="70%"/><p>A chest x-ray with the heart (red), lungs (green), and clavicles (blue) are segmented.</p>

---
### **Representing the task**
---

  Simply, our goal is to take either a RGB color image (height × width × 3) or a grayscale image (height × width × 1) and output a segmentation map where each pixel contains a class label represented as an integer (height × width × 1). 

  <img src="https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-17-at-9.02.15-PM.png" width="80%"/>

  Similar to how we treat standard categorical values, we'll create our target by one-hot encoding the class labels - essentially creating an output channel for each of the possible classes.

  True label contain the one hot encoding vector 1 indicates perticular class is present and 0 indicate the perticular class is absent , and predicted label contain the probability of individual class present in the each channel.


  <img src="https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-16-at-9.36.00-PM.png" width="80%"/>

  A prediction can be collapsed into a segmentation map (as shown in the first image) by taking the argmax of each depth-wise pixel vector.

We can easily inspect a target by overlaying it onto the observation.

  <img src="https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-16-at-9.36.38-PM.png" width="80%"/>

When we overlay a single channel of our target (or prediction), we refer to this as a mask which illuminates the regions of an image where a specific class is present.

---
### **Receptive field**
---
The receptive field is perhaps one of the most important concepts in Convolutional Neural Networks (CNNs) that deserves more attention from the literature. All of the state-of-the-art object recognition methods design their model architectures around this idea.



<p align="center"><img src="https://www.researchgate.net/publication/316950618/figure/fig4/AS:495826810007552@1495225731123/The-receptive-field-of-each-convolution-layer-with-a-3-3-kernel-The-green-area-marks.png" width="40%"/>

**Receptive field** : each pixel in the feature map , how much are area in the input image it is affecting or how much area in input can affect , one pixel in output.

The green part (3x3) in layer 1 is a receptive field of green pixel(1x1) in layer 2.

A receptive field of a feature can be described by its center location and its size. (Edit later) However, not all pixels in a receptive field is equally important to its corresponding CNN’s feature. Within a receptive field, the closer a pixel to the center of the field, the more it contributes to the calculation of the output feature. Which means that a feature does not only look at a particular region (i.e. its receptive field) in the input image, but also focus exponentially more to the middle of that region.
why are using a conv layer instead of fully connected layer ?
so instead of classifying each pach in the input image , we are using big gient structure of conv layers which will give output in the form of w * h * c , where c is number of classes or number of category . 

Figure 1 shows some receptive field examples. By applying a convolution C with kernel size k = 3x3, padding size p = 1x1, stride s = 2x2 on an input map 5x5, we will get an output feature map 3x3 (green map). Applying the same convolution on top of the 3x3 feature map, we will get a 2x2 feature map (orange map). The number of output features in each dimension can be calculated using the following formula, which is explained in detail in [1].

<p align="center"><img src="https://miro.medium.com/max/660/1*D47ER7IArwPv69k3O_1nqQ.png"/>

to simplify things, we assume the CNN architecture to be symmetric, and the input image to be square. So both dimensions have the same values for all variables. If the CNN architecture or the input image is asymmetric, you can calculate the feature map attributes separately for each dimension.

<p align="center"><img src="https://miro.medium.com/max/2000/1*mModSYik9cD9XJNemdTraw.png" width="60%"/>

**Reference**:

* [https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807](https://medium.com/mlreview/a-guide-to-receptive-field-arithmetic-for-convolutional-neural-networks-e0f514068807)

---
### **Pixel and its spatial context**
---

<p align="center"><img src="https://i.ibb.co/xg3YFb7/Patch.jpg" alt="Patch" border="0" width="70%">

Larger the spatial context around the pixel , larger possibility of detecting a perticular class
We look at each patch of image and take the center pixel of this patches. considering the width and height of patches is odd.
Now i'll select a lot patches and assign a label based on the center pixel.
That's the approach of a semantic segmentation.

<p align="center"><img src="https://i.ibb.co/6w8Tvsz/Patch2.jpg" alt="Patch2" border="0" width="70%">

Naïve approach – classify patch based on the
label of the central pixel, then slide one pixel over.

**Disadvantage** : This is not very efficient!
Operations are being repeated without change in inputs or weights

---
### **Classification and Segmentation**
---

### **`Image Classification`**

  A CNN has
  * Convolutional layers
  * ReLU layers
  * Pooling layers
  * Fully connected layer

  **A classic CNN architecture would look something like this**:

  Input =>  Convolution => ReLU => Convolution => ReLU => Pooling => 
  ReLU => Convolution => ReLU => Pooling => Fully Connected <br/>

  A CNN convolves (not convolutes…) learned features with input data and uses 2D convolutional layers. This means that this type of network is ideal for processing 2D images. Compared to other image classification algorithms, CNNs actually use very little preprocessing. This means that they can learn the filters that have to be hand-made in other algorithms. CNNs can be used in tons of applications from image and video recognition, image classification, and recommender systems to natural language processing and medical image analysis.

  CNNs have an input layer, and output layer, and hidden layers. The hidden layers usually consist of convolutional layers, ReLU layers, pooling layers, and fully connected layers. 

  <p align="center"><img src="https://i.ibb.co/cvkXpWH/Screenshot-429.png" alt="Screenshot-429" border="0" width="60%" />

  A CNN works by extracting features from images. This eliminates the need for manual feature extraction. The features are not trained! They’re learned while the network trains on a set of images. This makes deep learning models extremely accurate for computer vision tasks. CNNs learn feature detection through tens or hundreds of hidden layers. Each layer increases the complexity of the learned features.

  <p align="center"><img src="https://www.learnopencv.com/wp-content/uploads/2017/11/cnn-schema1.jpg" width="60%"/>

  A CNN
  * Starts with an input image
  * Applies many different filters to it to create a feature map
  * Applies a ReLU function to increase non-linearity
  * Applies a pooling layer to each feature map
  * Flattens the pooled images into one long vector.
  * Inputs the vector into a fully connected artificial neural network.
  * Processes the features through the network. The final fully connected Layer provides the “voting” of the classes that we’re after.
  * trains through forward propagation and backpropagation for many, many epochs. This repeats until we have a well-defined neural network with trained weights and feature detectors.

---
### **`Image segmentation`**
---

A naive approach towards constructing a neural network architecture for this task is to simply stack a number of convolutional layers (with same padding to preserve dimensions) and output a final segmentation map. This directly learns a mapping from the input image to its corresponding segmentation through the successive transformation of feature mappings; however, it's quite computationally expensive to preserve the full resolution throughout the network.

Here we are not using fully connected layer at the end , instead we are only using convolutional layer .

<p align="center"><img src="https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-19-at-12.32.20-PM.png" width="50%"/>

Recall that for deep convolutional networks, earlier layers tend to learn low-level concepts while later layers develop more high-level (and specialized) feature mappings. In order to maintain expressiveness, we typically need to increase the number of feature maps (channels) as we get deeper in the network.

This didn't necessarily pose a problem for the task of image classification, because for that task we only care about what the image contains (and not where it is located). Thus, we could alleviate computational burden by periodically downsampling our feature maps through pooling or strided convolutions (ie. compressing the spatial resolution) without concern. However, for image segmentation, we would like our model to produce a full-resolution semantic prediction.

One popular approach for image segmentation models is to follow an encoder/decoder structure where we downsample the spatial resolution of the input, developing lower-resolution feature mappings which are learned to be highly efficient at discriminating between classes, and the upsample the feature representations into a full-resolution segmentation map.

<p align="center"/><img src="https://i.ibb.co/4pJ8d9T/Encoder-Decoder.jpg" alt="Encoder-Decoder" border="0" width="80%">

**`Encoder`** : Collects spatial context from a large receptive field <br/>
**`Decoder`** : Refines the features from coarse to fine scale

---
### **Methods for upsampling**
---

<p align="center"/><img src="https://miro.medium.com/max/1934/1*cMI2rBTp3VVRiyusAPv8OQ.png" alt="Encoder-Decoder" border="0" width="80%">

There are a few different approaches that we can use to upsample the resolution of a feature map. Whereas pooling operations downsample the resolution by summarizing a local area with a single value (ie. average or max pooling), "unpooling" operations upsample the resolution by distributing a single value into a higher resolution.

<p align="center"/><img src="https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-19-at-12.54.50-PM.png" width="60%"/>

Nearest neighbour and bit of nails , max unpooling they are not learnable upsampling. they are just function to compute .

if we think the strided convolutional is kind of learnable downsampling , it learn the way that the network wants to perform down sampling. 

Similar thing we have in case of upsampling : transpose convolutional aka (also know as) "Learnable upsampling" aka "upconvolution" aka "fractionally strided convolution" aka "deconvolution".

So we are doing both upsampling the feature map and learns some weights about how it work.

**Normal convolution** : Normal 3x3 convolution , with stride 1 and padding 1 <br/>
input size is equal to output size.

<p align="center"><img src="https://i.ibb.co/7SgVS44/Screenshot-425.png" alt="Screenshot-425" border="0" width="50%">

**Strided convolution** : Normal 3x3 convolution , with stride 2 and padding 1 <br/>
where size is reducing suppose input of size 4x4 and then output in the form of 2x2.

<p align="center"><img src="https://i.ibb.co/gz7H4J3/Screenshot-426.png" alt="Screenshot-426" border="0" width="50%">

**Transpose convolution** : 3x3 convolution , with stride 2 and padding 1 <br/>
input size is 2x2 and output size is 4x4.

<p align="center"><img src="https://i.ibb.co/vDc406t/Screenshot-427.png" alt="Screenshot-427" border="0" width="50%">


In input , suppose we are considering one upper left corner pixel which contain the scalar value , now we multiply that one pixel with filter of size 3x3 to get scalar value in the output of one pixel.

---
### **Fully convolutional networks**
---

The approach of using a "fully convolutional" network trained end-to-end, pixels-to-pixels for the task of image segmentation was introduced by Long et al. in late 2014. The paper's authors propose adapting existing, well-studied image classification networks (eg. AlexNet) to serve as the encoder module of the network, appending a decoder module with transpose convolutional layers to upsample the coarse feature maps into a full-resolution segmentation map.

<p align="center"><img src="https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-20-at-9.53.20-AM.png" width="60%"/>

The full network, as shown below, is trained according to a pixel-wise cross entropy loss.

<p align="center"><img src="https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-16-at-10.34.02-PM.png" width="60%"/>

However, because the encoder module reduces the resolution of the input by a factor of 32, the decoder module struggles to produce fine-grained segmentations (as shown below).

<p align="center"><img src="https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-20-at-10.15.09-AM.png" width="60%"/>

---
### **Adding skip connections**
---

The authors address this tension by slowly upsampling (in stages) the encoded representation, adding "skip connections" from earlier layers, and summing these two feature maps.
<br/>

<p align="center"><b>U-NET Architecture</b></p>
<p align="center"><img src="https://miro.medium.com/max/2824/1*f7YOaE4TWubwaFF7Z1fzNw.png" width="60%"/>

U-Net name because network is like "U" shape.

The line passes from encoder to decoder indicate "Concatenation of feature maps" which helps to give localization information.

These skip connections from earlier layers in the network (prior to a downsampling operation) should provide the necessary detail in order to reconstruct accurate shapes for segmentation boundaries. Indeed, we can recover more fine-grain detail with the addition of these skip connections.

<p align="center"><img src="https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-20-at-12.10.25-PM.png" width="60%"/>

#### **Concatenation**

suppose one input shape is (224,224,3) and another is (224,224,3) 
after concatenation , output shape which we get is (224,224,6)

<p align="center"><img src="https://i.ibb.co/cb5L76q/Screenshot-443.png" alt="Screenshot-443" border="0" width="70%">
 
---
### **Type of convolutional**
---

* 2D Convolutional
* Dilated / atrous convolutional
* transposed covolutional

---
### 2D convolutional
---
It is typical 2D convolutional , having kernel size = (3,3) with stride 1 and padding 1

<p align="center"><img src="https://miro.medium.com/max/790/1*1okwhewf5KCtIPaFib4XaA.gif" width="40%">

**Kernel size** : It is define the field of view of the convolutional

**Stride** : The stride define the step size of the kernel when traversing the image .
while its default is usally 1 , we can use stride of 2 for downsampling a image similar to maxpooling.  

**Padding** : The padding define how the border of a sample is handled.
A (half) padding convolutional will keep the spatial output dimension equal to the input , whereas unpadded convolutional will crop away 
some of the borders , if the kernel size is larger then 1.

**Input and output channel**: A convolutional layer takes a certain number of input channels (I) and calculates a specific number of output channels (O). The needed parameters for such a layer can be calculated by I*O*K, where K equals the number of values in the kernel.

---
### **Dilated/atrous convolutions**
---
2D Convolutional using a 3x3 kernel size with dilated rate 2 and no padding.

<p align="center"><img src="https://miro.medium.com/max/790/1*SVkgHoFoiMZkjy54zM_SUw.gif" border="0" width="40%">


Dilated convolutions introduce another parameter to convolutional layers called the **dilation rate**. This defines a spacing between the values in a kernel. A 3x3 kernel with a dilation rate of 2 will have the same field of view as a 5x5 kernel, while only using 9 parameters. Imagine taking a 5x5 kernel and deleting every second column and row.

One benefit of downsampling a feature map is that it broadens the receptive field (with respect to the input) for the following filter, given a constant filter size. Recall that this approach is more desirable than increasing the filter size due to the parameter inefficiency of large filters (discussed here in Section 3.1). However, this broader context comes at the cost of reduced spatial resolution.

Dilated convolutions provide alternative approach towards gaining a wide field of view while preserving the full spatial dimension. As shown in the figure below, the values used for a dilated convolution are spaced apart according to some specified dilation rate.

<p align="center"><img src="https://i.ibb.co/vw1wLRm/Dilated-Convolution.jpg" alt="Dilated-Convolution" border="0" width="60%">

Atrous (dilated) convolutions can increase the receptive
field without increasing the number of weights. 

<p align="center"><img src="https://www.researchgate.net/publication/336002670/figure/fig1/AS:806667134455815@1569335840531/An-illustration-of-the-receptive-field-for-one-dilated-convolution-with-different.png" width="60%"/>

#### **`why we increase filter size from 3x3 to 5x5 ?`** 
**Answer** : It increases the receptive field 
The assumption that if we consider the one pixel and its neighbourhood pixel they are most commonly
same , so we ignore that neighbourhood by skipping one pixel.


**Attention to Refine through Multi-Scales for Semantic Segmentation**

<p align="center"><img src="https://storage.googleapis.com/groundai-web-prod/media/users/user_49766/project_215011/images/2.png" width="60%">

---
### **Transposed Convolutions (a.k.a. deconvolutions or fractionally strided convolutions)**
---

Some sources use the name deconvolution, which is inappropriate because it’s not a deconvolution. To make things worse deconvolutions do exists, but they’re not common in the field of deep learning. An actual deconvolution reverts the process of a convolution. Imagine inputting an image into a single convolutional layer. Now take the output, throw it into a black box and out comes your original image again. This black box does a deconvolution. It is the mathematical inverse of what a convolutional layer does.

A transposed convolution is somewhat similar because it produces the same spatial resolution a hypothetical deconvolutional layer would. However, the actual mathematical operation that’s being performed on the values is different. A transposed convolutional layer carries out a regular convolution but reverts its spatial transformation.

At this point you should be pretty confused, so let’s look at a concrete example. An image of 5x5 is fed into a convolutional layer. The stride is set to 2, the padding is deactivated and the kernel is 3x3. This results in a 2x2 image.

<p align="center"><img src="https://miro.medium.com/max/588/1*BMngs93_rm2_BpJFH2mS0Q.gif" width="40%">

If we wanted to reverse this process, we’d need the inverse mathematical operation so that 9 values are generated from each pixel we input. Afterward, we traverse the output image with a stride of 2. This would be a deconvolution.

<p align="center"><img src="https://miro.medium.com/max/790/1*Lpn4nag_KRMfGkx1k6bV-g.gif" width="40%">

A transposed convolution does not do that. The only thing in common is it guarantees that the output will be a 5x5 image as well, while still performing a normal convolution operation. To achieve this, we need to perform some fancy padding on the input.
As you can imagine now, this step will not reverse the process from above. At least not concerning the numeric values.
It merely reconstructs the spatial resolution from before and performs a convolution. This may not be the mathematical inverse, but for Encoder-Decoder architectures, it’s still very helpful. This way we can combine the upscaling of an image with a convolution, instead of doing two separate processes.

 
**why we should not call transposed convolutional as deconvolution** 

if we pass input image of size 4x4 and apply 2D conv then we get 2x2 image
and using that 2x2 we are producing 4x4 input size.
then it deconv but here we are modifiying by adding padding over 2x2 .

---
### **Separable Convolutions**
---

In a separable convolution, we can split the kernel operation into multiple steps. Let’s express a convolution as y = conv(x, k) where y is the output image, x is the input image, and k is the kernel. Easy. Next, let’s assume k can be calculated by: k = k1.dot(k2). This would make it a separable convolution because instead of doing a 2D convolution with k, we could get to the same result by doing 2 1D convolutions with k1 and k2.
<p align="center"><img src="https://miro.medium.com/max/860/1*owXMr9DonUUWP1c2Thg_Dw.png" width="70%">

Take the Sobel kernel for example, which is often used in image processing. You could get the same kernel by multiplying the vector [1, 0, -1] and [1,2,1].T. This would require 6 instead of 9 parameters while doing the same operation. The example above shows what’s called a spatial separable convolution, which to my knowledge isn’t used in deep learning.

<p align="center"><img src="https://miro.medium.com/max/2476/1*o3mKhG3nHS-1dWa_plCeFw.png" width="60%">

**Reference**

* [types of convolutions in deep learning](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)

---
### **Data augmentation**
---

If you have less number of training pairs, the results might not be good be because the model might overfit. We can increase the size of the dataset by applying random transformations on the images. We can change the color properties like hue, saturation, brightness, etc of the input images. We can also apply transformations such as rotation, scale, and flipping. For the transformations which change the location of the pixels, the segmentation image should also be transformed the same way.

<p align="center"><img src="https://divamgupta.com/assets/images/posts/imgseg/image13.png?style=centerme" width="60%"/>

---
#### **Defining a loss function**
---

The most commonly used loss function for the task of image segmentation is a pixel-wise cross entropy loss. This loss examines each pixel individually, comparing the class predictions (depth-wise pixel vector) to our one-hot encoded target vector.

<p align="center"><img src="https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-24-at-10.46.16-PM.png" width="70%"/>

Because the cross entropy loss evaluates the class predictions for each pixel vector individually and then averages over all pixels, we're essentially asserting equal learning to each pixel in the image. This can be a problem if your various classes have unbalanced representation in the image, as training can be dominated by the most prevalent class. Long et al. (FCN paper) discuss weighting this loss for each output channel in order to counteract a class imbalance present in the dataset.

Meanwhile, Ronneberger et al. (U-Net paper) discuss a loss weighting scheme for each pixel such that there is a higher weight at the border of segmented objects. This loss weighting scheme helped their U-Net model segment cells in biomedical images in a discontinuous fashion such that individual cells may be easily identified within the binary segmentation map.


<p align="center"><img src="https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-21-at-10.53.04-PM.png" width="70%"/>

---
### **Dice Coefficient**
---

Another popular loss function for image segmentation tasks is based on the Dice coefficient, which is essentially a measure of overlap between two samples. This measure ranges from 0 to 1 where a Dice coefficient of 1 denotes perfect and complete overlap. The Dice coefficient was originally developed for binary data, and can be calculated as:

Dice =  2 * |A ∩ B| / |A|+|B| 

<br/>
<p align="center"><img src="https://jinglescode.github.io/assets/img/posts/unet-03.webp" width="50%"/>

where |A ∩ B| represents the common elements between sets A and B, and 
|A|

represents the number of elements in set A (and likewise for set B).

For the case of evaluating a Dice coefficient on predicted segmentation masks, we can approximate |A∩B|
as the element-wise multiplication between the prediction and target mask, and then sum the resulting matrix.

<p align="center"><img src="https://www.jeremyjordan.me/content/images/2018/05/intersection-1.png" width="70%"/> 

Because our target mask is binary, we effectively zero-out any pixels from our prediction which are not "activated" in the target mask. For the remaining pixels, we are essentially penalizing low-confidence predictions; a higher value for this expression, which is in the numerator, leads to a better Dice coefficient.

In order to quantify |A| and |B|, some researchers use the simple sum whereas other researchers prefer to use the squared sum for this calculation. 

<p align="center"><img src="https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-25-at-5.53.46-PM.png" width="60%"/>

Dice coefficient is equal to F1 Score 

<p align="center"><img src="https://i.ytimg.com/vi/fcO9820wCXE/hqdefault.jpg" width="40%"/> 

Dice coefficient in terms of True positive , False positive and False negative.

<p align="center"><img src="https://miro.medium.com/max/1596/1*Z1hkDvyhFBogT9EkzVkX2A.png" width="50%"/>

---
### **Dice Loss**
---

In order to formulate a loss function which can be minimized, we'll simply use 1−Dice. This loss function is known as the **soft Dice loss** because we directly use the predicted probabilities instead of thresholding and converting them into a binary mask.

Range of Dice loss is between 0 to 1

With respect to the neural network output, the numerator is concerned with the common activations between our prediction and target mask, where as the denominator is concerned with the quantity of activations in each mask separately. This has the effect of normalizing our loss according to the size of the target mask such that the soft Dice loss does not struggle learning from classes with lesser spatial representation in an image.

<p align="center"><img src="https://www.jeremyjordan.me/content/images/2018/05/Screen-Shot-2018-05-24-at-10.50.59-PM.png" width="60%"/>

---
### **Transfer learning**
---

The CNN models trained for image classification contain meaningful information which can be used for segmentation as well. We can re-use the convolution layers of the pre-trained models in the encoder layers of the segmentation model. Using Resnet or VGG pre-trained on ImageNet dataset is a popular choice. You can read more about transfer learning [here](http://cs231n.github.io/transfer-learning/).

<p align="center"><img src="https://divamgupta.com/assets/images/posts/imgseg/image4.png?style=centerme" width="60%"/>

**Reference** :

* [A Beginner's guide to Deep Learning based Semantic Segmentation using Keras](https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html)

---
### **SegNet vs U-net**
---

**SegNet (Semantic Segmentation)** : The SegNet architecture adopts an encoder-decoder framework. The encoder and decoder layers are symmetrical to each other. The upsampling operation of the decoder layers use the max-pooling indices of the corresponding encoder layers. SegNet does not have any skip connections. Unlike FCN, no learnable parameters are used for upsampling.

SegNet is slower than FCN and DeepLabv1 because SegNet contains the decoder architecture. And it is faster than DeconvNet because it does not have fully connected layers.

And SegNet has low memory requirement during both training and testing. And the model size is much smaller than FCN and DeconvNet.


**U-Net** : The UNet architecture adopts an encoder-decoder framework with skip connections. Like SegNet, the encoder and decoder layers are symmetrical to each other.

### **References** :

* [DLV-2 Semantic Segmentation (Lecture) - SHALA 2020](https://www.youtube.com/watch?v=RVJJZtUS2ho)

* [An overview of semantic image segmentation](https://www.jeremyjordan.me/semantic-segmentation/)

* [Lecture 11 | Detection and Segmentation by standford university](https://www.youtube.com/watch?v=nDPWywWRIRo)

* [U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation](https://arxiv.org/abs/1801.05746)

* [Semantic Image Segmentation with DeepLab in TensorFlow ](https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html)

* [Semantic Segmentation — U-Net](https://medium.com/@keremturgutlu/semantic-segmentation-u-net-part-1-d8d6f6005066)

* [The Complete Beginner’s Guide to Deep Learning: Convolutional Neural Networks and Image Classification](https://towardsdatascience.com/wtf-is-image-classification-8e78a8235acb)

* [Review: SegNet (Semantic Segmentation)](https://towardsdatascience.com/review-segnet-semantic-segmentation-e66f2e30fb96)

### **Reference material** 

* [Different Popular Dataset for Various Task](https://www.robots.ox.ac.uk/~vgg/data/) 

* [Tensorflow tutorial on image segmentation](https://www.tensorflow.org/tutorials/images/segmentation)

* [Segnet Tutorial](https://github.com/alexgkendall/SegNet-Tutorial)

* [Semantic segmentation](https://colab.research.google.com/github/dhassault/tf-semantic-example/blob/master/01b_semantic_segmentation_basic_colab.ipynb#scrollTo=It1Ul9OhTUW0)

* [Scene Parsing through ADE20K Datase](http://people.csail.mit.edu/bzhou/publication/scene-parse-camera-ready.pdf)

* [Semantic Segmentation with tf.data in TensorFlow 2 and ADE20K dataset](https://yann-leguilly.gitlab.io/post/2019-12-14-tensorflow-tfdata-segmentation/)

* [Imagenet vggnet resnet inception xception keras](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/)

* [Automating cell nuclei detection with neural network](http://cs230.stanford.edu/files_winter_2018/projects/6937642.pdf)
