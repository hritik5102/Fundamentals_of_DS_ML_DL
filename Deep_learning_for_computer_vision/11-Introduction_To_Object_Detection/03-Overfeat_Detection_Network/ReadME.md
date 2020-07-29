## Content

* [Overfeat intuition](#overfeat-intuition)

  * [Overfeat classification network](#overfeat-classification-network)

  * [Overfeat detection network](#overfeat-detection-network)

* [1x1 convolution model size effective stride](#1x1-convolution-model-size-effective-stride)

  * [1x1 convolution and model size of convnet](#1x1-convolution-and-model-size-of-convnet)

  * [Effective stride of a network](#effective-stride-of-a-network)

* [Pre and post processing](#pre-and-post-processing)
  * [Confidence score thresholding](#confidence-score-thresholding)
  * [Non max suppression](#non-max-suppression)

* [Summary](#summary)

## Overfeat intuition

By putting all the concepts we have learnt till now, we will be able to understand Overfeat. By converting the FC layers into convolution operation, we have removed the fixed input size constraint. There are 3 advantages to this:

I can use the same localization network, without using the Sliding Window crops at different locations.
Since there are no input size constraint, I will be able to use the Image pyramids.
Since, I am using Image Pyramids, I will get the Spatial Output, which will give me detections at different locations of the image.
Since the entire network is using Convolution operations, it is way more efficient than taking crops.
This the intuition behind OverFeat Network.

This Network won the ImageNet 2013 localization task (ILSVRC2013) and obtained very competitive results for the detection and classifications tasks.

[OverFeat: Integrated Recognition, Localization and Detection using Convolutional Networks – Sermanet et al](https://www.youtube.com/watch?v=t5PHp8uSMKo&list=PL1GQaVhO4f_jLxOokW7CS5kY_J1t1T17S&index=53)

See detailed discussion [here](https://www.youtube.com/watch?v=t5PHp8uSMKo&list=PL1GQaVhO4f_jLxOokW7CS5kY_J1t1T17S&index=53)

## Overfeat Classification Network

This is the Overfeat Classification Network. It uses a modified AlexNet architecture for Convolution operations.

The first FC layer has 5x5 filter and 2nd and 3rd have 1x1 filters. The depth at both layers are 4096 each.

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/detection_overfeata_classify.jpg" width="70%">
</p>

Let’s look at the FC layers in detail. Here, I have shown only the filter sizes without the depth

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/detection_overefeat_1x1_convolution.jpg" width="70%"></p>

But, how would you change the design to get a depth of 4096 at 2nd and 3rd FC layers?

As you know, to get M Feature Map outpus from N input Feature Maps, we need MxN filters.

In the below image for example, with 3 Feature Map inputs, we need 3 filters to get 1 Feature Map output. But, if we need 6 FM outpus, we need 6 such sets of filters. In total, we need 6x3=18 filters.

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/detection_fm.gif" width="60%">
</p>

Similarly, to get a Feature Map output of depth 4096, at the 1st FC layer, we need 256*4096 filters. And each filter is of size 5x5. 256 - Since, the depth of AlexNet Conv layers is 256.

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/detection_overefeat_1x1_convolution_2.jpg" width="70%">
</p>

On the same lines, at the 2nd FC layer, we need to have 4096 filters, each of size 1x1.

The depth of the last FC layer, depends on the dataset. If we are using PascalVOC, it has 20 classes. We also need to account for the case, where the image patch contains no object or an unknown object. For this, we add one more class called ‘Background’.

So, in total we have 21 classes. To keep it generic, let us call this C.

Accordingly, the depth of last FC layer will be 4096*C and each filter will be of 1x1.


See detailed discussion [here](https://www.youtube.com/watch?v=JKTzkcaWfuk&list=PL1GQaVhO4f_jLxOokW7CS5kY_J1t1T17S&index=55)

## Overfeat Detection Network

For Object Detection, Overfeat used an Image Pyramid of 6 different scales as shown below. Accordingly, the sizes of the Conv Feature Maps and the output Feature Maps change.

It is in the Detection Network, that we get the Spatial Output. Overfeat does not use Image Pyramids for Classification.

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/detection_overfeata_detect.jpg" width="70%"></p>

As an example, I am showing the Network for an Image Pyramid of size 281x317. (Ignore the values for 245x245, it is only for reference). For this image size, we get a 2x3 Spatial Output.

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/detection_overefeat_1x1_convolution_3.jpg" width="70%"></p>

The Spatial Outputs for all other Image Sizes are shown below.

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/detection_overfeat_image_pyramid.jpg" width="70%"></p>

Here, we can see (not to scale), the Receptive field of different pixels of the Spatial Output for a 461x569 sized Image.

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/detection_overfeat_spatial_output.jpg" width="30%"></p>

## 1x1 Convolution, Model Size, Effective Stride

## 1x1 Convolution and Model Size of ConvNet

See detailed discussion on calculating Model size of ConvNet [here](https://www.youtube.com/watch?v=yLFe6TFE8L8&list=PL1GQaVhO4f_jLxOokW7CS5kY_J1t1T17S&index=58).

In the below image, we can see the size of FC layers when we use 1x1 Convolutions. It comes to around 45MB.

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/detection_1x1_convolution_model_size_0.jpg" width="70%"></p>

And, we get the same size, even if we use the dot product operations.

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/detection_1x1_convolution_model_size_1.jpg" width="70%"></p>


But, the advantage of 1x1 convolutions becomes significant when we take larger images.


Here, even for a 281x317 image, there is no increase in the model size for 1x1 Convolution case. But if FC layers are implemented as dot product operations, we get a significant increase in the model size (365MB).

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/detection_1x1_convolution_model_size_1.jpg" width="70%"></p>


So, that is how, the model size in Overfeat network remains constant irrespective of the size of the image that we use. This is the advantage of using 1x1 convolutions.

So, using 1x1 convolutions is one way of reducing the model sizes of your ConvNets, while not significantly compromising on the accuracy. This is something that can be kept in mind if you are designing any network.

## Effective Stride of a Network

This is one more concept that you should be aware of.

Effective Stride basically tells you, by how many pixels you are shifting your focus at the input side if you move by 1 pixel in the Spatial Output.

Ideally your Effective Stride should be as low as possible, to ensure you are scanning all possible areas of the image.

For example, the effective stride of this network is 4.

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/detection_effective_stride.jpg" width="70%"></p>

For this network, effective stride is 2.

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/detection_spatial_output.gif" width="50%"></p>

Effective Stride of Overfeat network is 36. But, if you want to improve this, that is, reduce the effective stride, you can employ a simple trick.

Below, you can see the input to the last Pooling layer in Overfeat. It is of size 15x15.

Overfeat uses a 3x3 pool with stride of 3 and 0 padding in the last pooling layer. With this, you get a 1x1 Spatial Output for a 245x245 image.

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/detection_improve_effective_stride.jpg" width="70%"></p>

And to improve the Effective Stride, you can change just the stride to 2 from 3. With this you will be able to get a 3x3 output. And this is what is done in Overfeat for higher accuracies.

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/detection_improve_effective_stride_1.jpg" width="70%"></p>


The modified Spatial Output sizes are shown below. Instead of 1x1xC, you will get a 3x3xC. Similarly for all other cases.

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/detection_overfeata_detect_1.jpg" width="70%"></p>

See detailed discussion [here](https://cogneethi.com/assets/images/evodn/detection_overfeata_detect_1.jpg)

## Post processing at Output side

## Confidence Score thresholding

Once we get the Spatial output, we will have multiple detections, each with different confidence scores. But the detections with low confidence scores will mostly be of background regions of the image and not that of any object. So we usually set a threshold of say 50% or 70% and eliminate all the bounding boxes that have lower scores than this.

## Non Max Suppression

After Confidence thresholding, there is one last problem remaining. That is, there will be multiple detections per object.

For example, in the below figure for a 2x3 Spatial output, we get 2 detections for the left cat and 4 for the right one. For each cat, only one of these detections is valid or more accurate.

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/detection_multiple.gif" width="50%"></p>



The question is, how do we select the best out of those multiple detections.

Option 1:

Select the bounding box with the highest confidence score.
And then remove all the boxes that overlap it.
This way, for the left cat, assuming that the red box has higher score than the yellow one, we eliminate the yellow one.

For the right cat, assuming that the green box has the highest score, we eliminate all the blue ones.

But this strategy may not work in all cases. Especially when the objects are close to each other. Consider the image below. The bounding boxes for both the persons overlap. (Left is Federer and right is Sachin)

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/detection_iou_nms_0.jpg" width="60%"></p>

So, if you pick, say the Green box and eliminate all the overlapping boxes, you will end up eliminating boxes even for Sachin. That way, you will miss detecting Sachin.

So instead, the sane thing to do is to only eliminate boxes that have a significant overlap with the selected box.

But, mathematically, how do you measure the amount of overlap? For this, we have a measure called Intersection over Union (IoU).

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/detection_iou.jpg" width="70%"></p>

Option 2: With this, we modify the technique slightly as:

* Select the sliding window location with the highest confidence score.
* And then remove all the boxes that overlap it with an IoU > 70%.

This technique is called **Non Max Suppression (NMS)**. This will be the result after applying NMS.

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/detection_iou_nms_1.jpg" width="70%"></p>



See detailed discussion on NMS [here](https://youtu.be/Uzg4eicmpO4)


## Summary

## Pre and Post processing

In general, we use:

* Sliding Windows to identify objects at different Locations
* Image Pyramid to identify objects of different sizes

These 2 techniques are applied at the input side.

At the output side, we will have multiple detections per object and some invalid detections mostly on the background regions of the image with very low confidence scores.

So, we first do a confidence score thresholding to eliminate detections on background regions of the image.
Then we apply NMS to get the one best detection for each of the objects in the image.
These are the 2 techniques used at the output side.

## Learning

This completes the discussion on one of the first Object Detection Networks, Overfeat. You not only learnt about the network design, you also learnt many concepts, that are generic to CNNs like:

* Receptive Field
* Implementing FC layers as convolution operation
* ConvNet’s Sliding Window Efficiency
* 1x1 Convolution
* Spatial Output
* Effective Stride
* Confidence score thresholding
* Non Max Suppression and IoU

These concepts are pretty generic and you will come across them in many other papers.

<p>
   <h3 align="center">GO TO THE NEXT SECTION -  <a href="../04-Object_Detection_Metric/ReadME.md" style="text-decoration:none">Object Detection Metric ➡️ </a> </h3>
</p>