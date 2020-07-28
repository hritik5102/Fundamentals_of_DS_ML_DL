### **Introduction of object detection**
---

### **Difference between image classification and object detection** 

<p align="center"><img src="https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1522766480/1_6j34dAOTijqP6HDFnjxPFA_udggex.png" width="60%"/>

When performing standard image classification, given an input image, we present it to our neural network, and we obtain a single class label and perhaps a probability associated with the class label as well.

This class label is meant to characterize the contents of the entire image, or at least the most dominant, visible contents of the image.

For example, given the input image in Figure 1 above (left) our CNN has labeled the image as “Cat”.

We can thus think of image classification as:

* One image in
* And one class label out

Object detection, regardless of whether performed via deep learning or other computer vision techniques, builds on image classification and seeks to localize exactly where in the image each object appears.

When performing object detection, given an input image, we wish to obtain:

* A list of bounding boxes, or the (x, y)-coordinates for each object in an image.
* The class label associated with each bounding box
* The probability/confidence score associated with each bounding box.

Figure 1 (right) demonstrates an example of performing deep learning object detection. Notice how both the cat , dog and the duck are localized with their bounding boxes and class labels predicted.

Therefore, object detection allows us to:

Present one image to the network
And obtain multiple bounding boxes and class labels out

3 Approaches of solving object detection problem 

* HOG (Histogram of gradient) + SVM
* DPM (Deformable part model) 
* R-CNN (regions with convolutional neural networks)
* Fast RCNN
* Faster RCNN
* Yolo (You only looks once)
* SSD (Single short detector)

<br/>

---
### **Object detection with HOG + SVM**
---

```
  what a feature descript or does ?

  It is a simplified representation of the image that contains only the most important information about the image.
```

There are a number of feature descriptors out there. Here are a few of the most popular ones:

* HOG: Histogram of Oriented Gradients 
* SIFT: Scale Invariant Feature Transform
* SURF: Speeded-Up Robust Feature

---

**Introduction to the HOG Feature Descriptor**

<p align="center"><img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2019/08/hog-feature.jpg" width="50%"/>

HOG, or Histogram of Oriented Gradients, is a feature descriptor that is often used to extract features from image data. It is widely used in computer vision tasks for object detection.

Let’s look at some important aspects of HOG that makes it different from other feature descriptors:

The HOG descriptor focuses on the structure or the shape of an object. Now you might ask, how is this different from the edge features we extract for images? In the case of edge features, we only identify if the pixel is an edge or not. HOG is able to provide the edge direction as well. This is done by extracting the gradient and orientation (or you can say magnitude and direction) of the edges
Additionally, these orientations are calculated in ‘localized’ portions. This means that the complete image is broken down into smaller regions and for each region, the gradients and orientation are calculated.

HOG measure 2 thing i.e gradient magnitude + Direction (orientation) combining this 2 will give us feature vector.

   if we move from low intensity to high intensity region , then the change in the intensity is measured by gradient magnitude.

<p align="center"><img src="https://www.researchgate.net/profile/Marco_Leo/publication/283543555/figure/fig10/AS:293969532604440@1447099206376/HOG-features-extraction-process-image-is-divided-in-cells-of-size-N-N-pixels-The.png" width="70%"/>

In Object detection, we create a feature vector for each block i.e (8x8 pixel blocks)

In each block we compute a histogram of gradient orientations<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; - Invariant to changes in lighting, small deformations, etc.

<p align="center"><img src="https://i.ibb.co/4Y2mxcb/Feature-Vector.jpg" alt="Feature-Vector" border="0" width="70%">

Finally the HOG would generate a Histogram for each of these regions separately. The histograms are created using the gradients and orientations of the pixel values, hence the name ‘Histogram of Oriented Gradients’.

  <p align="center"><img src="https://www.researchgate.net/profile/Vivienne_Sze/publication/267868361/figure/fig2/AS:295485538619402@1447460650791/Object-detection-algorithm-using-HOG-features.png" />


For Detail explanation :

* [C34 - HOG Feature Vector Calculation | Computer Vision | Object Detection | EvODN](https://www.youtube.com/watch?v=28xk5i1_7Zc&list=PL1GQaVhO4f_jLxOokW7CS5kY_J1t1T17S&index=20)

* [Feature engineering images introduction hog feature descriptor](https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/) 

* [Histogram of oriented gradients](https://www.learnopencv.com/histogram-of-oriented-gradients/)

---
### **Overview of Object detection using HOG + SVM**
---

The Histogram of Oriented Gradients method suggested by Dalal and Triggs in their seminal 2005 paper, **[Histogram of Oriented Gradients for Human Detection](http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf)** demonstrated that the Histogram of Oriented Gradients (HOG) image descriptor and a Linear Support Vector Machine (SVM) could be used to train highly accurate object classifiers — or in their particular study, human detectors.

Let us suppose we are doing padestrian detection.

**Steps include**

* We apply hog feature extractor on the image , then hog feature extractor will extract feature from the image.

* Now we apply template over the extracted features .
   the template is nothing but hog feature of human which act as filter. 

* So we get some output we apply SVM over the output to check whether 
   their is human or padestrian present or not.

<p align="center"><img src="https://i.ibb.co/DQ7MHg5/Screenshot-509.png" alt="Screenshot-509" border="0" width="70%">

So this is just the classification , now how can we get the bounding box over padestrian i.e we need x,y w, h of the bounding box

**`Question`** : So how can we get the bouding box over the detected padestrian in the input image ?

**`Answer`** : We will create a window and move the window over the image. we move patch (i.e. window) 
	from left to right , with stride = 1.
	
  <p align="center"><img src="https://pyimagesearch.com/wp-content/uploads/2014/10/sliding_window_example.gif"/>

  so for every patch, we will compute hog feature and then classify them using svm to check whether they are padestrian or not.
  
  so the above process will be repeated for the each patch of the input image.
  
  if the patch detect the padestrian then we find the x and y position of that patch and w,h will be width and height of that patch.

**Second Problem**

**`Question`** : How should we get the window of larger padestrian i.e if perticular human in the image is bigger then window size then it would be a problem.

**`Answer`** : Using **Image pyramid** we can solve that problem.

---
### **Image pyramid**
---

An “image pyramid” is a multi-scale representation of an image.

Utilizing an image pyramid allows us to find objects in images at different scales of an image. And when combined with a sliding window we can find objects in images in various locations.

<p align="center"><img src="https://i.ibb.co/vkFgdmm/Screenshot-522.png" alt="Screenshot-522" border="0" width="70%">


We downscale the image and repeat the sliding window technique. 
such that we can align the human inside the window.

It form a pyramid like structure i.e from larger image in the left to smaller image in the right.

<p align="center"><img src="https://www.pyimagesearch.com/wp-content/uploads/2015/03/pyramid_example.png" width="40%"/>

At the bottom of the pyramid we have the original image at its original size (in terms of width and height). And at each subsequent layer, the image is resized (subsampled) and optionally smoothed (usually via Gaussian blurring).

The image is progressively subsampled until some stopping criterion is met, which is normally a minimum size has been reached and no further subsampling needs to take place.

<p align="center"><img src="https://pyimagesearch.com/wp-content/uploads/2015/03/sliding-window-animated-sot.gif" width="30%"/>

See the reference video [how sliding window + image pyramid works](https://www.youtube.com/watch?v=ukKKz0moFEo)

#### **`Method #1: The traditional object detection pipeline`**

  The first method is not a pure end-to-end deep learning object detector.

  We instead utilize:

  * Fixed size sliding windows, which slide from left-to-right and top-to-bottom to localize objects at different locations
  * An image pyramid to detect objects at varying scales
  * Classification via a pre-trained (classification) Convolutional Neural Network
  At each stop of the sliding window + image pyramid, we extract the ROI, feed it into a CNN, and obtain the output classification for the ROI.

  <p align="center"><img src="https://d3i71xaburhd42.cloudfront.net/c5543534fa4596d98e564ab9f792e8d97bfedb7a/1-Figure1-1.png" width="50%"/>

  If the classification probability of label L is higher than some threshold T, we mark the bounding box of the ROI as the label (L). Repeating this process for every stop of the/ sliding window and image pyramid, we obtain the output object detectors. Finally, we apply non-maxima suppression to the bounding boxes yielding our final output detections.

<p align="center"><img src="https://media.geeksforgeeks.org/wp-content/uploads/20200219163223/nonmaxsuppression.jpg" width="50%"/>

---
### **Working of NMS (Non Maximum Suppression)**
---

**Input**: A list of Proposal boxes B, corresponding confidence scores S and overlap threshold N.<br/>
**Output**: A list of filtered proposals D.

1. Select the proposal with highest confidence score, remove it from B and add it to the final proposal list D. (Initially D is empty).

2. Now compare this proposal with all the proposals — calculate the IOU (Intersection over Union) of this proposal with every other proposal. If the IOU is greater than the threshold N, remove that proposal from B.
let us say are thresold is 70% then we remove all the boxes with overlap it with IOU > 70%.

  **`NOTE`** : In NMS we compare it with boxes of the same class.

3. Again take the proposal with the highest confidence from the remaining proposals in B and remove it from B and add it to D.
4. Once again calculate the IOU of this proposal with all the proposals in B and eliminate the boxes which have high IOU than threshold.
5. This process is repeated until there are no more proposals left in B.

<p align="center"><img src="https://images2017.cnblogs.com/blog/606386/201708/606386-20170826152918652-24374253.png" width="50%"/>

#### **Algorithm** :

    select only rectangles above a confidence threshold 
    sort the thresholded rectangles in descending order
    create an empty set of kept rectangle
    loop over the sorted thresholded rectangles:
      loop over the set of kept rectangles:
        compute IOU between the rectangles
        if IOU is above IOU threshold break loop
      if all IOU are below the IOU threshold add to kept

---
### **How will you perform Object detection using HOG + SVM ?**
---

**Steps**

1. Get the dataset which contain set of 2 folder <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    - Positive sample <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;    - negative sample

2. Label Positive sample as 1 and negative sample as 0 or vice varsa.

3. Now apply the sliding window technique and slide your window across the image. At each window compute your **`HOG descriptors`** , get the feature vector (x) and mutliply that with the feature vector of object template or svm coeff (w) and then add bias term (b).
  ```
     y = x * w + b
  ```
  and then apply classifier , then classifier predict 1 (Human) or 0 (Not human) 

  <p align="center"><img src="https://i.ibb.co/Hh3FB2W/Object-detection-SVM.jpg" alt="Object-detection-SVM" border="0" width="60%">
  
  If classifier predict the object correctly then it is **`True Positive`** case
  If classifier predict the false object , as a true object then it is **`False Positive`** case.

4. So step 3 is repeated for each patch or window in the image. Each patch will tell class probability , confidence score and bounding box.

5. So now we get some bounding boxes over full resolution image , now we downscale the image by half and repeat step 2,3,4 again we get some bounding box on downscaled image. 

  we store bouding box which have confidence score above some thresold value.

  that's how we perform both **`sliding window`** and **`image pyramid`** on the input image. 

6. As we get lot of bouding boxes on the image we want only one bounding box for perticular class. we apply **`NMS (Non maximum suppression technique)`**
to get only one bounding box.

  See the [demo video](https://www.youtube.com/watch?v=SPXocFBjr70)
---

**`Question`** : According to the paper published by Dalal and Triggs, they suggest gamma correction is not necessary. I have the doubt about whether correcting gamma is a good option to go for or not. If Gamma correction is necessary what is the gamma value I have to take for better performance ?

[what is gamma correction ?](https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/)

**`Answer`** : 

log-based gamma correction normally hurts HOG + Linear SVM performance. The square-root (in most cases) or simple variance normalization is better option.

You can normalize by either taking the log or the square-root of the image channel before applying the HOG descriptor (normally the square-root is used). Another method to make HOG more robust is to compute the gradient magnitude over all channels of the RGB or L*a*b* image and take the maximum gradient response across all channels before forming the histogram.

---

### **`NOTE`** :

For rotated or skewed image , it will be harder for HOG + Linear SVM to return good results (since HOG is not rotation invariant).

If you wanted to use HOG + Linear SVM for rotated objects you would have to train a separate object detector for each rotation interval, probably in the range of 5-30 degrees depending on your objects.

It will give some false prediction for human or padestrian prediction because human body are deformable .  

**References** :

* [gentle guide to deep learning object detection](https://www.pyimagesearch.com/2018/05/14/a-gentle-guide-to-deep-learning-object-detection/)

* [histogram oriented gradients object detection](https://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/)

* [image pyramids with python and opencv](https://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/)

---
### **Deformable part model**
---

DPM is also known as object detection with discriminately trained parts based model.

Dalal and triggs detector is special case of DPM.

You can see in the below image that some of the padestrian are not properly detected.

<p align="center"><img src="https://i.ibb.co/XJbpDjj/Dalal-Thesis.png" alt="Dalal-Thesis" border="0" width="60%">

DPM overcome the problem of Dalal & triggs model problem. Instead of considering only whole object as filter or template (eg. Human filter) DPM also uses the part of objects as filter or template (eg hand filters,legs filter , head filter) for detection.

```
So DPM model uses both global template + part templates - Panelty.
```
Panelty ensure the gap between the 2 same class object . if object found in the part templates then it would be fine but they found far from part template then thier would be panelty.


The Deformable Parts Model (DPM) (Felzenszwalb et al., 2010) recognizes objects with a mixture graphical model (Markov random fields) of deformable parts. The model consists of three major components:

* A coarse **root filter** defines a detection window that approximately covers an entire object. A filter specifies weights for a region feature vector.

* Multiple **part filters** that cover smaller parts of the object. Parts filters are learned at twice resolution of the root filter.

* A **spatial model** for scoring the locations of part filters relative to the root.

<p align="center"><img src="https://lilianweng.github.io/lil-log/assets/images/DPM.png" width="60%"/>

Over all structure of DPM Object detection

<p align="center"><img src="https://lilianweng.github.io/lil-log/assets/images/DPM-matching.png" width="60%"/>

**References**

* [lilianweng - object recognition for dummies part 2](https://lilianweng.github.io/lil-log/2017/12/15/object-recognition-for-dummies-part-2.html)

* [C3.10 - DPM | Deformable Parts Model | Object Detection | Machine Learning | Computer Vision | EvODN](https://www.youtube.com/watch?v=vAPH6zz4JKc&list=PL1GQaVhO4f_jLxOokW7CS5kY_J1t1T17S&index=26)

* [PyImageSearch - sliding windows for object detection with python and opencv](https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/)

* [The AI Summer - Localization and Object Detection](https://theaisummer.com/Localization_and_Object_Detection/)

* [Machine learning mastery - object recognition with deep learning](https://machinelearningmastery.com/object-recognition-with-deep-learning/)

* [Medium - Deeplearning series objection detection and localization yolo algorithm r cnn](https://medium.com/machine-learning-bites/deeplearning-series-objection-detection-and-localization-yolo-algorithm-r-cnn-71d4dfd07d5f)

* [Analyticsvidhya - First multi label image classification model python](https://www.analyticsvidhya.com/blog/2019/04/build-first-multi-label-image-classification-model-python/)

<p>
   <h3 align="center">GO TO THE NEXT SECTION -  <a href="../02-Localization_and_Detection/ReadME.md" style="text-decoration:none">Localization and Detection ➡️ </a> </h3>
</p>