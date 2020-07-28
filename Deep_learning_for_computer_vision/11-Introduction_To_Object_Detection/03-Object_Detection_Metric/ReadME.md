### **Metrics for object detection**

- [Different competitions, different metrics](#different-competitions-different-metrics)

- [Important definitions](#important-definitions)

- [Intersection Over Union (IOU)](#intersection-over-union-iou)

- [True Positive, False Positive, False Negative and True Negative](#true-positive-false-positive-false-negative-and-true-negative)

  - [Precision](#precision)
  - [Recall](#recall)

- [Metrics](#metrics)

  - [Precision x Recall curve](#precision-x-recall-curve)
  - [Average Precision](#average-precision)
    - [11-point interpolation](#11-point-interpolation)
    - [Interpolating all points](#interpolating-all-points)
    - [An ilustrated example](#an-ilustrated-example)
    - [Calculating the 11-point interpolation](#calculating-the-11-point-interpolation)
    - [Calculating the interpolation performed in all points](#calculating-the-interpolation-performed-in-all-points)
  - [References](#references)


<a name="different-competitions-different-metrics"></a> 

## Different competitions, different metrics 

* **[PASCAL VOC Challenge](http://host.robots.ox.ac.uk/pascal/VOC/)** offers a Matlab script in order to evaluate the quality of the detected objects. Participants of the competition can use the provided Matlab script to measure the accuracy of their detections before submitting their results. The official documentation explaining their criteria for object detection metrics can be accessed [here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#SECTION00050000000000000000). The current metrics used by the current PASCAL VOC object detection challenge are the **Precision x Recall curve** and **Average Precision**.  
The PASCAL VOC Matlab evaluation code reads the ground truth bounding boxes from XML files, requiring changes in the code if you want to apply it to other datasets or to your speficic cases. Even though projects such as [Faster-RCNN](https://github.com/rbgirshick/py-faster-rcnn) implement PASCAL VOC evaluation metrics, it is also necessary to convert the detected bounding boxes into their specific format. [Tensorflow](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/evaluation_protocols.md) framework also has their PASCAL VOC metrics implementation.  

* **[COCO Detection Challenge](https://competitions.codalab.org/competitions/5181)** uses different metrics to evaluate the accuracy of object detection of different algorithms. [Here](http://cocodataset.org/#detection-eval) you can find a documentation explaining the 12 metrics used for characterizing the performance of an object detector on COCO. This competition offers Python and Matlab codes so users can verify their scores before submitting the results. It is also necessary to convert the results to a [format](http://cocodataset.org/#format-results) required by the competition.  

* **[Google Open Images Dataset V4 Competition](https://storage.googleapis.com/openimages/web/challenge.html)** also uses mean Average Precision (mAP) over the 500 classes to evaluate the object detection task. 

* **[ImageNet Object Localization Challenge](https://www.kaggle.com/c/imagenet-object-detection-challenge)** defines an error for each image considering the class and the overlapping region between ground truth and detected boxes. The total error is computed as the average of all min errors among all test dataset images. [Here](https://www.kaggle.com/c/imagenet-object-localization-challenge#evaluation) are more details about their evaluation method.  

## Important definitions  

## **Intersection Over Union (IOU)**

Intersection Over Union (IOU) is measure based on Jaccard Index that evaluates the overlap between two bounding boxes. It requires a ground truth bounding box ![](http://latex.codecogs.com/gif.latex?B_%7Bgt%7D) and a predicted bounding box ![](http://latex.codecogs.com/gif.latex?B_p). By applying the IOU we can tell if a detection is valid (True Positive) or not (False Positive).  highest
IOU is given by the overlapping area between the predicted bounding box and the ground truth bounding box divided by the area of union between them:  

<p align="center"> 
<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRdqY1aM6AOYS4DhN6t6klvfa0qRJCTgrpbNA&usqp=CAU" width="20%">
</p>

<!---
\text{IOU}=\frac{\text{area}\left(B_{p} \cap B_{gt} \right)}{\text{area}\left(B_{p} \cup B_{gt} \right)} 
--->

The image below illustrates the IOU between a ground truth bounding box (in green) and a detected bounding box (in red).

<!--- IOU --->
<p align="center">
<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSaJv7-c-5s5V6IDcSRWq_kmnfVnqgG0Fz4Cw&usqp=CAU" width="30%" align="center"/></p>

## **True Positive, False Positive, False Negative and True Negative**  

Some basic concepts used by the metrics:  

* **True Positive (TP)**: A correct detection. Detection with IOU ≥ _threshold_  

* **False Positive (FP)**: A wrong detection. Detection with IOU < _threshold_  

* **False Negative (FN)**: A ground truth not detected  

* **True Negative (TN)**: Does not apply. It would represent a corrected misdetection. In the object detection task there are many possible bounding boxes that should not be detected within an image. Thus, TN would be all possible bounding boxes that were corrrectly not detected (so many possible boxes within an image). That's why it is not used by the metrics.

_threshold_: depending on the metric, it is usually set to 50%, 75% or 95%.

### **Precision**

Precision is the ability of a model to identify **only** the relevant objects. It is the percentage of correct positive predictions and is given by:

<p align="center"> 
<img src="https://latex.codecogs.com/gif.latex?%5Ctext%7BPrecision%7D%20%3D%20%5Cfrac%7B%5Ctext%7BTP%7D%7D%7B%5Ctext%7BTP%7D&plus;%5Ctext%7BFP%7D%7D%3D%5Cfrac%7B%5Ctext%7BTP%7D%7D%7B%5Ctext%7Ball%20detections%7D%7D">
</p>

<!---
\text{Precision} = \frac{\text{TP}}{\text{TP}+\text{FP}}=\frac{\text{TP}}{\text{all detections}}
--->

### **Recall** 

Recall is the ability of a model to find all the relevant cases (all ground truth bounding boxes). It is the percentage of true positive detected among all relevant ground truths and is given by:

<p align="center"> 
<img src="https://latex.codecogs.com/gif.latex?%5Ctext%7BRecall%7D%20%3D%20%5Cfrac%7B%5Ctext%7BTP%7D%7D%7B%5Ctext%7BTP%7D&plus;%5Ctext%7BFN%7D%7D%3D%5Cfrac%7B%5Ctext%7BTP%7D%7D%7B%5Ctext%7Ball%20ground%20truths%7D%7D">
</p>

<!--- 
\text{Recall} = \frac{\text{TP}}{\text{TP}+\text{FN}}=\frac{\text{TP}}{\text{all ground truths}}
--->

## **Metrics**

In the topics below there are some comments on the most popular metrics used for object detection.

### **Precision x Recall curve**

The Precision x Recall curve is a good way to evaluate the performance of an object detector as the confidence is changed by plotting a curve for each object class. An object detector of a particular class is considered good if its precision stays high as recall increases, which means that if you vary the confidence threshold, the precision and recall will still be high. Another way to identify a good object detector is to look for a detector that can identify only relevant objects (0 False Positives = high precision), finding all ground truth objects (0 False Negatives = high recall).  

A poor object detector needs to increase the number of detected objects (increasing False Positives = lower precision) in order to retrieve all ground truth objects (high recall). That's why the Precision x Recall curve usually starts with high precision values, decreasing as recall increases. You can see an example of the Prevision x Recall curve in the next topic (Average Precision). This kind of curve is used by the PASCAL VOC 2012 challenge and is available in our implementation.  

### **Average Precision**

Another way to compare the performance of object detectors is to calculate the area under the curve (AUC) of the Precision x Recall curve. As AP curves are often zigzag curves going up and down, comparing different curves (different detectors) in the same plot usually is not an easy task - because the curves tend to cross each other much frequently. That's why Average Precision (AP), a numerical metric, can also help us compare different detectors. In practice AP is the precision averaged across all recall values between 0 and 1.  

From 2010 on, the method of computing AP by the PASCAL VOC challenge has changed. Currently, **the interpolation performed by PASCAL VOC challenge uses all data points, rather than interpolating only 11 equally spaced points as stated in their [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.5766&rep=rep1&type=pdf)**. As we want to reproduce their default implementation, our default code (as seen further) follows their most recent application (interpolating all data points). However, we also offer the 11-point interpolation approach. 

#### 11-point interpolation

The 11-point interpolation tries to summarize the shape of the Precision x Recall curve by averaging the precision at a set of eleven equally spaced recall levels [0, 0.1, 0.2, ... , 1]:

<p align="center">
<img src="https://latex.codecogs.com/gif.latex?%5Ctext%7BAP%7D%3D%5Cfrac%7B1%7D%7B11%7D%20%5Csum_%7Br%5Cin%20%5Cleft%20%5C%7B%200%2C%200.1%2C%20...%2C1%20%5Cright%20%5C%7D%7D%5Crho_%7B%5Ctext%7Binterp%7D%5Cleft%20%28%20r%20%5Cright%20%29%7D">
</p>
<!---
\text{AP}=\frac{1}{11} \sum_{r\in \left \{ 0, 0.1, ...,1 \right \}}\rho_{\text{interp}\left ( r \right )}
--->

with

<p align="center"> 
<img src="https://latex.codecogs.com/gif.latex?%5Crho_%7B%5Ctext%7Binterp%7D%7D%20%3D%20%5Cmax_%7B%5Ctilde%7Br%7D%3A%5Ctilde%7Br%7D%20%5Cgeq%20r%7D%20%5Crho%5Cleft%20%28%20%5Ctilde%7Br%7D%20%5Cright%20%29">
</p>
<!--- 
\rho_{\text{interp}} = \max_{\tilde{r}:\tilde{r} \geq r} \rho\left ( \tilde{r} \right )
--->

where ![](http://latex.codecogs.com/gif.latex?%5Crho%5Cleft%20%28%20%5Ctilde%7Br%7D%20%5Cright%20%29) is the measured precision at recall ![](http://latex.codecogs.com/gif.latex?%5Ctilde%7Br%7D).

Instead of using the precision observed at each point, the AP is obtained by interpolating the precision only at the 11 levels ![](http://latex.codecogs.com/gif.latex?r) taking the **maximum precision whose recall value is greater than ![](http://latex.codecogs.com/gif.latex?r)**.

#### Interpolating all points

Instead of interpolating only in the 11 equally spaced points, you could interpolate through all points <img src="https://latex.codecogs.com/gif.latex?n"> in such way that:

<p align="center"> 
<img src="https://latex.codecogs.com/gif.latex?%5Csum_%7Bn%3D0%7D%20%5Cleft%20%28%20r_%7Bn&plus;1%7D%20-%20r_%7Bn%7D%20%5Cright%20%29%20%5Crho_%7B%5Ctext%7Binterp%7D%7D%5Cleft%20%28%20r_%7Bn&plus;1%7D%20%5Cright%20%29">
</p>
<!---
\sum_{n=0} \left ( r_{n+1} - r_{n} \right ) \rho_{\text{interp}}\left ( r_{n+1} \right )
--->
 
with

<p align="center"> 
<img src="https://latex.codecogs.com/gif.latex?%5Crho_%7B%5Ctext%7Binterp%7D%7D%5Cleft%20%28%20r_%7Bn&plus;1%7D%20%5Cright%20%29%20%3D%20%5Cmax_%7B%5Ctilde%7Br%7D%3A%5Ctilde%7Br%7D%20%5Cge%20r_%7Bn&plus;1%7D%7D%20%5Crho%20%5Cleft%20%28%20%5Ctilde%7Br%7D%20%5Cright%20%29">
</p>

<!---
\rho_{\text{interp}}\left ( r_{n+1} \right ) = \max_{\tilde{r}:\tilde{r} \ge r_{n+1}} \rho \left ( \tilde{r} \right )
--->


where ![](http://latex.codecogs.com/gif.latex?%5Crho%5Cleft%20%28%20%5Ctilde%7Br%7D%20%5Cright%20%29) is the measured precision at recall ![](http://latex.codecogs.com/gif.latex?%5Ctilde%7Br%7D).

In this case, instead of using the precision observed at only few points, the AP is now obtained by interpolating the precision at **each level**, ![](http://latex.codecogs.com/gif.latex?r) taking the **maximum precision whose recall value is greater or equal than ![](http://latex.codecogs.com/gif.latex?r&plus;1)**. This way we calculate the estimated area under the curve.

To make things more clear, we provided an example comparing both interpolations.


#### **An ilustrated example** 

An example helps us understand better the concept of the interpolated average precision. Consider the detections below:
  
<!--- Image samples 1 --->
<p align="center">
<img src="https://i.ibb.co/KrzJFP6/samples-1-v2.png" alt="samples-1-v2" border="0"align="center"/></p>
  
There are 7 images with 15 ground truth objects representented by the green bounding boxes and 24 detected objects represented by the red bounding boxes. Each detected object has a confidence level and is identified by a letter (A,B,...,Y).  

The following table shows the bounding boxes with their corresponding confidences. The last column identifies the detections as TP or FP. In this example a TP is considered if IOU ![](http://latex.codecogs.com/gif.latex?%5Cgeq) 30%, otherwise it is a FP. By looking at the images above we can roughly tell if the detections are TP or FP.

<!--- Table 1 --->
<p align="center">
<img src="https://i.ibb.co/r4PbbPh/table-1-v2.png" alt="table-1-v2" border="0" align="center"/></p>


In some images there are more than one detection overlapping a ground truth (Images 2, 3, 4, 5, 6 and 7). For those cases the first detection is considered TP while the others are FP. This rule is applied by the PASCAL VOC 2012 metric: "e.g. 5 detections (TP) of a single object is counted as 1 correct detection and 4 false detections”.

The Precision x Recall curve is plotted by calculating the precision and recall values of the accumulated TP or FP detections. For this, first we need to order the detections by their confidences, then we calculate the precision and recall for each accumulated detection as shown in the table below: 

<!--- Table 2 --->
<p align="center">
<img src="https://i.ibb.co/XV5ccyf/table-2-v2.png" alt="table-2-v2" border="0" align="center"></p>


 Plotting the precision and recall values we have the following *Precision x Recall curve*:
 
 <!--- Precision x Recall graph --->
<p align="center">
<img src="https://i.ibb.co/S6vq7Qr/precision-recall-example-1-v2.png" alt="precision-recall-example-1-v2" border="0" align="center"/>
</p>
 
As mentioned before, there are two different ways to measure the interpolted average precision: **11-point interpolation** and **interpolating all points**. Below we make a comparisson between them:

#### Calculating the 11-point interpolation

The idea of the 11-point interpolated average precision is to average the precisions at a set of 11 recall levels (0,0.1,...,1). The interpolated precision values are obtained by taking the maximum precision whose recall value is greater than its current recall value as follows: 

<!--- interpolated precision curve --->
<p align="center">
<img src="https://i.ibb.co/DCVSmtv/11-point-Interpolation.png" alt="11-point-Interpolation" border="0" align="center"/>
</p>

By applying the 11-point interpolation, we have:  

![](http://latex.codecogs.com/gif.latex?AP%20%3D%20%5Cfrac%7B1%7D%7B11%7D%5Csum_%7Br%5Cin%5C%7B0%2C0.1%2C...%2C1%5C%7D%7D%5Crho_%7B%5Ctext%7Binterp%7D%5Cleft%20%28r%5Cright%20%29%7D)  
![](http://latex.codecogs.com/gif.latex?AP%20%3D%20%5Cfrac%7B1%7D%7B11%7D%20%5Cleft%20%28%201&plus;0.6666&plus;0.4285&plus;0.4285&plus;0.4285&plus;0&plus;0&plus;0&plus;0&plus;0&plus;0%20%5Cright%20%29)  
![](http://latex.codecogs.com/gif.latex?AP%20%3D%2026.84%5C%25)


#### Calculating the interpolation performed in all points

By interpolating all points, the Average Precision (AP) can be interpreted as an approximated AUC of the Precision x Recall curve. The intention is to reduce the impact of the wiggles in the curve. By applying the equations presented before, we can obtain the areas as it will be demostrated here. We could also visually have the interpolated precision points by looking at the recalls starting from the highest (0.4666) to 0 (looking at the plot from right to left) and, as we decrease the recall, we collect the precision values that are the highest as shown in the image below:
    
<!--- interpolated precision AUC --->
<p align="center">
<img src="https://i.ibb.co/DCTv7bS/interpolated-precision-v2.png" alt="interpolated-precision-v2" border="0" align="center"/>
</p>
  
Looking at the plot above, we can divide the AUC into 4 areas (A1, A2, A3 and A4):
  
<!--- interpolated precision AUC --->
<p align="center">
<img src="https://i.ibb.co/MRxzCwz/interpolated-precision-AUC-v2.png" alt="interpolated-precision-AUC-v2" border="0" align="center"/>
</p>

Calculating the total area, we have the AP:  

![](http://latex.codecogs.com/gif.latex?AP%20%3D%20A1%20&plus;%20A2%20&plus;%20A3%20&plus;%20A4)  
  
![](http://latex.codecogs.com/gif.latex?%5Ctext%7Bwith%3A%7D)  
![](http://latex.codecogs.com/gif.latex?A1%20%3D%20%280.0666-0%29%5Ctimes1%20%3D%5Cmathbf%7B0.0666%7D)  
![](http://latex.codecogs.com/gif.latex?A2%20%3D%20%280.1333-0.0666%29%5Ctimes0.6666%3D%5Cmathbf%7B0.04446222%7D)  
![](http://latex.codecogs.com/gif.latex?A3%20%3D%20%280.4-0.1333%29%5Ctimes0.4285%20%3D%5Cmathbf%7B0.11428095%7D)  
![](http://latex.codecogs.com/gif.latex?A4%20%3D%20%280.4666-0.4%29%5Ctimes0.3043%20%3D%5Cmathbf%7B0.02026638%7D)  
   
![](http://latex.codecogs.com/gif.latex?AP%20%3D%200.0666&plus;0.04446222&plus;0.11428095&plus;0.02026638)  
![](http://latex.codecogs.com/gif.latex?AP%20%3D%200.24560955)  
![](http://latex.codecogs.com/gif.latex?AP%20%3D%20%5Cmathbf%7B24.56%5C%25%7D)  

The results between the two different interpolation methods are a little different: 24.56% and 26.84% by the every point interpolation and the 11-point interpolation respectively.  

  
### **References**

* [Object detection metric - rafaelpadilla](https://github.com/rafaelpadilla/Object-Detection-Metrics/blob/master/README.md)

* [Evolution of Object Detection Networks -  Cogneethi](https://www.youtube.com/playlist?list=PL1GQaVhO4f_jLxOokW7CS5kY_J1t1T17S)

* The Relationship Between Precision-Recall and ROC Curves (Jesse Davis and Mark Goadrich)
Department of Computer Sciences and Department of Biostatistics and Medical Informatics, University of
Wisconsin  
http://pages.cs.wisc.edu/~jdavis/davisgoadrichcamera2.pdf

* The PASCAL Visual Object Classes (VOC) Challenge  
http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.157.5766&rep=rep1&type=pdf

* Evaluation of ranked retrieval results (Salton and Mcgill 1986)  
https://www.amazon.com/Introduction-Information-Retrieval-COMPUTER-SCIENCE/dp/0070544840  
https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html
