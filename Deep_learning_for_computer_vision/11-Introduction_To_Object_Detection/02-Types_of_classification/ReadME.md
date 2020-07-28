
---
### **Type of classification**
---

* Binary classification
* Multiclass classification
* Multilabel classification

---

### **Binary classification**

<p align="center"><img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2019/04/Screenshot-from-2019-04-11-13-18-46.png" width="60%"/>

The object in image 1 is a car. That was a no-brainer. Whereas, there is no car in image 2 – only a group of buildings. Can you see where we are going with this? We have classified the images into two classes, i.e., car or non-car.

**Defination of Binary classification** 

```
When we have only two classes in which the images can be classified, this is known as a binary image classification problem.
```

---

### **How is Multi-Label Image Classification different from Multi-Class Image Classification ?**


Suppose we are given images of animals to be classified into their corresponding categories. For ease of understanding, let’s assume there are a total of 4 categories (cat, dog, rabbit and parrot) in which a given image can be classified. Now, there can be two scenarios:

Each image contains only a single object (either of the above 4 categories) and hence, it can only be classified in one of the 4 categories
The image might contain more than one object (from the above 4 categories) and hence the image will belong to more than one category
Let’s understand each scenario through examples, starting with the first one:

<p align="center"><img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2019/04/Screenshot-from-2019-04-05-12-35-21.png" width="60%"/>

Here, we have images which contain only a single object. The keen-eyed among you will have noticed there are 4 different types of objects (animals) in this collection.

Each image here can only be classified either as a cat, dog, parrot or rabbit. There are no instances where a single image will belong to more than one category.

**Defination of multiclass classification**

```
1. When there are more than two categories in which the images can be classified, and
2. An image does not belong to more than one category

If both of the above conditions are satisfied, it is referred to as a multi-class image classification problem.
```

### **Multilabel classification**

Now, let’s consider the second scenario – check out the below images:

<p align="center"><img src="https://cdn.analyticsvidhya.com/wp-content/uploads/2019/04/Screenshot-from-2019-04-09-10-30-23.png" width="60%"/>

* First image (top left) contains a dog and a cat
* Second image (top right) contains a dog, a cat and a parrot
* Third image (bottom left) contains a rabbit and a parrot, and
* The last image (bottom right) contains a dog and a parrot

**Defination of multi-label classification**

```
Each image here belongs to more than one class and hence it is a multi-label image classification problem.
```

<p align="center"><img src="https://www.microsoft.com/en-us/research/uploads/prod/2017/12/40250.jpg" width="50%"/>

---
#### **classification + localization**
---

Image classification involves assigning a class label to an image, whereas object localization involves drawing a bounding box around one or more objects in an image. Object detection is more challenging and combines these two tasks and draws a bounding box around each object of interest in the image and assigns them a class label. Together, all of these problems are referred to as object recognition.


In these case for image classification we have one object in the image and in classification with localization thier also we only one object (of perticular cateogry) per image and we have to localize that object in the image.


<p align="center"><img src="https://miro.medium.com/proxy/1*vaAAxQh_pNIe-Tu0K-9i1A.png" width="60%"/>

**Image Classification** : Predict the type or class of an object in an image.
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **Input** : An image with a single object, such as a photograph.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **Output** : A class label (e.g. one or more integers that are mapped to class labels).

<p align="center"><img src="https://i.ibb.co/vqXWTxp/Screenshot-533.png" alt="Screenshot-533" border="0" width="60%">


**Object Localization** : Locate the presence of objects in an image and indicate their location with a bounding box.
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Input** : An image with one or more objects, such as a photograph.
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Output** : One or more bounding boxes (e.g. defined by a point, width, and height).

<p align="center"><img src="https://i.ibb.co/3sGY1kj/Screenshot-536.png" alt="Screenshot-536" border="0" width="60%">

The bouding boxes can pe represented in the form of xc , yc and w , h. <br/>
where xc and yc are center of the window and w and h will be width and height of window.

In the dataset we label the object with thier class and create a region around the object i.e Ground truth bounding box over the object i.e.

**Class name** :   <class_name> <br/>
**Bouding box** :  <left> <top> <width> <height>

|    Input Image      |     Class name    |        Bounding Box            |
| ----------------|  --------------- |  ------------------------    |
|   bottle.jpg    |       bottle     |       6 234 45 362           |
|   person.jpg    |       person     |       1 156 103 336          |
|   car.jpg    |       car     |       36 111 198 416           |


we train the model for both classification and localization with different loss function. 

For **classification** : we use categorical cross entropy loss / sparse categorical entropy loss

For **localization** : we use Mean square error loss or L2 loss

<p align="center"><img src="https://miro.medium.com/max/3816/1*NTVoRZYBWbwRxNidyLCxPw.png" width="60%"/>

Here we can see that our ground truth bounding box is (200 , 250 ) and (600 , 400) , we are representing bounding boxes with (x1,y1,x2,y2) but we can also use (xc,yc,w,h).

Now based on the value in the loss function , we update the weights i.e filter , updation will be perform during backpropogation. 

We keep training on the set of images , now are filter learns for wh


<p align="center"><img src="https://i.ibb.co/kymqjPw/Screenshot-534.png" alt="Screenshot-534" border="0" width="60%"/>


---

### **`Question`** : 

Now we localize the single object present in the image but now how should we detect the bouding box for the 2 or more object present in the same image with different scale i.e. image shown below ?

<p align="center"><img src="https://i.ibb.co/qrScJbF/Screenshot-546.png" alt="Screenshot-546" border="0" width="50%">


### **`Answer`** : 

**Single Object**

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/intro_resize_1.jpg" width="50%"/>

If there is only one object in an image, there is no problem. We can just resize the image to the required dimension and use the same localization network.

**Two Objects** 

How do we handle the case of 2 objects?

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/intro_resize_2.jpg" width="50%"/>

* One simple hack would be to just split the image into 2, so that the 2 objects are separated. Now we can resize the images and reuse the same localization network, locate the objects in the 2 separate images. Later, we can just translate the co-ordinates onto the original image.

But still this doesn't work because for object detection we don't know either how of many object present in the image and not the location.

**Problem - Any number of objects**

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/intro_resize_3.jpg" width="30%"/>

While the above approach would work, this would not be a generic solution. Since there can be any number of objects in the image and neither do I know the number of objects, nor their locations before hand.

So, how can we solve this problem?

**Solution - Sliding Window** <br/>

Since we neither know the number of objects nor their location, the only option left is to scan all possible locations in an image.

To do this, we can take crops at all possible locations in the image and feed it to the localization network.

Then, this network, will take a look at the patch and will decide if there is an object in that patch. If so, it will give us the bounding box (bbox).

This technique is called the ‘Sliding Window’.

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/intro_sliding_window.gif" alt="Screenshot-547" border="0" width="80%"> -->

So, this way, by reusing the same Localization network, I will be able to do object detection by doing some preprocessing on the image.

**What are the preprocessing steps?**

* Use Sliding Window and crop the image patches and
* Resize the image to a fixed size (Since the Localization network expects the input to be of a fixed size)

**Problem - Objects of Different Sizes**

But still there is a problem with this approach. We cant expect objects in the image to be of similar sizes. As you can see in the image below, the sliding window size (red box) is insufficient to cover the person or bycycle completely. So using a fixed size Sliding Window may not solve the problem.

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/intro_pyramid.jpg" width="50%"/>

So, what is the solution to this problem?

**Solution - Image Pyramid**

Either I have to use Sliding windows of different sizes or resize the main image itself, keeping the Sliding Window size constant. Usually the 2nd approach is taken.

Experimentally, it is found that scaling the image to six different sizes would be good enough to locate most of the objects. Smaller objects get detected in Larger scaled image and Larger objects get detected in Small Scaled images. This concept is called Image Pyramid.

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/intro_pyramid_1.jpg" width="50%"/>

Preprocessing at the input side
So, these are the additional preprocessing I have to do on the input side to do Object Detection. The Sliding Window will take care of the location of the object and Image Pyramid will take care of the Size of the object.

With these changes, I will be able to do Object Detection with the same Localization Network as the base. Only thing that I added were a bit of pre-processing steps.

<p align="center"><img src="https://cogneethi.com/assets/images/evodn/intro_detection_1.jpg" width="50%"/>


**Object Detection** : Locate the presence of objects with a bounding box and types or classes of the located objects in an image.
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; **Input** : An image with one or more objects, such as a photograph.
  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  **Output** : One or more bounding boxes (e.g. defined by a point, width, and height), and a class label for each bounding box.

Predicting the location of the object along with the class is called **object Detection** . In place of predicting the class of object from an image, we now have to predict the class as well as a rectangle(called bounding box) containing that object. It takes 4 variables to uniquely identify a rectangle. So, for each instance of the object in the image, we shall predict following variables:

* class_name, 

* Window_center_x_coordinate,

* Window_center_y_coordinate,

* bounding_box_width,

* bounding_box_height

Just like multi-label image classification problems, we can have multi-class object detection problem where we detect multiple kinds of objects in a single image