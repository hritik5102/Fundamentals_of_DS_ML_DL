print("[INFO] Importing Package")

from skimage.feature import hog
from skimage.transform import pyramid_gaussian
import joblib
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import numpy as np
import cv2
import os
import glob
from warnings import filterwarnings
filterwarnings('ignore')



#Define HOG Parameters
# change them if necessary to orientations = 8, pixels per cell = (16,16), cells per block to (1,1) for weaker HOG

print("[INFO] Defining HOG Parameters")
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3

# define the sliding window:
def sliding_window(image, stepSize, windowSize):# image is the input, step size is the no.of pixels needed to skip and windowSize is the size of the actual window
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):# this line and the line below actually defines the sliding part and loops over the x and y coordinates
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])

print("[INFO] Load svm classification model ")
model = joblib.load('./Model/Vehicle_Classification_model.npy')

print("[INFO] Test the trained classifier on the input image")
scale = 0
detections = []
# read the image you want to detect the object in:
folder = './testing_image'
img= cv2.imread(folder + '/' + 'test-8.jpg')

h,w,c = img.shape
print("({} , {})".format(w, h))
cv2.imshow('img',img)


# Try it with image resized if the image is too big
img= cv2.resize(img,(300,300)) # can change the size to default by commenting this code out our put in a random number
h,w,c = img.shape
print("({} , {})".format(w, h))
cv2.imshow('resize_img',img)

# defining the size of the sliding window (has to be, same as the size of the image in the training data)
(winW, winH)= (64,64)
windowSize=(winW,winH)
downscale=1.5
# Apply sliding window:
for resized in pyramid_gaussian(img, downscale=1.5): # loop over each layer of the image that you take!
    # loop over the sliding window for each layer of the pyramid
    for (x,y,window) in sliding_window(resized, stepSize=10, windowSize=(winW,winH)):
        # if the window does not meet our desired window size, ignore it!
        if window.shape[0] != winH or window.shape[1] !=winW: # ensure the sliding window has met the minimum size requirement
            continue

        fds = hog(window, orientations, pixels_per_cell, cells_per_block, block_norm='L2-Hys',transform_sqrt=True)  # extract HOG features from the window captured
        fds = fds.reshape(1, -1) # re shape the image to make a silouhette of hog
        pred = model.predict(fds) # use the SVM model to make a prediction on the HOG features extracted from the window
        
        if pred == 1:
            if model.decision_function(fds) > 0.6:  # set a threshold value for the SVM prediction i.e. only firm the predictions above probability of 0.6
                print("Detection:: Location -> ({}, {})".format(x, y))
                print("Scale ->  {} | Confidence Score {} \n".format(scale,model.decision_function(fds)))
                detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fds),
                                   int(windowSize[0]*(downscale**scale)), # create a list of all the predictions found
                                      int(windowSize[1]*(downscale**scale))))
    scale+=1
    
clone = img.copy()
for (x_tl, y_tl, _, w, h) in detections:
    cv2.rectangle(clone, (x_tl, y_tl), (x_tl + w, y_tl + h), (0, 0, 255), thickness = 2)
cv2.imshow("Raw Detections before NMS", clone)
rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections]) # do nms on the detected bounding boxes
sc = [score[0] for (x, y, score, w, h) in detections]
sc = np.array(sc)
print("detection confidence score: ", sc)
pick = non_max_suppression(rects, probs = sc, overlapThresh = 0.4)

# the peice of code above creates a raw bounding box prior to using NMS
# the code below creates a bounding box after using nms on the detections
# you can choose which one you want to visualise, as you deem fit... simply use the following function:
# cv2.imshow in this right place (since python is procedural it will go through the code line by line).
        
for (xA, yA, xB, yB) in pick:
    cv2.rectangle(img, (xA, yA), (xB, yB), (0,255,0), 2)
cv2.imshow("Raw Detections after NMS", img)
#### Save the images below
k = cv2.waitKey(0) & 0xFF 
if k == 27:             #wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'):
    cv2.imwrite('Output-without-NMS-2.png',clone)
    cv2.imwrite('Output-with-NMS-2.png',img)
    cv2.destroyAllWindows()
