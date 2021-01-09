# Import package
import cv2
import math
import numpy as np
import matplotlib as plt

# FilePath = 'F:/Opencv_Python/theroad.mp4'
FilePath = 'D:/Github-Projects/Machine_learning/Advanced-Lane-Lines/project_video.mp4'
cap = cv2.VideoCapture(FilePath)
w = cap.get(3)
h = cap.get(4)

def mouse_event(event, x, y, flags, para):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(event)
        pixel = hsv[y, x]

        cv2.setTrackbarPos('lowh', 'cor', pixel[0] - 10)
        cv2.setTrackbarPos('highh', 'cor', pixel[0] + 10)
        cv2.setTrackbarPos('lows', 'cor', pixel[1] - 20)
        cv2.setTrackbarPos('highs', 'cor', pixel[1] + 20)
        cv2.setTrackbarPos('lowv', 'cor', pixel[2] - 50)
        cv2.setTrackbarPos('highv', 'cor', pixel[2] + 50)


def callback(x):
    pass


cv2.namedWindow('cor')

cv2.createTrackbar('lowh', 'cor', 0, 180, callback)
cv2.createTrackbar('highh', 'cor', 0, 180, callback)
cv2.createTrackbar('lows', 'cor', 0, 255, callback)
cv2.createTrackbar('highs', 'cor', 0, 255, callback)
cv2.createTrackbar('lowv', 'cor', 0, 255, callback)
cv2.createTrackbar('highv', 'cor', 0, 255, callback)

cv2.createTrackbar('minline', 'cor', 0, 500, callback)
cv2.createTrackbar('maxgap', 'cor', 0, 500, callback)

cv2.createTrackbar('rad', 'cor', 0, 1800, callback)
cv2.createTrackbar('rad2', 'cor', 0, 1800, callback)
cv2.createTrackbar('width', 'cor', 0, 1800, callback)

cv2.createTrackbar('x1', 'cor', 456, 1800, callback)
cv2.createTrackbar('x2', 'cor', 1000, 1800, callback)
cv2.createTrackbar('y1', 'cor', 271, 1800, callback)
cv2.createTrackbar('y2', 'cor', 1011, 1800, callback)

cv2.createTrackbar('m', 'cor', 0, 4000, callback)
cv2.setTrackbarPos('m', 'cor', 2000)

cv2.resizeWindow('cor', 500, 500)


while True:
    _, frame = cap.read()

    # Set mouse callback
    cv2.setMouseCallback('imag', mouse_event)

    # Get region of interest
    x1 = cv2.getTrackbarPos('x1', 'cor')
    x2 = cv2.getTrackbarPos('x2', 'cor')
    y1 = cv2.getTrackbarPos('y1', 'cor')
    y2 = cv2.getTrackbarPos('y2', 'cor')

    # Crop the image
    img = frame[x1:x2, y1:y2]

    # Hide corner using ellipse
    rad = cv2.getTrackbarPos('rad', 'cor')
    rad2 = cv2.getTrackbarPos('rad2', 'cor')
    width = cv2.getTrackbarPos('width', 'cor')

    cv2.ellipse(img, (640, 640), (rad, rad2), 0, 0, 360, (0, 0, 0), width)

    # Apply Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Canny edge detection
    edges = cv2.Canny(gray, 100, 230, apertureSize=3)

    # thresolding
    _, thresh = cv2.threshold(edges, 100, 200, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    dilute = cv2.dilate(thresh, kernel, iterations=2)

    # cv2.imshow('Elipse', img)
    cv2.imshow('Canny edge', edges)
    cv2.imshow('threshold', thresh)
    cv2.imshow('Dilation', dilute)

    minline = cv2.getTrackbarPos('minline', 'cor')
    maxgap = cv2.getTrackbarPos('maxgap', 'cor')

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100, None)
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            m = cv2.getTrackbarPos('m', 'cor')
            pt1 = (int(x0 + m*(-b)), int(y0 + m*(a)))
            pt2 = (int(x0 - m*(-b)), int(y0 - m*(a)))
            cv2.line(img, pt1, pt2, (0, 255, 0), 3, cv2.LINE_AA)

    # lowh = cv2.getTrackbarPos('lowh', 'cor')
    # highh = cv2.getTrackbarPos('highh', 'cor')
    # lows = cv2.getTrackbarPos('lows', 'cor')
    # highs = cv2.getTrackbarPos('highs', 'cor')
    # lowv = cv2.getTrackbarPos('lowv', 'cor')
    # highv = cv2.getTrackbarPos('highv', 'cor')

    # lowred = np.array([lowh, lows, lowv])
    # highred = np.array([highh, highs, highv])
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # mask = cv2.inRange(hsv, lowred, highred)

    # kernel = np.ones((10, 10), np.uint8)
    # dilute = cv2.dilate(mask, kernel, iterations=1)
    # opening = cv2.morphologyEx(dilute, cv2.MORPH_CLOSE, kernel)

    # cv2.imshow('mask', mask)
    # cv2.imshow('opening', opening)

    # contours, hierarchy = cv2.findContours(
    #     black1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for i in contours:
    #     area = cv2.contourArea(i)
    #     if(area > 1000):
    #         x, y, w, h = cv2.boundingRect(i)
    #         rect = cv2.minAreaRect(i)
    #         box = cv2.boxPoints(rect)
    #         box = np.int0(box)
    #         cv2.drawContours(img, contours, -1, (255, 0, 0), 3)

    cv2.imshow('Output', img)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
