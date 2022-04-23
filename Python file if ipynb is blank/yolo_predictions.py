#!/usr/bin/env python
# coding: utf-8

# In[25]:


import cv2
import os
import pytesseract as pt
import numpy as np


# In[26]:


INPUT_WIDTH = 640
INPUT_HEIGHT = 640


# In[27]:


# loading my images

#img = cv2.imread('./test_images/getty_sample.jpg')
#cv2.namedWindow('test image', cv2.WINDOW_KEEPRATIO)
#cv2.imshow('test image',img)
#cv2.waitKey()
#cv2.destroyAllWindows()


# In[28]:


# loading the yolo models

fileNet = cv2.dnn.readNetFromONNX('./Model/weights/best.onnx')
fileNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
fileNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


# In[29]:


#converting processed images into yolo format
def get_dtdcns(img, fileNet):

    image = img.copy()
    row, col, d = image.shape
    rcMaximum = max(row,col)
    prcImage = np.zeros((rcMaximum, rcMaximum, 3),dtype = np.uint8)
    #getting trained brain from yolo
    prcImage[0:row, 0:col] = image
    blob = cv2.dnn.blobFromImage(prcImage, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), swapRB = True, crop = False)
    fileNet.setInput(blob)
    preds = fileNet.forward()
    dtdcns = preds[0]
    return prcImage, dtdcns


# filter images beased on probability and confi points

def non_maximum_supression(prcImage, dtdcns):

    boxes = []
    confis = []
    wImg, hImg = prcImage.shape[:2]
    xValue = wImg/INPUT_WIDTH
    yValue = hImg/INPUT_HEIGHT
    
    for i in range(len(dtdcns)):
        row = dtdcns[i]
        confi = row[4] # confi score
        if confi > 0.4:
            class_score = row[5] # probability score
            if class_score > 0.25:
                cx, cy, w, h = row[0:4]
                lft1 = int((cx - 0.5*w)* xValue)
                tp1 = int((cy - 0.5*h)* yValue)
                wdt1 = int(w*xValue)
                hgt1 = int(h*yValue)
                box = np.array([lft1, tp1, wdt1, hgt1])
                confis.append(confi)
                boxes.append(box)
    #cleaning boxes
    NPbox = np.array(boxes).tolist()
    NPconfi = np.array(confis).tolist()
    # using non maximum supression
    index = cv2.dnn.NMSBoxes(NPbox, NPconfi, 0.25, 0.45)
    return NPbox, NPconfi, index


#Drow the boxes
def drawings(image, NPbox, NPconfi, index):
    
    for ind in index:
        x, y, w, h = NPbox[ind]
        BBconf = NPconfi[ind]
        conf_text = 'plate: {:.0f}%'.format(BBconf* 100)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.rectangle(image, (x, y-30), (x+w, y), (255, 0, 255), -1)
        cv2.putText(image, conf_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return image


# In[30]:


def yolo_predictions(img, fileNet):
    
    #getting my dtdcns
    prcImage, dtdcns = get_dtdcns(img, fileNet)
    
    #applying non maximal supression
    NPbox, NPconfi, index = non_maximum_supression(prcImage, dtdcns)
    
    #drawing the imgs
    resultImg = drawings(img, NPbox, NPconfi, index)
    
    return resultImg


# In[31]:


#testing process
#make sure your directory is the same
img = cv2.imread('./test_images/N22.jpg')

photoResults = yolo_predictions(img, fileNet)

cv2.namedWindow('photoResults', cv2.WINDOW_KEEPRATIO)
cv2.imshow('photoResults', photoResults)
cv2.waitKey()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:


# processing real time image object detection
#make sure your directory is the same
cap = cv2.VideoCapture('./test_images/traffic.mp4')

while True:
    x, frame = cap.read()
    if x == False:
        print('unable to read video')
        break
        
    videoResults = yolo_predictions(frame, fileNet)
    
    cv2.namedWindow('videoResult', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('videoResult', videoResults)
    if cv2.waitKey(1) == 27:
        break
        
cv2.destroyAllWindows()
cap.release()


# In[ ]:




