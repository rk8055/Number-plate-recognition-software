#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy as np
import cv2
import csv
import pandas as pd
from glob import glob
import xml.etree.ElementTree as xet

from shutil import copy


# In[58]:


import os


# In[59]:


rf = pd.read_csv('labels.csv')
rf.head()


# In[60]:


# Labeling in yolo
#    height, width, center_x, center_y


# In[61]:


def chkng(path):

    chkng = xet.parse(path).getroot()
    name = chkng.find('filename').text
    filename = f'./images/{name}'
    chkng_size = chkng.find('size')
    #width
    width = int(chkng_size.find('width').text)
    height = int(chkng_size.find('height').text)
    return filename, width, height
    
rf[['filename','width','height']] = rf['filepath'].apply(chkng).apply(pd.Series)


# In[62]:


rf.head()


# In[63]:


rf['center_x'] = (rf['xmax'] + rf['xmin'])/ (2* rf['width'])
rf['center_y'] = (rf['ymax'] + rf['ymin'])/ (2* rf['height'])
rf['bb_width'] = (rf['xmax'] - rf['xmin'])/rf['width']
rf['bb_height'] = (rf['ymax'] - rf['ymin'])/rf['height']


# In[64]:


rf.head()


# In[65]:


#splitting data for training and testing

rf_train = rf.iloc[:200]
rf_test = rf.iloc[200:]


# In[66]:


#text ID must consist of class_id, centery, centerx, bb_width, bb_height


# In[67]:


fileNameTrain = './data_images/train'
values = rf_train[['filename','center_x','center_y','bb_width','bb_height']].values
for fname, x, y, w, h in values:
    nameImage = os.path.split(fname)[-1]
    textName = os.path.splitext(nameImage)[0]
    pathImage = os.path.join(fileNameTrain, nameImage)
    fileLabel = os.path.join(fileNameTrain, textName + '.txt')
# copying the images into this folder
    copy(fname, pathImage)
#generating a text file with its labels in it.
    labelling = f'0 {x} {y} {w} {h}'
    with open(fileLabel, mode = 'w') as f:
        f.write(labelling)
        f.close()


# In[68]:


fileNameTest = './data_images/test'
values = rf_test[['filename','center_x','center_y','bb_width','bb_height']].values
for fname, x, y, w, h in values:
    nameImage = os.path.split(fname)[-1]
    textName = os.path.splitext(nameImage)[0]
    pathImage = os.path.join(fileNameTest, nameImage)
    fileLabel = os.path.join(fileNameTest, textName + '.txt')
# copying the images into this folder
    copy(fname, pathImage)
#generating a text file with its labels in it.
    labelling = f'0 {x} {y} {w} {h}'
    with open(fileLabel, mode = 'w') as f:
        f.write(labelling)
        f.close()


# In[ ]:




