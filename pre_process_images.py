# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 12:20:17 2022

@author: Chris
"""
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import pickle

#import tensorflow as tf



N_IMAGES = 202599
Data_SIZE = 50
IMG_SIZE = 256
IMG_PATH = 'images_%i_%i.pb' % (IMG_SIZE, IMG_SIZE)
ATTR_PATH = 'attributes.pb'

#start = 1500
#end = 2000
    
def preprocess_images(start, end):
    
    print("Start")
    
    for n in range(start,end):
        raw_images = []
        for i in range ( 1+(n*50) , Data_SIZE+1+(n*50) ):
            raw_images.append(mpimg.imread('img_align_celeba/%06i.jpg' % i)[20:-20])
        
        all_images = []
        for i, image in enumerate(raw_images):
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
            all_images.append(image)
    
        data = np.concatenate([img.transpose((2, 0, 1))[None] for img in all_images], 0)
        
        # Save data in .npy file
        np.save(f"data/{n}",data)
        
    
    print("Done image save from :",start," ->",end," !")

    return 1


def preprocess_attributes():
    attr_lines = [line.rstrip() for line in open('list_attr_celeba.txt', 'r')]

    attr_keys = attr_lines[1].split()
    attributes = {k: np.zeros(N_IMAGES, dtype=np.bool) for k in attr_keys}

    for i, line in enumerate(attr_lines[2:]):
        image_id = i + 1
        split = line.split()
        assert len(split) == 41
        assert split[0] == ('%06i.jpg' % image_id)
        assert all(x in ['-1', '1'] for x in split[1:])
        for j, value in enumerate(split[1:]):
            attributes[attr_keys[j]][i] = value == '1'

    print("Saving attributes to %s ..." % ATTR_PATH)
    return attributes


def save_attributes(name):
    att = preprocess_attributes()
    # create a binary pickle file 
    f = open(f"{name}.pkl","wb")
    
    # write the python object (dict) to pickle file
    pickle.dump(att,f)
    
    # close file
    f.close()


