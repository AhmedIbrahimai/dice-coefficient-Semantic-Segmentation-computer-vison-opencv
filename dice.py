import warnings
import os
warnings.filterwarnings('ignore')
import cv2
import numpy as np
from keras import backend as K
os.chdir(r'C:\Users\amb\Downloads')

def dice_coefficient(y_true, y_pred, smooth=1):

    # Flatten the segmentation masks
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()

    # Calculate the intersection between the segmentation masks
    intersection = np.sum(y_true_f * y_pred_f)

    # Calculate the sum of the segmentation masks
    y_true_sum = np.sum(y_true_f)
    y_pred_sum = np.sum(y_pred_f)

    # Calculate the Dice coefficient
    dice = (2. * intersection + smooth) / (y_true_sum + y_pred_sum + smooth)

    return dice

y_pred1 = cv2.imread('image_1.png')/255.
y_true1 = cv2.imread('image_2.png')/255.

print(y_pred1.shape)
print(y_true1.shape)

y_pred = y_pred1.reshape((-1, 853, 640, 3 ))
y_true = y_true1.reshape((-1, 853, 640, 3 ))

print(y_pred.shape)
print(y_true.shape)

dice_score = dice_coefficient(y_true1, y_pred1)
print ("Dice Coeff: {}".format(dice_score))
