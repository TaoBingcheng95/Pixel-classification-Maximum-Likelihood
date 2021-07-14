# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 02:12:16 2020

@author: Lenovo
"""

import os
import sys
from pathlib import Path
import random
import math
import cv2
import numpy as np
from scipy.io import loadmat
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import cohen_kappa_score
#import matplotlib.pyplot as plt

random.seed(0)

ROW = 211
COL = 356

numClasses = 4
numDimensions = 6
numSamples = 30
allSamples = [30, 30, 30, 30]
classWeights = [1] * numClasses

colorkey = {
    1: [0, 0, 255],
    2: [0, 255, 0],
    3: [255, 0, 0],
    4: [128, 128, 128]
}  # Colour key


def load_data(data_path):
    """
    load all six images and samples mat.

    Parameters
    ----------
    data_path : pathlib.Path
        数据存储路径.

    Returns
    -------
    data ： numpy.array
        影像数据.
    groundTruth ： numpy.array
        地面真值数据.
    """

    matFile = loadmat(data_path / 'ground_truth.mat')
    gt = matFile['labelled_ground_truth']
    img_list = ['fe.bmp', 'le.bmp', 'r.bmp', 'g.bmp', 'b.bmp', 'nir.bmp']
    for idx, item in enumerate(img_list):
        if idx == 0:
            img = cv2.imread(str(data_path / item), 0)
        else:
            tmp = cv2.imread(str(data_path / item), 0)
            img = np.dstack((img, tmp))
    return img.astype(float), gt


def shuffle_multi_arr(arr_list):
    """

    Parameters
    ----------
    arr_list : list of numpy.array
        未打乱顺序的数组列表.

    Returns
    -------
    new_list : list of numpy.array
        打乱顺序的数组列表.

    """
    new_list = []
    state = np.random.get_state()
    for idx, item in enumerate(arr_list):
        np.random.set_state(state)
        np.random.shuffle(item)
        new_list.append(item)
    return new_list


def sampleImages(img,
                 gt,
                 numSamples=30,
                 numClasses=4,
                 numDimensions=6,
                 allSamples=[30, 30, 30, 30]):
    """
    SELECT RANDOM DISTROBUTION OF SAMPLE PIXELS FROM THE PROVIDED ground_truth

    Parameters
    ----------
    img : numpy.array
        image array.
    gt : numpy.array
        ground truth array.
    numSamples : int, optional
        number of samples per class - make sure numSamples is the same size as the max element of allSamples. The default is 30.
    numClasses : int, optional
        number of classes. The default is 4.
    numDimensions : int, optional
        number of dimensions per vector. The default is 6.
    allSamples : list, opential
        ability to change the number of samples per class. The default is [30, 30, 30, 30].
    
    Returns
    -------
    samples : numpy.array
        提取的样本数组.
    tgt : numpy.array
        提取的地表真值数组.

    """
    # holds tuples of positions for each class
    # samples = []

    for i in range(0, numClasses):
        yy, xx = np.where(gt == (i + 1))
        # numSamples = allSamples[i]
        rank = random.sample(range(0, len(xx)), numSamples)
        # samples.append(img[yy[rank], xx[rank], :])
        if i == 0:
            samples = img[yy[rank], xx[rank], :]
            tgt = gt[yy[rank], xx[rank]]
        else:
            tmp = img[yy[rank], xx[rank], :]
            samples = np.concatenate((samples, tmp), axis=0)
            tmp = gt[yy[rank], xx[rank]]
            tgt = np.concatenate((tgt, tmp), axis=0)
    samples, tgt = shuffle_multi_arr((samples, tgt))
    return samples, tgt


def calcGauss(samples, gt, numClasses, show=False):
    """
    use the previously gathered data to calculate mean vectors, 
    covariance matrices and a gaussian model for each class

    Parameters
    ----------
    samples : TYPE
        DESCRIPTION.
    show : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    means : list
        holds the vector of means of each class.4x6
    covs : list
        holds the covariance matrix of each class.4x[6x6]

    """

    meanVectors = [0] * numClasses
    covMatrices = [0] * numClasses
    gaussModels = [None] * numClasses

    # if len(samples) != numClasses:
    #     numClasses = len(samples)
    for i in range(numClasses):
        # Calculate the mean and covariance of each class samples
        data_part = samples[gt == i + 1, :]
        m = np.mean(data_part, axis=0)
        meanVectors[i] = m
        c = np.cov(data_part.astype(float), rowvar=False)
        covMatrices[i] = c
        gaussModels[i] = multivariate_normal(m, c)

    # if show:
    #     # print covariance matrix for each class
    #     print("COVARIANCE MATRICES")
    #     for i in range(0, 4):
    #         print("Class [{}]".format(i + 1))
    #         print(np.int_(covs[i]))
    #         print("\n")
    return meanVectors, covMatrices, gaussModels


def mlc_model(x, covs, means, classWeights=classWeights):
    """

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    covs : TYPE
        DESCRIPTION.
    means : TYPE
        DESCRIPTION.
    classWeights : TYPE, optional
        DESCRIPTION. The default is classWeights.

    Returns
    -------
    classMax : TYPE
        DESCRIPTION.
    secondBest : TYPE
        DESCRIPTION.

    """

    classMax = 0
    currentMax = 0
    secondBest = 0
    for classCount in range(0, numClasses):
        # a convertion from convertions lists
        cov = covs[classCount]
        # calculate the normal
        n = 1.0 / ((2 * math.pi**(3)) * (math.sqrt(np.linalg.det(cov))))
        # calculate the exponetional
        pixelVariance = np.transpose(
            x - means[classCount]
        )  # pixel variance - pixel (feature vector of the image) minus the vector of means
        # calculate the inverse of the covariance matrix to divide by variance
        a = np.dot(
            np.linalg.inv(cov), pixelVariance
        )  # in order to square the variance transpose matrix and preform the dot product
        # dot product with non-transposed feature vector to square the vectors values
        exponent = -0.5 * np.dot(
            a, (x - means[classCount])
        )  # calculate final exponent (multiply by -0.5 in accordance with normal distrobution)
        pdf = n * math.exp(
            exponent)  # the normal multipled by euler's number to the power of
        if (pdf > currentMax * classWeights[classCount]):
            secondBest = classMax  # second highest probability
            currentMax = pdf * classWeights[classCount]
            classMax = classCount + 1  # classMax holds the class integer with the highest pdf.
    return classMax, secondBest


def ClassificationAccuracyEvaluation(y_train, y_predict):
    """

    Parameters
    ----------
    y_train : TYPE
        DESCRIPTION.
    y_predict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    labelkey = "B | V | C | G"
    cm = confusion_matrix(y_train.flatten(), y_predict.flatten())
    print(f"CONFUSION MATRIX:\n{labelkey}\n{cm}")
    print('-' * 5)
    
    # correct = np.sum(cm.diagonal())
    # acc = correct / np.sum(cm)
    acc = accuracy_score(y_train, y_predict)
    print(f"Total classification accuracy : {acc:.3f}")
    print('-' * 5)
    
    target_names = ['buildings', 'vegetation', 'car', 'ground']
    print(
        "[1] - buildings (RED)\n[2] - vegetation (GREEN)\n[3] - car (BLUE)\n[4] - ground (GREY)"
    )
    print('-' * 5)
    
    cr = classification_report(y_train, y_predict, target_names=target_names, labels=np.unique(y_train))
    print(f"Classification report:\n{cr}")
    print('-' * 5)
    
    # calculate Cohen's Kappa score
    # (a, b) flat matrix of predicted and ground truth respectively
    kp = cohen_kappa_score(gt_train, gt_predict)
    print(f"Cohen's Kappa score: {kp:.3f}")
    print('-' * 5)

    # calculate user error for each class | TP/(TP+FP) | precision
    precisions = []
    for i in range(numClasses):
        total = np.sum(cm[:, i])
        correct = cm[i, i]
        error = correct / total
        precisions.append(error)
        print(f"User error for class [{i+1}] error: {error:.3f}")
    print('-' * 5)
    
    # calculate producer error for each class | TP/(TP+FN) | recall
    recalls = []
    for i in range(numClasses):
        total = np.sum(cm[i, :])
        correct = cm[i, i]
        error = correct / total
        recalls.append(error)
        print(f"Producer error for class [{i+1}] error: {error:.3f}")
    print('-' * 5)
    
    # calculate F1 Score for each class | 2/f1 = 1/precision + 1/recall | F1 score
    for i in range(numClasses):
        f1 = (2 * precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
        print(f"F1 for class [{i+1}] : {f1:.3f}")
    print('-' * 5)
    
    
    return None


def calculateConfutionImage(prediction, groundTruth):
    """
    in real world example you would not usually aquire so much ground truth values
    however, we can be use it to visualise the difference between the predicted values from n sample

    CLASSIFICATIONS:
    buildings (RED), vegetation (GREEN), car (BLUE), ground (GREY)
    MISS-CLASSIFICATIONS:
    building-vegetation (YELLOW), building-car (MAGENTA), building-ground (pink)
    vegetation-building (light yellow), vegetation-car (CYAN), vegetation-ground (light green)
    car-building (light magenta), car-vegetation (light cyan), car-ground (lavender)
    ground-building (light pink), ground-vegetation (very light green), ground-car (dark lavender)

    Parameters
    ----------
    prediction : TYPE
        DESCRIPTION.
    groundTruth : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    colorkeyConfusion = {
        '11': [0, 0, 255],
        '12': [0, 255, 255],
        '13': [0, 255, 0],
        '14': [255, 0, 0],
        '21': [0, 255, 255],
        '22': [0, 255, 0],
        '23': [255, 255, 0],
        '24': [128, 255, 128],
        '31': [128, 255, 255],
        '32': [255, 255, 128],
        '33': [255, 0, 0],
        '34': [255, 128, 128],
        '41': [180, 255, 255],
        '42': [70, 255, 70],
        '43': [255, 70, 70],
        '44': [128, 128, 128]
    }  # Colour key

    cm_img = 10 * prediction + groundTruth
    img_new = np.zeros_like(cm_img)
    color_map_cm = np.zeros((len(colorkeyConfusion), 3), dtype=np.uint8)
    for idx, key in enumerate(colorkeyConfusion):
        img_new[cm_img == int(key)] = idx
        color_map_cm[idx, :] = colorkeyConfusion[key]

    confusionImage = color_map_cm[img_new, :]
    # cv2.imshow("Confusion image", confusionImage)
    # cv2.imwrite('confusionImage.png', confusionImage)
    # cv2.waitKey(0)
    return None


def createDifferenceImages(predicted):
    """

    Parameters
    ----------
    predicted : TYPE
        DESCRIPTION.

    Returns
    -------
    allDifference : TYPE
        DESCRIPTION.

    """
    allDifference = np.zeros((ROW, COL, 0), dtype=np.uint8)
    for i in range(numClasses):
        difference = np.zeros_like(predicted)
        difference[predicted == i + 1] = 255
        allDifference = np.dstack((allDifference, difference))
    return allDifference


def MorphologicalOperations(im):
    """
    
    Parameters
    ----------
    im : TYPE
        DESCRIPTION.

    Returns
    -------
    tmp : TYPE
        DESCRIPTION.

    """

    kernel = np.ones((2, 2), np.uint8)

    kernelA = np.array([[1, -2, 1], [-2, 1, -2], [1, -2, 1]], np.uint8)

    kernel2 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], np.uint8)

    kernal2Compliment = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)

    tmp = cv2.erode(im, kernal2Compliment, iterations=3)
    tmp = cv2.dilate(tmp, kernal2Compliment, iterations=5)

    # # Buildings
    # imOP = im[:, :, 0]
    # imOP = cv2.erode(imOP, kernel, iterations=3)
    # imOP = cv2.dilate(imOP, kernel, iterations=5)
    # alteredImage[:, :, 0] = imOP

    # # Vegetation
    # imOP2 = im[:, :, 1]
    # imOP2 = cv2.erode(imOP2, kernel, iterations=1)
    # imOP2 = cv2.dilate(imOP2, kernel, iterations=2)
    # alteredImage[:, :, 1] = imOP2

    # # Cars
    # imOP3 = im[:, :, 2].astype(np.uint8)
    # imOP3 = cv2.erode(imOP3, kernel, iterations=1)
    # imOP3 = cv2.dilate(imOP3, kernel, iterations=2)
    # alteredImage[:, :, 2] = imOP3

    # # Ground
    # imOP4 = im[:, :, 3]
    # imOP4 = cv2.erode(imOP4, kernel, iterations=1)
    # imOP4 = cv2.dilate(imOP4, kernel, iterations=2)
    # alteredImage[:, :, 3] = imOP4

    return tmp


def calculateCorrectPercentage(im, im2):
    """
    
    Parameters
    ----------
    im : TYPE
        DESCRIPTION.
    im2 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    total = im.size
    tmp = im == im2
    correct = tmp.sum()
    error = correct / total
    print("After morphological operations")
    print(f"correct: {error:.3f}")
    return None


if __name__ == '__main__':

    data, ground_truth = load_data(Path('.\data'))

    samples, gt_train = sampleImages(data, ground_truth)

    means, covs, gm = calcGauss(samples, gt_train, numClasses, show=False)
    
    """
    CLASSIFICATION
    Calculate the class conditional pdf for every pixel in the image
    Select highest probability and place corrosponding class colour in the x, y positions of the classified output image
    holds classified pixels in integer form predict data of model
    """
    gt_predict = np.zeros_like(gt_train)  # maxProbValues
    data_num = gt_train.shape[0]
    for idx in range(data_num):
        x = samples[idx, :]
        # y = gt_train[idx]
        
        # method 1
        classMax, secBest = mlc_model(x, covs, means)
        gt_predict[idx] = classMax

        # # method 2
        # # reset probabilities
        # classProbabilities = [0] * numClasses
        # # for each class
        # for classNo in range(numClasses):
        #     # take natural log of the pdf
        #     classProbabilities[classNo] = gm[classNo].logpdf(x)
        # # set predicted class respectively
        # gt_predict_1[idx] = classProbabilities.index(
        #     max(classProbabilities)) + 1

    ClassificationAccuracyEvaluation(gt_train, gt_predict)

    # # holds classified pixels in integer form
    # maxProbValues = np.zeros((ROW, COL), dtype=np.uint8)
    # # secondProbValues = np.zeros((ROW, COL), dtype=np.uint8)
    # for x1 in range(0, COL):
    #     for y1 in range(0, ROW):
    #         x = data[y1, x1, :]  # 6x(1) feature vector
    #         classMax, secBest = mlc_model(x, covs, means)
    #         # class of max value - used for confusion matrix and visualising difference image
    #         maxProbValues[y1, x1] = classMax

    # calculateCorrectPercentage(maxProbValues, ground_truth)
    # calculateConfutionImage(maxProbValues, ground_truth)
    # classlayer = createDifferenceImages(maxProbValues)

    # alteredImage = MorphologicalOperations(maxProbValues)
    # calculateCorrectPercentage(alteredImage, ground_truth)

    # # final image - holds RGB representation of classified pixels
    # color_map = np.zeros((numClasses, 3), dtype=np.uint8)
    # for idx, key in enumerate(colorkey):
    #     # color_map[idx, 0] = key
    #     color_map[idx, :] = colorkey[key]

    # trueImg = color_map[ground_truth-1,:]
    # finalImg = color_map[maxProbValues-1,:]
    # finalImgModified = color_map[alteredImage-1,:]

    # cv2.imshow("Ground Truth", trueImg)
    # cv2.imwrite('true.png', trueImg)
    # cv2.imshow("Predictions", finalImg)
    # cv2.imwrite('prediction.png', finalImg)
    # cv2.imshow("Predicted + Morphological operations", finalImgModified)
    # cv2.imwrite('predictionModified.png', finalImgModified)
    # cv2.waitKey(0)
