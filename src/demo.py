'''
Author: Apoorv Agnihotri

Code samples from:
https://kushalvyas.github.io/stitching.html
'''
import cv2 as cv
import numpy as np
import sys
from src import *
import matplotlib.pyplot as plt
##########################################################
#Reading files
##########################################################
paths = ['I1'] # valid options here, I1, I2, I3
ext = 'jpg'
for pathI in paths:
    if pathI in ['I2', 'I3']:
        ext = 'png'
    path = '../data/normal/' + pathI + '/'
    imagesNames = ['a.'+ext, 'b.'+ext, 'c.'+ext, 'd.'+ext]#, 'e.jpg', 'f.jpg']
    scale = (0.2, 0.2)
    if pathI in ['I2', 'I3']:
        scale = (0.4, 0.4)
    images = {} # will have 3 channel color imgs
    imageNos = len(imagesNames)
    m = 3
    k = 4

    ##########################################################
    #Rescaling
    ##########################################################
    images = {}
    for i in range(len(imagesNames)):
        img = imagesNames[i]
        print(path + img)
        temp = cv.imread(path + img)
        temp = cv.resize(temp, None, fx=scale[0], fy=scale[1], interpolation=cv.INTER_CUBIC)
        # plt.imshow(temp)
        # plt.show()
        images[i]=temp
    del temp

    ##########################################################
    #Finding KeyPoints and Discriptors
    ##########################################################
    print('finding keypoints and discriptors')
    imageKeyPoints, imageDescriptors = keyPoints(images)
    print('done keypoints and discriptors')
    # retured dictionaries with keys as imageNames

    ##########################################################
    #Finding matchings for best 'm' matching images for each image
    ##########################################################
    print('finding keymatches')
    lowsR = 0.85 # low's ratio
    goodMatchings={}
    for i in range(imageNos-1):
        imgA = i
        imgB = 3
        goodMatchings[(imgA,imgB)]= keyPointMatching(images, 
                                  imageKeyPoints, imageDescriptors, 
                                  imgA, imgB, lowsR)
    print('done keymatches')

    ##########################################################
    #Finding H for each of the pairs of images
    ##########################################################
    n = 1000 # iterations
    r = 8 # no of point to calc fundamental matrix
    t = 2 # pixel threashold
    Tratio = 0.95 # majority threashold
    Hs = []
    Ss = []
    print('finding Fundamentals')
    for i in range(imageNos -1):
        imgA = i
        imgB = 3 # we want fundamental matrices w.r.t. 4th image only
        list_kp = goodMatchings[(imgA, imgB)]
        H, S = cv.findFundamentalMat(np.array(list_kp[1]), np.array(list_kp[0]))
        Hs.append(H)
        Ss.append(S)
    print('done Fundamentals')
    print('Fundamental Matrices:', Hs)

    ##########################################################
    # Mannually selecting the interest points on I1, I2, I3, I4
    ##########################################################
    Points = {}
    if pathI == 'I2':
        Points[0] = np.array([[410, 182]])
        Points[1] = np.array([[420, 197]])
        Points[2] = np.array([[383, 193]])
        Points[3] = np.array([[416, 216]]) # for testing
    elif pathI == 'I1':
        Points[0] = np.array([[207, 242]])
        Points[1] = np.array([[243, 272]])
        Points[2] = np.array([[434, 225]])
        Points[3] = np.array([[500, 116]]) # for testing
    elif pathI == 'I3':
        Points[0] = np.array([[366, 85]])
        Points[1] = np.array([[381, 84]])
        Points[2] = np.array([[389, 83]])
        Points[3] = np.array([[479, 74]]) # for testing

    ##########################################################
    # Find the epipolar lines on I4, and draw them (assuming temporal order is 4)
    print('finding epipolar lines')
    ##########################################################
    lines = [] # lines[0,1,2]
    for i in range(3):
        lines.append(cv.computeCorrespondEpilines(Points[i].reshape(-1,1,2),
                                                1, Hs[i]))
        lines[i] = lines[i].reshape(-1,3)
        temp, dsf = drawlines(images[3], images[i],
                              lines[i], Points[3], Points[i])
    print('epipolar lines calulated')

    ##########################################################
    # Find the intersections of epipolar lines
    ##########################################################
    # intersiction[-1,0,1] = lines[2,0], lines[0,1], lines[1,2]
    intersection = {}
    inter2d = {}
    for i in range(-1, 2):
        inter = intersection_point(lines[i][0], lines[i+1][0])
        intersection[i] = inter
        j = i + 1
        if i == -1:
            i = 2
        if i < j:
            inter2d[(i, j)] = inter
        else:
            inter2d[(j, i)] = inter

    ##########################################################
    # find the equations of the parrellel lines
    ##########################################################
    # line parallel to lines[0] would be in newlines[0]
    print('finding parallel epipolar lines')
    newlines = {} # list of lines
    for i in range(-1,2):
        inx = -1*((2*i)+1) # custom index
        if inx == -3:
            inx = 0
        xy = intersection[i] # getting the correct intersection point
        c =  -(lines[inx][0][1]*xy[1,0] + lines[inx][0][0]*xy[0,0])
        newlines[inx] = [lines[inx][0][0]]
        newlines[inx].append(lines[inx][0][1])
        newlines[inx].append(c)
    # printing the newlines on the image
    temp = drawlinesP(temp, newlines)
    plt.imshow(temp)
    plt.show()
    for i in range(0,2):
        line = newlines[i]
        lines.append([line])

    lines.append([newlines[-1]])
    lines = np.squeeze(np.array(lines), axis=1)
    print('parallel epipolar lines calulated')


    ##########################################################
    # Find the intersection point of all the lines.
    ##########################################################
    print('finding intersection points')
    inter2dLines = []
    inter2dPts = []
    for i in range(6):
        for j in range(i):
            inter = intersection_point(lines[i], lines[j])
            if type(inter).__module__ == np.__name__:
                if i < j:
                    inter2dLines.append((i,j))
                    inter2dPts.append(inter)
                else:
                    inter2dLines.append((j, i))
                    inter2dPts.append(inter)

    # remove the duplicate intersection points
    inter2dLines, inter2dPts = rem(inter2dLines, inter2dPts)
    print('intersection points calulated')

    # for each point in intersection points, find the 6x6 arry denoting dist frm lines.
    ptVectors = ptLocs(inter2dLines, inter2dPts, lines, tol=1e-2)

    # function that returns the label of a point.
    print('finding labellings')
    label_img = label(images[3], inter2dPts, ptVectors, lines, tol=1)
    print('done labellings')
    # get the valid regions according to the temporal order of the images
    print('printing valid regions')
    out = get_valid_regions(label_img, images[3], temporal_order = 4)
    plt.imshow(out)
    plt.show()
    cv.imwrite('../result/'+pathI+'out.jpg', out)
    print('check', '../result/'+pathI+'out.jpg')

sys.exit()
