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
built_in = True # to switch between built-in or custom
paths = ['I1']
for pathI in paths:
    path = '../data/normal/' + pathI + '/'
    imagesNames = ['a.jpg', 'b.jpg', 'c.jpg', 'd.jpg']#, 'e.jpg', 'f.jpg']
    scale = (0.2, 0.2)
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
        # finding the fundamental matrices, using ransac
        if (not built_in):
            H, S = findFundaMatrixRanSac(n, r, list_kp, t, Tratio) # not working now
        else:
            H, S = cv.findFundamentalMat(np.array(list_kp[1]), np.array(list_kp[0]))
        Hs.append(H)
        Ss.append(S)
    print('done Fundamentals')
    print('Fundamental Matrices:', Hs)

    ##########################################################
    # Mannually selecting the interest points on I1, I2, I3
    ##########################################################
    # Points = {} #I4
    # Points[0] = np.array([[340, 294]])
    # Points[1] = np.array([[411, 300]])
    # Points[2] = np.array([[403, 284]])
    # Points[3] = np.array([[633, 262]]) # for testing
    Points = {} #I1
    Points[0] = np.array([[208, 243]])
    Points[1] = np.array([[245, 272]])
    Points[2] = np.array([[434, 223]])
    Points[3] = np.array([[500, 119]]) # for testing


    ##########################################################
    # Find the epipolar lines on I4, and draw them
    ##########################################################
    lines = [] # lines[0,1,2]
    for i in range(3):
        lines.append(cv.computeCorrespondEpilines(Points[i].reshape(-1,1,2),
                                                1, Hs[i]))
        lines[i] = lines[i].reshape(-1,3)
        temp, dsf = drawlines(images[3], images[i],
                              lines[i], Points[3], Points[i])
    # plt.imshow(temp)
    # plt.show()


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
    # print ('intersection points:', intersection)
    # print ('inter2d points:', inter2d)

    ##########################################################
    # find the equations of the parrellel lines
    ##########################################################
    # line parallel to lines[0] would be in newlines[0]
    newlines = {} # list of lines
    # print (lines)
    for i in range(-1,2):
        inx = -1*((2*i)+1) # custom index
        if inx == -3:
            inx = 0
        xy = intersection[i] # getting the correct intersection point
        # print('xy',xy)
        c =  -(lines[inx][0][1]*xy[1,0] + lines[inx][0][0]*xy[0,0])
        newlines[inx] = [lines[inx][0][0]]
        newlines[inx].append(lines[inx][0][1])
        newlines[inx].append(c)
    # print ('oldlines:', lines)
    # print ('newlines:', newlines)
    # printing the newlines on the image
    temp = drawlinesP(temp, newlines)
    # plt.imshow(temp)
    # plt.show()
    for i in range(0,2):
        line = newlines[i]
        lines.append([line])

    lines.append([newlines[-1]])
    # print ('lines', np.array(lines).shape)
    lines = np.squeeze(np.array(lines), axis=1)
    # print ('lines', lines)


    ##########################################################
    # Find the intersection point of all the lines.
    ##########################################################
    inter2dLines = []
    inter2dPts = []
    for i in range(6):
        for j in range(i):
            inter = intersection_point(lines[i], lines[j])
            if type(inter).__module__ == np.__name__:
                if i < j:
                    # print ('i,j', i,j)
                    inter2dLines.append((i,j))
                    inter2dPts.append(inter)
                else:
                    inter2dLines.append((j, i))
                    inter2dPts.append(inter)
            # inter2d[(j,i)] = inter

    # print ('inter2dLines', inter2dLines)
    # print ('inter2dPts', inter2dPts)
    # sys.exit()

    # remove the intersection points that are same
            # print('line', i, line)
    # number of resulting points would be six
    inter2dLines, inter2dPts = rem(inter2dLines, inter2dPts)
    # print ('NEWinter2dLines', inter2dLines)
    # print ('NEWinter2dPts', inter2dPts)
    # sys.exit()

    # for each point in intersection points, find the 6 arry.
    ptVectors = ptLocs(inter2dLines, inter2dPts, lines, tol=1e-2)
    # print('pts', inter2dPts)
    # print ('ptVectors', ptVectors)

    # Now we have a binary 2d array of size 6 (as there are six points of interest
    # and 6 lines with wich we need to find the distance),
    # which would denote the distance value of each intersection point, we take in 
    # threashold which would help us allow some tardiness.

    # We now define a function that returns the label of a point, given the temporal order.
    # and the image. For now we only have one case of temporal order supported.
    # this function takes in the 2d 6x6 matrix that we generated and divides the points
    # according to the sign of the multiplications.

    label_img = label(images[3], inter2dPts, ptVectors, lines, tol=1)
    out = get_valid_regions(label_img, images[3], temporal_order = 4)
    plt.imshow(out)
    plt.show()
