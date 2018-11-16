import numpy as np
import cv2 as cv
import sys

# images is a dict of numpy arrays, containing images
def keyPoints(images):
    # for every image find keypoint discriptors
    sift = cv.xfeatures2d.SIFT_create()
    imageKeyPoints = {}
    imageDescriptors = {}
    for i in range(len(images)):
        img = images[i]
        # finding dicriptors
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        keyPoints, descriptors = sift.detectAndCompute(img, None)
        imageDescriptors[i] = descriptors
        imageKeyPoints[i] = keyPoints
    # compare each image with every other
    return (imageKeyPoints, imageDescriptors)

def keyPointMatching(images, imageKeyPoints, imageDescriptors, imgA, imgB, lowsR):
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(imageDescriptors[imgA],
                             imageDescriptors[imgB], k=2)
                             # matches 2 nearest neigbours
    #using lows ratio test
    good = [[],[]]
    for i, (m, n) in enumerate(matches):
        if m.distance < lowsR * n.distance: # if closest match is ratio 
                                          # closer than the second closest one,
                                          # then the match is good
            good[0].append(imageKeyPoints[imgA][m.queryIdx].pt)
            good[1].append(imageKeyPoints[imgB][m.trainIdx].pt)
    return good

'''
imgwithlines - image on which we draw the epilines
for the points in img2 lines - corresponding epilines
https://docs.opencv.org/trunk/da/de9/tutorial_py_epipolar_geometry.html
'''
def drawlines(imgwithlines,img2,lines,pts1,pts2):
    r,c, chl = imgwithlines.shape
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        imgwithlines = cv.line(imgwithlines, (x0,y0), (x1,y1), color,1)
        imgwithlines = cv.circle(imgwithlines,tuple(pt1),5,color,-1)
        img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return imgwithlines,img2

def drawlinesP(imgwithlines,lines):
    r,c, chl = imgwithlines.shape
    for r in lines.values():
        # print (r)
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        imgwithlines = cv.line(imgwithlines, (x0,y0), (x1,y1), color,1)
    return imgwithlines

def intersection_point(line1, line2):
    a = np.zeros((2,2))
    a[0:1,:] = np.array([[line1[0], line1[1]]])
    a[1:2,:] = np.array([[line2[0], line2[1]]])
    b = np.zeros([2,1])
    b[0,0] = np.array(-line1[2])
    b[1,0] = np.array(-line2[2])
    try:
        inter = np.linalg.solve(a,b)
    except np.linalg.linalg.LinAlgError:
        inter = None
    return inter

def ptLocs(inter2dLines, inter2dPts, lines, tol=1e-2): 
    ptVectors = np.zeros((6,6)) # will store data for a pt in rows
    vals = []
    for j in range(len(inter2dLines)):
        key = inter2dLines[j]
        ipt = inter2dPts[j]
        for i in range(len(lines)):
            line = lines[i]
            d = line[0]*ipt[0] + line[1]*ipt[1] + line[2]
            d = d / (line[0]**2 + line[1]**2)**0.5
            if abs(d) < tol:
                d = 0
            ptVectors[j, i] = d
    return ptVectors

def rem(inter2dLines, inter2dPts):
    vals_to_del = []
    for i in range(len(inter2dLines)):
        interLine = inter2dLines[i]
        if interLine[0] < 3 and interLine[1] >= 3:
            vals_to_del.append(i)
    inter2dLinesCopy = []
    inter2dPtsCopy = []
    for i in range(len(inter2dLines)):
        if i not in vals_to_del:
            inter2dLinesCopy.append(inter2dLines[i])
            inter2dPtsCopy.append(inter2dPts[i])
    return inter2dLinesCopy, inter2dPtsCopy

'''
brief: outputs an label_img, that marks the temporal epipolar regions
 acoording to the label given below. Please refer the paper to know what
 LHS corresponds
labels:
 R1(i,j,k) - 0
 R2(i,j) - 1
 R2(i,k) - 2
 R2(j,k) - 3
 R3(i,j',k') - 4
 R3(i',j',k) - 5
 R3(i',j,k') - 6
 R4(i',j') - 7
 R4(j',k') - 9
 R4(i',k') - 8
 R5(i,i',k') - 10
 R5(i,i',j') - 11
 R5(j,j',i') - 12
 R5(j,j',k') - 13
 R5(k,k',i') - 14
 R5(k,k',j') - 15

@param img the original image
@param pts the 6 distict intersection points
@param ptVectors the 6x6 array of 6x1 vectors that tell the distance of
 the intersection points from the epipolar lines.
@param lines the 3 epipolar lines and 3 parallel lines to them
@param tol tolerance for making distance = 0 if abs(d) < tol
'''
def label(img, pts, ptVectors, lines, tol=1):
    h,w, chl = img.shape
    out = np.zeros((h,w))
    cols, rows, chl = img.shape
    vector = np.zeros(6)
    for i in range(cols):
        for j in range(rows):
            la = ptVec((j, i), lines, vector, tol)
            prodcts = ptVectors*la[None, :]
            ##### labelling of the regions | happens here.
            if np.all(prodcts[0:3,0:3] >= 0):
                out[i, j] = 0
            elif np.all(prodcts[0:3,0:2] <= 0) and np.all(prodcts[0:3,2] >= 0):
                out[i, j] = 1
            elif np.all(prodcts[0:3,list([0,2])] <= 0) and np.all(prodcts[0:3,1] >= 0):
                out[i, j] = 2
            elif np.all(prodcts[0:3][1:3] <= 0) and np.all(prodcts[0:3,0] >= 0):
                out[i, j] = 3
            elif prodcts[5,0] >= 0 and prodcts[0,4] >= 0 and prodcts[1,5] >= 0:
                out[i, j] = 4
            elif prodcts[3,2] >= 0 and prodcts[1,3] >= 0 and prodcts[2,4] >= 0:
                out[i, j] = 5
            elif prodcts[4,1] >= 0 and prodcts[0,3] >= 0 and prodcts[2,5] >= 0:
                out[i, j] = 6
            elif prodcts[0,3] <= 0 and prodcts[0,4] <= 0:
                out[i, j] = 7
            elif prodcts[1,3] <= 0 and prodcts[1,5] <= 0:
                out[i, j] = 8
            elif prodcts[2,4] <= 0 and prodcts[2,5] <= 0:
                out[i, j] = 9
            elif prodcts[5,0] <= 0 and prodcts[5,3] >= 0 and prodcts[2,4] <= 0:
                out[i, j] = 10
            elif prodcts[5,0] <= 0 and prodcts[5,3] >= 0 and prodcts[2,5] <= 0:
                out[i, j] = 11
            elif prodcts[4,1] <= 0 and prodcts[4,4] >= 0 and prodcts[1,3] <= 0:
                out[i, j] = 12
            elif prodcts[4,1] <= 0 and prodcts[4,4] >= 0 and prodcts[1,5] <= 0:
                out[i, j] = 13
            elif prodcts[3,2] <= 0 and prodcts[3,5] >= 0 and prodcts[0,3] <= 0:
                out[i, j] = 14
            elif prodcts[3,2] <= 0 and prodcts[3,5] >= 0 and prodcts[0,4] <= 0:
                out[i, j] = 15
    return out

# find the distance vector
def ptVec(ipt, lines, vector, tol):
    for i in range(len(lines)):
        line = lines[i]
        d = line[0]*ipt[0] + line[1]*ipt[1] + line[2]
        d = d / (line[0]**2 + line[1]**2)**0.5
        if abs(d) < tol:
                d = 0
        vector[i] = d
    return vector

# given a temporal order, get the corrent regions
def get_valid_regions(label_img, img, temporal_order):
    valid_TERS = valid_labels(temporal_order)
    out = np.copy(img)
    for i in range(out.shape[0]): # iterate through cols
        for j in range(out.shape[1]): # iterate through rows
            label = label_img[i,j] # get the label of the current pixel
            if label not in valid_TERS:
                out[i,j] = [0]*img.shape[2]
    return out

# the lookup table that the paper talks about
def valid_labels(temporal_order):
    valid_TERS = None
    if temporal_order == 1:
        valid_TERS = [1,2,4,8,9,10,13,15]
    elif temporal_order == 2:
        valid_TERS = [0,1,5,6,7,10,11,12]
    elif temporal_order == 3:
        valid_TERS = [0,1,4,6,7,13,14,15]
    elif temporal_order == 4:
        valid_TERS = [2,3,5,7,9,11,12,14]
    return valid_TERS