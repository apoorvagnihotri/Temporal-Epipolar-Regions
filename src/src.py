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

# temporal order of the images
def label(img, pts, ptVectors, lines, temp, tol=1):
    h,w, chl = img.shape
    out = np.zeros((h,w, chl))
    cols, rows, chl = img.shape
    vector = np.zeros(6)
    for i in range(cols):
        for j in range(rows):
            la = lebely((j, i), lines, vector, tol)
            prodcts = ptVectors*la[None, :]
            if temp == 4:
                ##### labelling of the regions happening here.
                if np.all(prodcts[0:3,0:3] >= 0):
                    out[i, j] = [0,0,0]
                elif np.all(prodcts[0:3,0:2] <= 0) and np.all(prodcts[0:3,2] >= 0):
                    out[i, j] = [1,1,1]
                elif np.all(prodcts[0:3,list([0,2])] <= 0) and np.all(prodcts[0:3,1] >= 0):
                    out[i, j] = [2,2,2]
                elif np.all(prodcts[0:3][1:3] <= 0) and np.all(prodcts[0:3,0] >= 0):
                    out[i, j] = [3,3,3]

    return out

def lebely(ipt, lines, vector, tol):
    for i in range(len(lines)):
        line = lines[i]
        d = line[0]*ipt[0] + line[1]*ipt[1] + line[2]
        d = d / (line[0]**2 + line[1]**2)**0.5
        if abs(d) < tol:
                d = 0
        vector[i] = d
    return vector