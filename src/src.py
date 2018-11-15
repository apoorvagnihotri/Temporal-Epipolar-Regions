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
brief:
    find the fundamental matrix, such that
    list_kp[0], transforms to list_kp[1]
params:
    n is the number of times to repeat ransac
    r is the number of points to approximate H(min 4)
    t is the number of pixels tolerance allowed
    list_kp is [list1, list2] where list1 and list2 contain
        the matched keypoints, on the same index, list1 is [(x1,y1),..]
    Tratio is the ratio of the for which we terminate early.
'''
def findFundaMatrixRanSac(n, r, list_kp, t, Tratio):
    list_kp1 = list_kp[0]
    list_kp2 = list_kp[1]
    T = int(Tratio * len(list_kp2))

    Sis = []
    Sisno = []
    for i in range(n):
        list_kp1r = []
        list_kp2r = []
        
        # selecting ramdomly r points
        for i in range(r):
            key = np.random.choice(len(list_kp2))
            list_kp1r.append(list_kp1[key])
            list_kp2r.append(list_kp2[key])
        # print (list_kp1r, list_kp2r)

        # find the homo, inlier set
        P = make_P(list_kp1r, list_kp2r)
        # print(P)
        H, Si = findH_Si(P, list_kp, t)
        Sis.append(Si)
        # print ('Si:',Si)
        Sisno.append(len(Si[0]))

        # if majority return with new H
        if len(Si[0]) >= T:
            P = make_P(Si[0], Si[1])
            # print('threashold crossed')
            # print('P output as:', P)
            H, Si = findH_Si(P, list_kp, t)
            # print ('si',Si)
            return (H, Si)

    # print('Sisno',Sisno)
    Sisnoi = np.argmax(np.array(Sisno)) # taking the first index 
                                        # with global max cardinality
    # print('i', Sisnoi)
    # print('maxii', Sisno[Sisnoi])
    Si = Sis[Sisnoi]
    P = make_P(Si[0], Si[1])
    H, Si = findH_Si(P, list_kp, t)
    # print ('si',Si)
    return (H, Si)

def findH_Si(P, list_kp, t):
    # print(P.shape)
    # do svd on P get perlimns H
    u, s, vh = np.linalg.svd(P, full_matrices=True)
    H = vh[-1].reshape(3,3) # taking the last singular vector
    u, s, vh = np.linalg.svd(H, full_matrices=True)
    s[2] = 0 # assuring that Fundamental matrix calculated is rank 2
    # print (s)
    H = s*np.matmul(u, vh)
    H = H/np.linalg.norm(H)
    Si = [[],[]]

    # multiply all the matches and find if within tol
    initialPts = list_kp[0]
    finalPts = list_kp[1]
    # print('no of keypts', len(initialPts))
    for i in range(len(initialPts)):
        inPt = initialPts[i]
        fPt = finalPts[i]
        vi = np.array([[inPt[0]],[inPt[1]], [1]])
        vi2 = np.array([[fPt[0]],[fPt[1]], [1]]).T
        temp = np.matmul(H, vi)
        vf = np.matmul(vi2, temp)

        # check if within some tolerance
        if np.linalg.norm(vf) <= t:
            Si[0].append(inPt)
            Si[1].append(fPt)
    return (H, Si)
'''
I assume that i recieve 2 lists
'''
def make_P(list_kp1, list_kp2):
    k = len(list_kp1)
    # making P matrix
    P = np.zeros((k, 9))
    for i in range(k):
        # print(list_kp1[int(i/2)])
        x = list_kp1[int(i/2)][0]
        x_ = list_kp2[int(i/2)][0]
        y = list_kp1[int(i/2)][1]
        y_ = list_kp2[int(i/2)][1]
        P[i,:] = [x*x_, x*y_, x, y*x_, y*y_, y, x_, y_, 1]
    return P

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

def ptLocs(inter2d, lines, tol=1e-2): 
    ptDist = np.zeros((6,6)) # will store data for a pt in rows
    keys = list(inter2d.keys())
    vals = []
    for j in range(len(keys)):
        key = keys[j]
        vals.append(inter2d[key])
        ipt = inter2d[key]
        for i in range(len(lines)):
            line = lines[i]
            # print ('shape', line.shape)
            # print ('iptshape', ipt.shape, ipt[1], ipt)
            # sys.exit()
            d = line[0]*ipt[0] + line[1]*ipt[1] + line[2]
            if abs(d) < tol:
                d = 0
            ptDist[j, i] = d
    return vals, ptDist

def rem(inter2d):
    vals = []
    keys = list(inter2d.keys())
    for i in range(len(keys)):
        key = keys[i]
        if key[0] < 3 and key[1] >= 3:
            del inter2d[key]
    return inter2d

def label(img, pts, ptVectors, lines, tol=1):
    out = np.copy(img)
    cols, rows, chl = img.shape
    vector = np.zeros(6)
    for i in range(cols):
        for j in range(rows):
            la = lebely((j, i), lines, vector, tol)
            # print('shape', ptVectors.shape, la.shape)
            prodcts = ptVectors*la[:, None]
            ##### casess sattements here....
            if np.all(prodcts[0:3,0:3] >= 0):
                out[i, j] = [0,0,0]
    return out

def lebely(ipt, lines, vector, tol):
    for i in range(len(lines)):
        line = lines[i]
        d = line[0]*ipt[0] + line[1]*ipt[1] + line[2]
        if abs(d) < tol:
                d = 0
        vector[i] = d
    return vector