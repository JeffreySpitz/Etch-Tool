import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import math

# get Stats returns the perimeter and density of the etching
# Asumes the image is larger by 1px used identify perimeters
def getStats(img):
    # Convert image to binary where white is 1 and all other colors are 0
    im1 = (img[:,:,0]==255)*(img[:,:,1]==255)*(img[:,:,2]==255)
    binImg = im1.astype(int)
    p = getPerimeter(binImg)
    # Crop out extra image perimeter
    binImg = binImg[1:-1,1:-1]
    d = getDensity(binImg)
    return p, d
    

def getPerimeter(img):
    struct1 = ndimage.generate_binary_structure(2, 1)
    expanded = ndimage.binary_dilation(img, structure=struct1).astype(img.dtype)
    borders = expanded - img
    # Crop out extra image perimeter
    borders = borders[1:-1,1:-1]
    return np.sum(borders)
    
    

def getDensity(img):
    # Input image should be binary ones and zeros representing free and etched space 
    # We need to flip the ones and zeros so ones count for etched areas
    areas = 1-img
    area = np.sum(areas)
    totArea = areas.size
    return float(area)/float(totArea)

# avgWindow: size of nxn window we use to average the input
def averageData(data, avgWindow):
    data['windowSize'] = data['windowSize']*avgWindow
    data['windowResolution'] = data['windowResolution']*avgWindow
    data['isAveraged'] = True
    data['perimeter'] = averageImg(data['perimeter'], avgWindow)
    data['density'] = averageImg(data['density'], avgWindow)

# img: image to be averaged
# avgWindow: size of nxn window we use to average the input
def averageImg(img, avgWindow):
    s = img.shape
    if s[0]%avgWindow:
        xPad = s[0] - (math.floor(s[0]/avgWindow) + 1)*avgWindow
    if s[1]%avgWindow:
        yPad = s[1] - (math.floor(s[1]/avgWindow) + 1)*avgWindow
    if xPad > 0 or yPad > 0:
        warnings.warn("Warning: avgWindow ("+str(avgWindow)+") does not divide the size of die image ("+str(s[0])+","+str(s[1])+") evenly. The die will be padded with pixels to accomodate averaging with the current window size")
        img = np.pad(a, ((0,xPad),(0,yPad)),'constant')
    outputData = np.zeros((s[0]/avgWindow, s[1]/avgWindow))
    
    xVals = np.arange(0, s[0], avgWindow)
    yVals = np.arange(0, s[1], avgWindow)
    for i in range(len(xVals)):
        for j in range(len(yVals)):
            outputData[i,j] = img[xVals:xVals+avgWindow,yVals:yVals+avgWindow]
    