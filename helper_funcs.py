import os.path
from gdsCAD import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time
import warnings
import pickle
import math
from scipy import ndimage

# returns a dictionary containing all data and metadata of GDS process
# filename: name of gds file
# minPt: minimum boundary coordinates of the die
# maxPt: maximum boundary coordinates of the die
# windowSize: size of the nxn window used to analyze the die
# windowResolution: pxp resolution of the window
# outputFile: file to save dictionary output to
def analyzeGDS(filename, minPt, maxPt, windowSize, windowResolution, outputFile='none'):
    # Allow for 1 pixel pad:
    padSize = float(windowSize)/windowResolution
    padWindowSize = windowSize + padSize*2
    windowResolution = windowResolution+2
    
    # Start and end positions:
    totX = maxPt[0] - minPt[0]
    totY = maxPt[1] - minPt[1]
    if totX%windowSize or totY%windowSize:
        warnings.warn("Warning: windowSize ("+str(windowSize)+") does not divide the size of die ("+str(totX)+","+str(totY)+") evenly. Some pixels outside the die's range will be included in the calculations")
        if totX%windowSize:
            maxPt[0] = (1 + math.floor(totX/windowSize))*windowSize + minPt[0]
        if totY%windowSize:
            maxPt[1] = (1 + math.floor(totY/windowSize))*windowSize + minPt[1]
    # Display what is to be analyzed
    fig0 = plt.figure()
    ax = fig0.add_subplot(1, 1, 1)
    ourLayout = core.GdsImport('output.gds')
    ourLayout.show()
    ax.axis([minPt[0], maxPt[0], minPt[1], maxPt[1]])
    
    # Setup Figure
    fig, ax = plt.subplots(figsize=(1,1),constrained_layout=True, dpi=windowResolution)
    layout = core.GdsImport(filename)
    layout.show()
    ax.axis('off')
    ax.set_position([0, 0, 1, 1])
    print(padSize,padWindowSize)
    
    xVals = np.arange(minPt[0], maxPt[0], windowSize) - padSize
    yVals = np.arange(minPt[1], maxPt[1], windowSize) - padSize
    
    perimeter = np.zeros((len(xVals), len(yVals)))
    density = np.zeros((len(xVals), len(yVals)))
    
    for i in log_progress(range(len(xVals)), every=1):
        for j in range(len(yVals)):
            xVal = xVals[i]
            yVal = yVals[j]
            ax.axis([xVal, xVal+padWindowSize, yVal, yVal+padWindowSize])
            canvas = plt.get_current_fig_manager().canvas
            agg = canvas.switch_backends(FigureCanvasAgg)
            agg.draw()
            s = agg.tostring_rgb()
            # get the width and the height to resize the matrix
            l, b, w, h = agg.figure.bbox.bounds
            w, h = int(w), int(h)
            X = np.frombuffer(s, dtype='uint8')
            X.shape = h, w, 3
            p,d = getStats(X)
            perimeter[i,j] = p
            density[i,j] = d
    outputDict = {
        "filename":filename,
        "minPt":minPt,
        "maxPt":maxPt,
        "windowSize":windowSize,
        "windowResolution":windowResolution,
        "perimeter":np.rot90(perimeter),
        "density":np.rot90(density),
        "isAveraged": False
    }
    if outputFile != 'none':
        save_obj(outputDict, outputFile)
    return outputDict
    
               
    
    

def boxExample():
    fig, ax = plt.subplots(figsize=(1,1),constrained_layout=True, dpi=8)
    box = shapes.Rectangle((0, 0), (400, 400), layer=2)
    cell = core.Cell('EXAMPLE')
    cell.add([box])
    layout = core.Layout('LIBRARY')
    layout.add(cell)
    layout.show()
    ax.axis([-1, 1, -1, 1])
    ax.axis('off')
    ax.set_position([0, 0, 1, 1])
    canvas = plt.get_current_fig_manager().canvas
    agg = canvas.switch_backends(FigureCanvasAgg)
    agg.draw()
    s = agg.tostring_rgb()
    plt.close(fig)
    # get the width and the height to resize the matrix
    l, b, w, h = agg.figure.bbox.bounds
    w, h = int(w), int(h)
    X = np.frombuffer(s, dtype='uint8')
    X.shape = h, w, 3
    return X

def loopTest():
    a = [1,2,3,4,5,6,7,8,9,10]
    for elem in log_progress(a, every=3):
        time.sleep(.2)

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )

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
def avgData(die, avgWindow):
    die['windowSize'] = die['windowSize']*avgWindow
    die['windowResolution'] = die['windowResolution']*avgWindow
    die['isAveraged'] = True
    die['perimeter'] = sumImg(die['perimeter'], avgWindow)
    die['density'] = averageImg(die['density'], avgWindow)
    return die

# img: image to be averaged
# avgWindow: size of nxn window we use to average the input
def averageImg(img, avgWindow):
    s = img.shape
    xPad = 0
    yPad = 0
    if s[0]%avgWindow:
        xPad = s[0] - (math.floor(s[0]/avgWindow) + 1)*avgWindow
    if s[1]%avgWindow:
        yPad = s[1] - (math.floor(s[1]/avgWindow) + 1)*avgWindow
    if xPad > 0 or yPad > 0:
        warnings.warn("Warning: avgWindow ("+str(avgWindow)+") does not divide the size of die image ("+str(s[0])+","+str(s[1])+") evenly. The die will be padded with pixels to accomodate averaging with the current window size")
        img = np.pad(a, ((0,xPad),(0,yPad)),'constant')
    outputData = np.zeros((s[0]/avgWindow, s[1]/avgWindow))
    
    xVals = np.arange(0, s[0], avgWindow).astype(int)
    yVals = np.arange(0, s[1], avgWindow).astype(int)
    for i in range(len(xVals)):
        for j in range(len(yVals)):
            outputData[i,j] = np.mean(img[xVals[i]:xVals[i]+avgWindow,yVals[j]:yVals[j]+avgWindow])
    return outputData
    
# img: image to be summed
# sumWindow: size of nxn window we use to sum the input (used for perimeter)
def sumImg(img, sumWindow):
    s = img.shape
    xPad = 0
    yPad = 0
    if s[0]%sumWindow:
        xPad = s[0] - (math.floor(s[0]/sumWindow) + 1)*sumWindow
    if s[1]%sumWindow:
        yPad = s[1] - (math.floor(s[1]/sumWindow) + 1)*sumWindow
    if xPad > 0 or yPad > 0:
        warnings.warn("Warning: sumWindow ("+str(sumWindow)+") does not divide the size of die image ("+str(s[0])+","+str(s[1])+") evenly. The die will be padded with pixels to accomodate summing with the current window size")
        img = np.pad(a, ((0,xPad),(0,yPad)),'constant')
    outputData = np.zeros((s[0]/sumWindow, s[1]/sumWindow))
    
    xVals = np.arange(0, s[0], sumWindow).astype(int)
    yVals = np.arange(0, s[1], sumWindow).astype(int)
    for i in range(len(xVals)):
        for j in range(len(yVals)):
            outputData[i,j] = np.sum(img[xVals[i]:xVals[i]+sumWindow,yVals[j]:yVals[j]+sumWindow])
    return outputData

def dieWaferGrid(die, mmDiameter):
    # convert diameter from mm to micrometers
    diameter = mmDiameter*1000
    radius = float(diameter)/2
    dieShape = die['density'].shape
    dieLen = die['windowSize']*dieShape[0]
    dieWidth = die['windowSize']*dieShape[1]
    dieX = math.floor(radius/dieLen)
    dieY = math.floor(radius/dieWidth)
    x = np.append(np.arange(-1*(dieX), 0, 1),np.arange(1, dieX+1, 1))
    y = np.append(np.arange(-1*(dieY), 0, 1),np.arange(1, dieY+1, 1))
    xx, yy = np.meshgrid(x,y)
    xx = xx*dieLen
    yy = yy*dieWidth
    waferGrid = (xx**2+yy**2) <= radius**2
    return waferGrid

def fillWafer(die, mmDiameter):
    return setWafer([die], dieWaferGrid(die, mmDiameter))

def setWafer(dies, waferGrid):
    gridShape = waferGrid.shape
    dieShape = dies[0]['density'].shape
    wafer = np.zeros((gridShape[0]*dieShape[0], gridShape[1]*dieShape[1]))
    for i in range(gridShape[0]):
        for j in range(gridShape[1]):
            if waferGrid[i,j]:
                die = dies[waferGrid[i,j]-1]
                startX = i*dieShape[0]
                startY = j*dieShape[1]
                wafer[startX:startX+dieShape[0], startY:startY+dieShape[1]] = die['density']
    return wafer
    