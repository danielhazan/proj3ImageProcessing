import numpy as np
import math
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from scipy.misc import imread
from skimage.color import rgb2gray
import os


def relpath(filename):
    """retrns the relative path of the given filename, as instructed in the PDF"""
    return os.path.join(os.path.dirname(__file__),filename)


def read_image(filename, representation):
    image = imread(filename)
    if(len(image.shape)<3):
        #the third dimension which indicates the colour-channels
        #is missing, meaning its a gray-scale image

        #convert it to float64
        im_float = image.astype(np.float64)
        im_float /= 255
        return im_float
    if(len(image.shape) == 3):
        #RGB image

        if(representation ==1):
            #convert to gray-scale
            im_g = rgb2gray(image)
            im_g = im_g.astype(np.float64)


            return im_g
        if representation ==2:
            im_f =  image.astype(np.float64)
            im_f /= 255

            return im_f




def gaussKernel(kernel_size):

    GaussKernel = np.array([1,1])
    while len(GaussKernel) < kernel_size:
        GaussKernel  = np.convolve(GaussKernel,[1,1]).astype(np.float64) #using binomial coefficients


    GaussKernel = GaussKernel/np.sum(GaussKernel)#normalize the kernel
    return GaussKernel.astype(np.float64)


def build_gaussian_pyramid(im,max_levels,filter_size):

    #first compute the num of levels for not succeeding the image' size
    numOfLevels = min(max_levels,np.floor(math.log(min(im.shape[0],im.shape[1]),2)) -3)
    if isinstance(numOfLevels,float):
        numOfLevels = numOfLevels.astype(int)
    gaussPyrArray = [1 for i in range(numOfLevels)]
    gaussFilter = gaussKernel(filter_size).reshape((1,-1))
    gaussIm = im.copy()
    for i in range(numOfLevels):
        gaussPyrArray[i] = gaussIm
        gaussIm = blurAndReduce(gaussIm,gaussFilter)

    return gaussPyrArray, gaussFilter

def blurAndReduce(gaussIm,gaussFilter):

    bluredIm = convolve2d(gaussIm, gaussFilter.reshape(-1,1),"same")
    bluredIm = convolve2d(bluredIm,np.transpose(gaussFilter).reshape(1,-1),'same')
    return bluredIm[0::2,0::2]

def build_laplacian_pyramid(im,max_levels,filter_size):

    gaussArray, filter = build_gaussian_pyramid(im,max_levels,filter_size)
    numOfLevels = len(gaussArray)
    laplacePyrArray = [1 for k in range(numOfLevels)]
    for i in range(numOfLevels -1):
        blAndExpand = blurAndExpand(gaussArray[i+1],filter)
        laplacePyrArray[i] = gaussArray[i] - blAndExpand
    laplacePyrArray[numOfLevels-1] = gaussArray[numOfLevels-1]
    return laplacePyrArray, filter

def blurAndExpand(gaussIm, filter):
    """blurring the gaussian image by convolving with the filter by rows and columns
    :param gaussIm the gaussian image
    :param filter a row vector of the gaussian filter"""

    filter = filter*2
    expandedIm = np.zeros((2*gaussIm.shape[0],2*gaussIm.shape[1]))

    #pad with zeros -->
    expandedIm[1::2,1::2] = gaussIm# expand by padding with zeros every odd pixel!
    expandedIm = convolve2d(expandedIm,filter.reshape(-1,1), 'same')
    expandedIm = convolve2d(expandedIm,np.transpose(filter).reshape(1,-1), 'same')
    return expandedIm.copy()

def imageNormalize(image):

    """streching the values of image before displaying in Ptramid"""
    return (image - np.amin(image))/(np.amax(image) - np.amin(image))

def laplacian_to_image(lpyr, filter_vec,coeff):
    image = lpyr[-1]
    for i in range(len(lpyr),1,-1):
        image = coeff[i-2]*lpyr[i-2] + blurAndExpand(image,filter_vec)
    return image

def pyramid_blending(im1, im2, mask,max_levels, filter_size_im,filter_size_mask):

    gauss_vec_im = gaussKernel(filter_size_im)
    gauss_vec_mask = gaussKernel(filter_size_mask)
    laplPyrIm1,filter1 = build_laplacian_pyramid(im1,max_levels,filter_size_im)
    laplPyrIm2,filter2 = build_laplacian_pyramid(im2,max_levels,filter_size_im)
    gaussPyrMask, filterG = build_gaussian_pyramid(mask.astype(float),max_levels,filter_size_mask)

    laplPyrOut = [1 for k in range(len(laplPyrIm1))]
    laplPyrOut[0] = gaussPyrMask[0]*laplPyrIm1[0] +(1-gaussPyrMask[0])*laplPyrIm2[0]
    for i in range(1,len(laplPyrIm1)):
        laplPyrOut[i] = gaussPyrMask[i]*laplPyrIm1[i] +(1-gaussPyrMask[i])*laplPyrIm2[i]

    resultIm = laplacian_to_image(laplPyrOut,filter1,[1 for i in range(len(laplPyrOut))])
    return resultIm

def pyramid_blendingRGB(im1, im2, mask,max_levels, filter_size_im,filter_size_mask):
    imageBlended = np.zeros(im1.shape)
    for i in range(3):
        image1 = im1[:,:,i]
        image2 = im2[:,:,i]
        imageBlended[:,:,i] = pyramid_blending(image1,image2,mask,max_levels,filter_size_im,filter_size_mask)
    return imageBlended


def blending_example1():
    image1 = read_image(relpath('externals/glasses_1512.jpg'),2)
    image2 = read_image(relpath('externals/aqua_1512.jpg'),2)
    mask = np.round(read_image(relpath('externals/aquaMask.jpg'),1))
    blendedIm = pyramid_blendingRGB(image1,image2,mask,7,75,55)
    displaying_Blending(image1,image2,mask,blendedIm)
    return image1,image2, mask.astype(bool), blendedIm



def blending_example2():
    image1 = read_image(relpath('externals/mokey512.jpg'),2)
    image2 = read_image(relpath('externals/elaphant512.jpg'),2)
    mask = np.round(read_image(relpath('externals/elaphant512Mask.jpg'),1))
    blendedIm = pyramid_blendingRGB(image1,image2,mask,7,75,55)
    displaying_Blending(image1,image2,mask,blendedIm)
    return image1,image2, mask.astype(bool), blendedIm


def displaying_Blending(image1,image2,mask,blendedIm):
    """this function displaying images before anfd after blending with mask
    :param image1 - original image
    :param image2 - image from which the mask is taken
    :param mask - binary image taken from part of image2
    :param blendedIm - the blended image after applying the mask on image1"""

    fig = plt.figure(figsize=(10,10))
    fig.add_subplot(2,2,1)
    plt.imshow(image1)
    fig.add_subplot(2,2,2)
    plt.imshow(image2)
    fig.add_subplot(2,2,3)
    plt.imshow(mask,cmap='gray')
    fig.add_subplot(2,2,4)
    plt.imshow(blendedIm)
    plt.show()

def render_pyramid(pyr,levels):
    rows_array = []
    cols_array = []

    for im in pyr:
        rows_array.append(im.shape[0])
        cols_array.append(im.shape[1])
    res = np.zeros((rows_array[0],np.sum(cols_array[:levels])))
    columnBorders =  np.cumsum(cols_array)
    columnBorders = np.insert(columnBorders,0,0)
    for i in range(levels):
        im = pyr[i].copy()
        im = imageNormalize(im)
        res[:rows_array[i],columnBorders[i] :columnBorders[i+1]] = im
    return res

def display_pyramid(pyr,levels):
    res = render_pyramid(pyr,levels)
    plt.imshow(res, cmap= 'gray')
plt.show()
