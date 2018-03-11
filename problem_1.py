from os import listdir, path
import cv2
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

#-------------------------------------------------------------------------------
def processImage(matrix_image, name):
    # define figure
    plt.figure(figsize = (12,12))
    # display original picture
    plt.subplot(2,2,1)
    plt.imshow(matrix_image, cmap = 'gray')
    plt.title("Original picture")
    plt.axis('off')
    # display orignal histogram and cdf
    #plt.subplot
    plt.subplot(2,2,2)
    histogram(matrix_image)
    plt.title("Original Histogram and CDF")
    # display equzalized image
    plt.subplot(2,2,3)
    equ_img = cv2.equalizeHist(matrix_image)
    plt.imshow(equ_img, cmap = 'gray')
    plt.title('Equalized Image')
    plt.axis('off')
    # display equalized histogram
    plt.subplot(2,2,4)
    histogram(equ_img)
    plt.title('Equalized Histogram and CDF')
    plt.suptitle('Image: ' + name , fontsize=16)
#-------------------------------------------------------------------------------
def histogram(image):
    [hist_image, bins] = np.histogram(image.ravel(),256, normed = True);
    original_cdf = hist_image.cumsum()
    #normalize cdf
    h_max = hist_image.max()
    normalized_cdf = original_cdf * h_max/ original_cdf.max()
    #plot histogram and cdf
    plt.plot(normalized_cdf, color = 'r')
    plt.hist(image.ravel(), 256, color = 'b', normed = True) 
    plt.xlim([0,256])
    plt.grid(True)

#------------------------------------------------------------------------------
def equalize_image(image):
    equalized_image = cv2.equalizeHist(image)
    return equalized_image

#-------------------------------------------------------------------------------
def main():
    #images = [] # List with all the images read
    image_names = ['darkPollen.jpg', 'lightPollen.jpg', 'lowContrastPollen.jpg', 
                   'pollen.jpg', 'spine.jpg', 'runway.jpg']
    for name in (image_names):
        #load image as RCB format
        #image = cv2.imread("images/" + name, cv2.IMREAD_COLOR);
        image = cv2.imread("images/" + name, 0)
        processImage(image, name)
    plt.show()

#-------------------------------------------------------------------------------
main()
