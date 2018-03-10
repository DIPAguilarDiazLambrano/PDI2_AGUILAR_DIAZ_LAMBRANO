import cv2
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
#----------------------------------------------------------------------------
#global variables
sobel_kernel_x = np.array([[-1, -2, -1],[0, 0, 0], [1, 2, 1]])
sobel_kernel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#------------------------------------------------------------------------------
def gradient(image):
    gradient_x = cv2.filter2D(image,-1,sobel_kernel_x)
    gradient_y = cv2.filter2D(image,-1,sobel_kernel_y)
    return [gradient_x, gradient_y]
#------------------------------------------------------------------------------
def magnitude_gradient(image):
    gradient_x, gradient_y = gradient(image)
    mag_grad =  np.power(np.power(gradient_x,2) + np.power(gradient_y,2),0.5)
    return mag_grad
#------------------------------------------------------------------------------
def aprox_gradient(image):
    gradient_x, gradient_y = gradient(image)
    ap_gradient = np.absolute(gradient_x) + np.absolute(gradient_y)    
    return ap_gradient
#------------------------------------------------------------------------------
def plot_image(img, title):
    plt.imshow(img,cmap = 'gray')
    plt.title(title)
    plt.axis('off')
#------------------------------------------------------------------------------
def processImg(name):
    # COMPUTER 
    # original image      
    img = cv2.imread("images/"+name, cv2.IMREAD_GRAYSCALE)
    # gradient_x and y
    [Gx, Gy] = gradient(img)
    # magnitude_grad
    mag_grad = magnitude_gradient(img)
    # aprox_gradient
    ap_gradient = aprox_gradient(img)

    # display original image 
    plt.figure(figsize=(12,12))
    plt.subplot(1,2,1)
    plot_image(img, 'Original Image')
    
    # display gradients
    plt.figure(figsize=(12,12))
    plt.subplot(2,2,1)
    plot_image(Gx, 'Gradient X')
    plt.subplot(2,2,2)
    plot_image(Gy, 'Gradient Y')
    # display gradient magnitude 
    plt.subplot(2,2,3)
    plot_image(mag_grad, 'Magnitude Gradient')
    plt.subplot(2,2,4)
    plot_image(ap_gradient, 'Magnitude Gradient Aproximation')
    
    plt.suptitle('Gradient', fontsize=16)
#------------------------------------------------------------------------------
def main():
    filenames = ['contact_lens_original.png', 'face.png']
    for name in filenames:
        processImg(name)
    plt.show()
#------------------------------------------------------------------------------
main()
