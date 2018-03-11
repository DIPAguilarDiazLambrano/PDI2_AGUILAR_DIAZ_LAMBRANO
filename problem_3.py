import cv2 #libreria de procesamiento de imagenes
import numpy as np
from scipy import ndimage #liberia contiene filtros
from matplotlib import pyplot as plt #libreria de graficos de imagenes
#%matplotlib inline
#------------------------------------------------------------------------------
def blurre_image(image, sigma):
    return cv2.blur(image,(sigma, sigma))
#------------------------------------------------------------------------------
def gaussian_filter(image, sigma):
    return ndimage.gaussian_filter(image, sigma = sigma)
#------------------------------------------------------------------------------    
def main():
    # Load image using RGB format
    imagen = cv2.imread('images/test_pattern_blurring_orig.png',cv2.IMREAD_COLOR) 
    sigma_list = [3, 5, 9, 15, 35]
    # display original image
    plt.figure(figsize = (15,15))
    plt.subplot(3, 2, 1)
    plt.imshow(imagen, cmap='gray')
    plt.axis('off')
    plt.title('Original Image')
    pos = 2     
    for sigma in sigma_list:
        plt.subplot(3, 2, pos)
        pos = pos + 1
        blurred_image = blurre_image(imagen, sigma)
        plt.imshow(blurred_image, cmap='gray')
        plt.axis('off')
        plt.title('Blurred Image. Averange matrix ' + str(sigma) + 'x'+ str(sigma))
    plt.show()
#----------------------------------------------------------------------------
main()
