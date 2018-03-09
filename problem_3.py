import cv2 #libreria de procesamiento de imagenes
import numpy as np
from scipy import ndimage #liberia contiene filtros
from matplotlib import pyplot as plt #libreria de graficos de imagenes
#%matplotlib inline
#------------------------------------------------------------------------------
def blurre_image(image, sigma):
    return ndimage.gaussian_filter(image, sigma = sigma)
    
#------------------------------------------------------------------------------    
def main():
    imagen = cv2.imread('images/test_pattern_blurring_orig.png',cv2.IMREAD_COLOR) 
    sigma_list = [3, 5, 9, 15, 35]
    #Carga la imagen con el formato RGB
    plt.figure(figsize = (14, 14))
    plt.subplot(2, 3, 1)
    plt.imshow(imagen, cmap='gray')
    plt.axis('off')
    plt.title('Original Image')
    pos = 2     
    for sigma in sigma_list:
        plt.subplot(2, 3, pos)
        pos = pos + 1
        blurred_image = blurre_image(imagen, sigma)
        plt.imshow(blurred_image, cmap='gray')
        plt.axis('off')
        plt.title('Blurred Image. sigma = ' + str(sigma))
    plt.show()
#----------------------------------------------------------------------------
main()
