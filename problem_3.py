import cv2 #libreria de procesamiento de imagenes
from scipy import ndimage #liberia contiene filtros
from matplotlib import pyplot as plt #libreria de graficos de imagenes
#%matplotlib inline
#------------------------------------------------------------------------
def main():
    imagen = cv2.imread('test_pattern_blurring_orig.PNG',cv2.IMREAD_COLOR) 
    #Carga la imagen con el formato RGB
    plt.figure()
    blurred = ndimage.gaussian_filter(imagen, sigma=3)
    very_blurred = ndimage.gaussian_filter(imagen, sigma=5)
    very_very_blurred = ndimage.gaussian_filter(imagen, sigma=9)
    very_very_very_blurred = ndimage.gaussian_filter(imagen, sigma=15)
    very_very_very_very_blurred = ndimage.gaussian_filter(imagen, sigma=35)
    plt.imshow(imagen)
    plt.show()
    plt.imshow(blurred)
    plt.show()
    plt.imshow(very_blurred)
    plt.show()
    plt.imshow(very_very_blurred)
    plt.show()
    plt.imshow(very_very_very_blurred)
    plt.show()
    plt.imshow(very_very_very_very_blurred)
    plt.show()
#------------------------------------------------------------------------
main()
