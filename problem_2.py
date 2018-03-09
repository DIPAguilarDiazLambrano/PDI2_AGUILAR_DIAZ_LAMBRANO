import cv2
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
#------------------------------------------------------------------------------
def T_fxy(imagen, gamma):
    imagen_out = 1 * np.power(imagen/255.0,gamma)
    return (imagen_out*255).astype('int')
#------------------------------------------------------------------------------
def processImage2(image):
    valores_gamma = [0.3, 0.5, 0.8, 1, 5, 10] 
    position = 1; 
    plt.figure(figsize = (12,12))
    for y in valores_gamma:
        plt.subplot(2, 3, position)
        position = position + 1
        plt.imshow(T_fxy(image,y), cmap='gray')
        plt.title("Valor Gamma = " + str(y))
        plt.axis('off')
#------------------------------------------------------------------------------
def main():
    imagenes = ["spine.jpg", "runway.jpg"]
    for nombre_imagen in imagenes:
        imagen = cv2.imread("images/"+nombre_imagen, cv2.IMREAD_GRAYSCALE)
        processImage2(imagen)
    plt.show()
#------------------------------------------------------------------------------
main()
