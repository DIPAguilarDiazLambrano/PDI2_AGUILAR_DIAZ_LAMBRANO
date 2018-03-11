import cv2
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
#-------------------------------------------------------------------------------
def salt_peeper_noise(imagen,porcentaje):
    imagen_out = imagen.copy()
    porcentaje = porcentaje/100.0
    npuntos = int(porcentaje/2 * imagen.size) #numero de puntos de sal y pimienta
    salt_points = [np.random.randint(0, i - 1, npuntos) for i in imagen.shape]
    imagen_out[salt_points] = 255 #Salt
    peeper_points = [np.random.randint(0, i - 1, npuntos) for i in imagen.shape]
    imagen_out[peeper_points] = 0 #Pepper
    return imagen_out
#-------------------------------------------------------------------------------
def median_filter(image,n):
    return cv2.medianBlur(image,n)
#-------------------------------------------------------------------------------
def averaging_filter(image,n):
    return cv2.blur(image,(n,n))
#-------------------------------------------------------------------------------
def processImage(name_image):
    original_image = cv2.imread("images/" + name_image, cv2.IMREAD_GRAYSCALE)
    # Display original image
    fig = plt.figure(figsize=(12,12))
    plt.subplot(2,2,1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')
    noise_image = salt_peeper_noise(original_image,25)
    # Display noised image 
    plt.subplot(2,2,2)
    plt.imshow(noise_image, cmap='gray')
    plt.title('Imagen con ruido')
    plt.axis('off')
    # Display output filter  
    plt.subplot(2,2,3)
    plt.imshow(median_filter(noise_image,3), cmap='gray')
    plt.title('Median filter n = 3')
    plt.axis('off')
    #Display output filter 
    plt.subplot(2,2,4)
    plt.imshow(median_filter(noise_image,5), cmap='gray')
    plt.title('Median filter n=5')
    plt.axis('off')
    fig.tight_layout() 
#-------------------------------------------------------------------------------
def main():
    imagenes = ["test_pattern_blurring_orig.png","pollen.jpg"]

    for name in imagenes:
        processImage(name)
        
    plt.show()
#--------------------------------------------------------------------------------------
main()
