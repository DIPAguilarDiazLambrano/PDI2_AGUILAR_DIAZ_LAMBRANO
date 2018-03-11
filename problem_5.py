import functools
import cv2
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
#-------------------------------------------------------------------------------
def T_fxy(imagen, gamma):
    imagen_out = 1 * np.power(imagen/255.0,gamma)
    return (imagen_out*255).astype('int')
#-------------------------------------------------------------------------------
def laplacian(image, kernel):
    output = cv2.filter2D(image, -1,kernel.astype(float))
    return np.abs(output).astype('int')
#-------------------------------------------------------------------------------
def directSharpen5(image):
    return laplacian(image, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
#-------------------------------------------------------------------------------
def altSharpen8(image):
    out1 = laplacian(image,np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]]))
    out = np.subtract(image,out1)
    return out
#------------------------------------------------------------------------------
def processImage5(name):
    # load original image
    img = cv2.imread("images/" + name, cv2.IMREAD_GRAYSCALE)
    plt.bwimshow = functools.partial(plt.imshow, vmin=0, vmax=255)
    # display original image
    plt.figure(figsize = (14, 14))
    plt.subplot(1,2,1)
    plt.bwimshow(img, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    # calculate laplacian 
    img_1 = laplacian(img, np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]))
    img_2 = np.subtract(img,img_1)
    # new figure
    fig = plt.figure(figsize=(12,12))
    plt.subplot(2,2,1)
    # display laplacian 
    # Use a gamma correction to show details 
    plt.bwimshow(T_fxy(img_1,0.5), cmap='gray')
    plt.title('Laplacian') 
    plt.axis('off')
    # Display the subtraction between original image and laplacian 
    plt.subplot(2,2,2)
    plt.bwimshow(img_2, cmap='gray')
    plt.title('Sharpened Image')
    plt.axis('off')
    # Display the subtraction between original image and lapliacian 
    plt.subplot(2,2,3)
    plt.bwimshow(directSharpen5(img), cmap='gray')
    plt.title('Direct Sharpened') 
    plt.axis('off')
    # Display the subtraction between original image and full laplacian image 
    plt.subplot(2,2,4)
    plt.bwimshow(altSharpen8(img), cmap='gray')
    plt.title('Alternative Sharpened') 
    plt.axis('off')
    
    fig.tight_layout()
#-------------------------------------------------------------------------------
def main(): 

    imagenes = ["blurry_moon.png"]

    for name in imagenes:
        processImage5(name)
        
    plt.show()
#-------------------------------------------------------------------------------------
main()
