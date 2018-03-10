from os import listdir, path
import cv2
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline


def sobel_operator(image):
    sobel_kernel_x = np.array([[-1, -2, -1],[0, 0, 0], [1, 2, 1]])
    sobel_kernel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    gradient_x = cv2.filter2D(image,-1,sobel_kernel_x)
    gradient_y = cv2.filter2D(image,-1,sobel_kernel_y)

    magnitude_grad =  np.power(np.power(gradient_x,2) + np.power(gradient_y,2),0.5)
    
    aprox_gradient = np.absolute(gradient_x) + np.absolute(gradient_y)

    return [magnitude_grad,aprox_gradient,gradient_x,gradient_y]

def main():
    filenames = listdir('images')

    for i,name in enumerate(filenames):
        print(name)
        img = cv2.imread("images/"+name, cv2.IMREAD_GRAYSCALE)
        fig = plt.figure(figsize=(14,14));
        plt.subplot(3,2,1)
        plt.imshow(img,cmap='gray'), plt.title("Original")
       
        [magnitude_grad,aprox_gradient,gx,gy]=sobel_operator(img)
        
        plt.subplot(3,2,3),plt.imshow(gx,cmap = 'gray')
        plt.title('Gradient X')
        
        plt.subplot(3,2,4),plt.imshow(gy,cmap = 'gray')
        plt.title('Gradient Y')
        
        plt.subplot(3,2,5),plt.imshow(aprox_gradient,cmap = 'gray')
        plt.title('Gradient magnitude aproximation')
        
        plt.subplot(3,2,6),plt.imshow(magnitude_grad,cmap= 'gray')
        plt.title('Gradient magnitude')
        
        plt.show()

main()
