import cv2
import matplotlib.pyplot as plt
%matplotlib inline

def median_filter(image,n):
    return cv2.medianBlur(image,n)

def averaging_filter(image,n):
    return cv2.blur(image,(n,n))

imagen = cv2.imread("images/ckt_board_saltpep_prob_pt05.png", cv2.IMREAD_GRAYSCALE)
plt.figure()
plt.imshow(imagen, cmap='gray')
plt.title('Imagen con Salt-Pepper noise')
plt.show()

kernel_size = [3,5,9,15] 
for n in kernel_size:
    plt.figure()
    
    plt.subplot(1,2,1)
    plt.imshow(median_filter(imagen,n), cmap='gray')
    plt.title('Median filter n='+str(n))
    
    plt.subplot(1,2,2)
    plt.imshow(averaging_filter(imagen,n), cmap='gray')
    plt.title('Averaging filter n='+str(n))
    
    plt.show()
