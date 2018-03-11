import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
#------------------------------------------------------------------------------
def median_filter(image,n):
    return cv2.medianBlur(image,n)
#------------------------------------------------------------------------------
def averaging_filter(image,n):
    return cv2.blur(image,(n,n))
#------------------------------------------------------------------------------
def main():
    imagen = cv2.imread("images/ckt_board_saltpep_prob_pt05.png", cv2.IMREAD_GRAYSCALE)
    # Display original image
    plt.figure(figsize = (12,12))
    plt.subplot(1,2,1)
    plt.imshow(imagen, cmap='gray')
    plt.axis('off')
    plt.title('Image with Salt-Pepper noise')
    kernel_size = [3, 5, 9, 15]
    for n in kernel_size:
        # Display median filter output 
        plt.figure(figsize = (12,12))
        plt.subplot(1,2,1)
        plt.imshow(median_filter(imagen,n), cmap='gray')
        plt.title('Median filter n='+str(n))
        plt.axis('off')
        # Display averaging filter output
        plt.subplot(1,2,2)
        plt.imshow(averaging_filter(imagen,n), cmap='gray')
        plt.title('Averaging filter n='+str(n))
        plt.axis('off')
    plt.show()
#-------------------------------------------------------------------------------
main()
