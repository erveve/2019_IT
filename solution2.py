import matplotlib.pyplot as plt
import numpy as np
from math import exp, pi


def get_gauss_kernel(sigma):
    n = 3*sigma
    p = n//2
    kernel = [[(exp(-(pow(i-p,2) + pow(j-p,2))/(2*pow(sigma,2)))/(sigma*pow(2*pi,1/2))) for i in range(n)] for j in range(n)]
    kernel/= np.array(kernel).sum()
    return kernel


def gfilter(img, sigma=3):
    window_size = 3*sigma
    img2 = np.zeros_like(img)
    kernel = get_gauss_kernel(sigma)
    p = window_size//2
    for k in range(img.shape[2]): # foreach color channel
        for i in range(p, img.shape[0]-p): # foreach row
            for j in range(p, img.shape[1]-p): # foreach column
                window = img[i-p:i+p+1, j-p:j+p+1, k]
                img2[i,j,k] = (kernel*window).sum()
    return img2


def main():
    img = plt.imread("img.png")[:, :, :3]
    sigma = 5
    img2 = gfilter(img, sigma)

    plt.imshow(img2)
    plt.show()


if __name__ == "__main__":
    main()