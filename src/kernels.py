import numpy as np

laplace_kernel_3 = np.array(
    [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ]
)
laplace_no_diag_3 = np.array(
    [
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0],
    ]
)
laplace_kernel_5 = np.array(
    [
        [-1, -1, -1, -1, -1],
        [-1, 2, 2, 2, -1],
        [-1, 2, 0, 2, -1],
        [-1, 2, 2, 2, -1],
        [-1, -1, -1, -1, -1],
    ]
)
sharpen_kernel_3 = np.array(
    [
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ]
)
unsharpen_kernel_3 = np.array(
    [
        [0, 1, 0],
        [1, -3, 1],
        [0, 1, 0],
    ]
)

sobel_horizontal = np.array([
    [-1,0,1],[-2,0,2],[-1,0,1]
])
sobel_vertical = np.array([
    [1,2,1],[0,0,0],[-1,-2,-1]
])


def mean_kernel(n):
    if n%2==0:
        raise ValueError("kernel size must be odd")
    return (1/(n*n))*np.ones(shape=(n,n))

def mean_kernel_with_vertical_strip(n,w):
    kernel = mean_kernel(n)
    strip_indicies = list(range(n//2-w//2,n//2+w//2+1))
    for i in range(n):
        for j in strip_indicies:
            kernel[i][j]=0
    
    kernel = kernel/np.sum(kernel)
    return kernel

def blur_x_strip(n,m):
    c = n-n//2-1
    d_to_center = lambda i: np.abs(i-c) if i!=c else 1
    kernel = np.array([[1/d_to_center(i) for i in range(n)] for j in range(m)])
    kernel = kernel/kernel.sum()
    return kernel

def blur_x_strip_with_hole(n,m,h):
    kernel = blur_x_strip(n,m)
    dx = (n-h)//2
    for i in range(dx,n-dx):
        for j in range(m):
            kernel[j][i]=0
    kernel = kernel/kernel.sum()
    return kernel

def blur_kernel(n):
    if n%2==0:
        raise ValueError("kernel size must be odd")
    
    c = n - n//2
    def d_to_center(i,j):
        s = np.sqrt((i-c)**2+(j-c)**2)
        if s==0:
            return 1
        return s
    kernel = [[1/d_to_center(i,j) for i in range(n)] for j in range(n)]
    kernel = kernel/np.sum(kernel)
    return np.array(kernel)

def blur_kernel_with_vertical_strip(n,w):
    kernel = blur_kernel(n)
    strip_indicies = list(range(n//2-w//2,n//2+w//2+1))
    for i in range(n):
        for j in strip_indicies:
            kernel[i][j]=0
    
    kernel = kernel/np.sum(kernel)
    return kernel
#
# n: kernel size
# h: hole size
def blur_kernel_with_hole(n,h):
    if n%2==0 or h%2==0:
        raise ValueError("kernel size and hole size must be odd")
    
    kernel = blur_kernel(n)
    hole_indicies = list(range(n//2-h//2,n//2+h//2+1))
    for i in hole_indicies:
        for j in hole_indicies:
            kernel[i][j]=0
    kernel = kernel/np.sum(kernel)
    return kernel
