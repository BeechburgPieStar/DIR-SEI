import numpy as np

def Rotate_DA(x,out_type):

    def rotate_matrix(theta):
        m = np.zeros((2,2))
        m[0, 0] = np.cos(theta)
        m[0, 1] = -np.sin(theta)
        m[1, 0] = np.sin(theta)
        m[1, 1] = np.cos(theta)
        # print(m)
        return m

    [N, L, C] = np.shape(x)
    x_rotate1 = np.matmul(x, rotate_matrix(np.pi/2))
    x_rotate2 = np.matmul(x, rotate_matrix(np.pi))
    x_rotate3 = np.matmul(x, rotate_matrix(3*np.pi/2))
    x_DA = np.vstack((x, x_rotate1, x_rotate2, x_rotate3))
    if out_type == 0:
         x_DA = np.vstack((x, x_rotate1, x_rotate2, x_rotate3))
    elif out_type==1:
        x_DA = x, x_rotate1, x_rotate2, x_rotate3
    return x_DA