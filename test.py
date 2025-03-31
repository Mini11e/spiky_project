import numpy as np



if __name__ == "__main__":

    arr = np.zeros(3)
    arr[1] = 1
    arr[2] = 2
    mat = np.zeros((3, 3)) 
    mat[0, 0] = 3
    mat[0, 1] = 4
    mat[0, 2] = 5
    mat[1, 0] = 6
    mat[1, 1] = 7
    mat[1, 2] = 8
    mat[2, 0] = 9
    mat[2, 1] = 10
    mat[2, 2] = 11

    dot = np.dot(mat, arr)
    dot2 = np.dot(arr, mat)

    print(arr)
    print(mat)
    print(dot)
    print(dot2)



