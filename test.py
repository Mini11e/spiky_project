import numpy as np



if __name__ == "__main__":
    array = np.zeros((3, 3))
    array[0,1] = 5
    array[2,2] = 3
    print(array)
    array = array.sum(axis=0)
    print(array)