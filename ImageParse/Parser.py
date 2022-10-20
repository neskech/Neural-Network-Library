
DOG_PATH = 'ImageParse/Images/PetImages/Dog/'
CAT_PATH = 'ImageParse/Images/PetImages/Cat/'
WRITE_PATH_DOGS = 'ImageParse/Dogs.txt'
WRITE_PATH_CATS = 'ImageParse/Cats.txt'
import numpy as np
import cv2


def getData(numCats : int, numDogs : int, resize_size : tuple[int,int], isGrayScale : bool):
        shape = (numCats + numDogs, resize_size[0] * resize_size[1])
        print(shape)
        X = np.empty(shape, dtype=np.float64)
        Y = np.empty(numCats + numDogs, dtype=int)
        
        for i in range(numCats):
            image = cv2.imread(CAT_PATH + str(i) + '.jpg')
            image = cv2.resize(image, resize_size)
            if isGrayScale:
                image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            image = image.reshape((-1,))  
            X[i] = image
            Y[i] = 0

        shift = numCats
        for i in range(numDogs):
            image = cv2.imread(CAT_PATH + str(i) + '.jpg')
            image = cv2.resize(image, resize_size)
            if isGrayScale:
                image=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.reshape((-1,))   
            X[i + shift] = image
            Y[i + shift] = 1
            
        return X, Y


