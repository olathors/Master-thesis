import numpy as np
from torchvision.transforms import functional, RandAugment
import random

np.random.seed(110323)
random.seed(110323)

def augment(X):

    p = np.random.randint(0, 100)/100

    m = np.random.randint(0, 100)/100

    n_color = np.random.randint(0, 9)

    #n_shape = np.random.randint(0, 100)

    #X = color_augment(X, p, m, n_color)

    #X = shape_augment(X, p, m, n_shape)

    aug = RandAugment(magnitude=2)

    X = aug(X)

    return X

def color_augment(X, p, m, n):

    transformations = random.sample(range(0, 9), n)

    for transformation in transformations:

        if transformation == 1 and np.random.randint(0, 100)/100 < p:
             
            #X = auto_contrast(X, m)
            X = X

        elif transformation == 2 and np.random.randint(0, 100)/100 < p:

            X = equalize(X, m)

        elif transformation == 3 and np.random.randint(0, 100)/100 < p:

            X = invert(X, m)

        elif transformation == 4 and np.random.randint(0, 100)/100 < p:

            X = posterize(X, m)

        elif transformation == 5 and np.random.randint(0, 100)/100 < p:

            X = solarize(X, m)

        elif transformation == 6 and np.random.randint(0, 100)/100 < p:

            X = contrast(X, m)

        elif transformation == 7 and np.random.randint(0, 100)/100 < p:

            X = brightness(X, m)

        elif transformation == 8 and np.random.randint(0, 100)/100 < p:

            X = sharpness(X, m)

        elif transformation == 9 and np.random.randint(0, 100)/100 < p:

            #X = gaussian_noise(X, m)
            X = X

        elif transformation == 0 and np.random.randint(0, 100)/100 < p:

            #X = gaussian_blur(X, m)
            X = X



    return X


def shape_augment(X, p, m, n):


    transformations = []

    return X

def auto_contrast(X, m):

    X = functional.autocontrast(X)

    return X

def equalize(X, m):

    X = functional.equalize(X)

    return X

def invert(X, m):

    X = functional.invert(X)

    return X

def posterize(X, m):

    #Remove 0 - 4 bits from the image depth.

    m = 4 + np.round(4*m)

    X = functional.posterize(X, m)

    return X

def solarize(X, m):

    m = np.round(256*m)

    X = functional.solarize(X, m)

    return X

def contrast(X, m):

    m = 0.1 + (1.8*m)

    X = functional.adjust_contrast(X,m)

    return X

def brightness(X, m):

    m = 0.1 + (1.8*m)

    X = functional.adjust_brightness(X,m)

    return X

def sharpness(X, m):

    m = 0.1 + (1.8*m)

    X = functional.adjust_sharpness(X,m)

    return X

def gaussian_noise(X, m):
    
    X = functional.gaussian_noise(X)

    return X

def gaussian_blur(X, m):
    
    m = np.round(2*m)

    m = int(m)
    
    X = functional.gaussian_blur(X, m)

    return X