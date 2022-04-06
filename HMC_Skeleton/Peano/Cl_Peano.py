import sys
sys.path.append('./Peano')
from PeanoImage import Peano
from InvPeanoImage import PeanoInverse
from PIL import Image
from PIL.Image import fromarray

import numpy as np
import matplotlib.pyplot as plt
import os
from func import *


if __name__ == '__main__':
    
    #############################################@
    # Set the parameters and read the data
    nbIter = 10

    imagename = 'Grey1.png'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    image_path = os.path.join(dir_path,'sources',imagename)
    image = np.array(Image.open(image_path).convert('L'), dtype=float)
    vecteur = Peano(image)
    # The data (simulated)
    Y = vecteur
    N = Y.shape[0]
    K = 16
    
    # Parameters of MM: mean, variance and a priori proba
    meanTabIter  = np.zeros(shape=(nbIter, K))
    sigmaTabIter = np.zeros(shape=(nbIter, K))
    cTabIter     = np.zeros(shape=(nbIter, K, K))
    tTabIter     = np.zeros(shape=(nbIter, K, K))
    ITabIter     = np.zeros(shape=(nbIter, K))

    ##########################################################################
    # Parameters initialization
    iteration = 0
    print('--->iteration=', iteration)
    meanTabIter[iteration, :], sigmaTabIter[iteration, :], cTabIter[iteration, :, :] = InitParam(K, N, Y)
    tTabIter[iteration, :, :], ITabIter[iteration, :] = getProbaMarkov(K, cTabIter[iteration, :, :])

    # Proba computations
    alpha, S = getAlpha(K, N, Y, meanTabIter[iteration, :], sigmaTabIter[iteration, :], ITabIter[iteration, :], tTabIter[iteration, :, :])
    beta     = getBeta(K, N, Y, meanTabIter[iteration, :], sigmaTabIter[iteration, :], ITabIter[iteration, :], tTabIter[iteration, :, :], S)
    gamma    = getGamma(K, N, alpha, beta)

    # MPM classification
    X_MPM = getMPMClassif(N, gamma)

    ##########################################################################
    # EM iterations
    for iteration in range(1, nbIter):
        print('--->iteration=', iteration)
        
        gamma = EM_Iter(iteration, K, N, Y, meanTabIter, sigmaTabIter, cTabIter, tTabIter, ITabIter)
        
        # MPM classification
        X_MPM = getMPMClassif(N, gamma)

    pathToSave = './results/'
    # DrawCurvesParam(nbIter, K, pathToSave, meanTabIter, sigmaTabIter, tTabIter)
    
    image = PeanoInverse(X_MPM)

    # put the pixel of each class to the corresponding grey level (gaussian median)
    for i in range (len(image)):
        for j in range(len(image[i])):
            image[i][j] = meanTabIter[-1][image[i][j]]
    imS = fromarray(np.uint8(image)) 
    imS.save(pathToSave+'imagereconstruite.png')