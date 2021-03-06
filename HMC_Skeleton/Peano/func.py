from random import gammavariate
import scipy.linalg as la
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

tolerance = 1e-8
fontS  = 12     # font size
colors = ['r', 'g', 'b', 'y', 'm']

def getSteadyState(P):

    theEigenvalues, leftEigenvectors = la.eig(P, right=False, left=True)
    theEigenvalues   = theEigenvalues.real
    leftEigenvectors = leftEigenvectors.real

    mask = abs(theEigenvalues - 1) < tolerance
    theEigenvalues   = theEigenvalues[mask]
    leftEigenvectors = leftEigenvectors[:, mask]
    # leftEigenvectors[leftEigenvectors < tolerance] = 0

    attractorDistributions = leftEigenvectors / leftEigenvectors.sum(axis=0, keepdims=True)
    attractorDistributions = attractorDistributions.T
    theSteadyStates = np.sum(attractorDistributions, axis=0)

    return theSteadyStates

###############################################################################

def getAlpha(K, N, Y, mu, sigma, I, t):
    # forward computing
    alpha = np.zeros(shape=(N, K))
    S     = np.zeros(shape=(N))
    
    np1=0
    for k in range(K):
        alpha[np1, k] = I[k] * norm.pdf(Y[np1], loc=mu[k], scale=sigma[k])
    alpha[np1, :] /= np.sum(alpha[np1, :]) 

    for np1 in range(1, N):
        for k in range(K):
            for l in range(K):
                alpha[np1, k] += t[l, k] * alpha[np1-1, l]
            alpha[np1, k] *= norm.pdf(Y[np1], loc=mu[k], scale=sigma[k])
        S[np1] = np.sum(alpha[np1, :])
        alpha[np1, :] /= S[np1]

    return alpha, S

def getBeta(K, N, Y, mu, sigma, I, t, S):
    # backward computing
    beta = np.zeros(shape=(N, K))

    np1 = N-1 
    for k in range(K):
        beta[np1,k] = 1

    for np1 in range(N-2,-1,-1): 
        for k in range(K):
            for l in range(K):
                beta[np1, k] += t[k,l] * beta[np1+1, l] * norm.pdf(Y[np1+1], loc=mu[l], scale=sigma[l])
        beta[np1,:] /= S[np1+1]

    return beta

def getGamma(K, N, alpha, beta):
    # gamma computing (marginal a posterori proba)
    gamma = np.zeros(shape=(N, K))
    
    for n in range(N):
        gamma[n, :] = alpha[n, :] * beta[n, :]
        if np.sum(gamma[n, :]) < 1-tolerance:
            print(np.sum(gamma[n, :]))
            print(gamma[n, :])
            # input('pb pause')
    
    return gamma

def getMPMClassif(N, gamma):
    # MPM classification
    X_MPM = np.zeros(shape=(N))
    
    for n in range(N):
        X_MPM[n] = np.argmax(gamma[n, :])
        # print('X_MPM[n]=', X_MPM[n])
        # input('classif')
        
    return X_MPM

def getMAPClassif(K, N, Y, mu, sigma, I, t):
    
    # MAP classification (Viterbi algo)
    X_MAP = np.zeros(shape=(N))
    delta = np.zeros(shape=(N, K))
    psi = np.zeros(shape=(N, K))
    
    for k in range(K):
        delta[0,k] = np.log(I[k]) + np.log(norm.pdf(Y[0], loc=mu[k], scale=sigma[k]))
    
    for n in range(1,N):
        for k in range(K):
            transi = np.zeros(shape=(K))
            for j in range(K):
                transi = delta[n-1,:] + np.log(t[:,k])
            delta[n,k] = np.log(norm.pdf(Y[n], loc=mu[k], scale=sigma[k])) + np.max(transi)
            psi[n,k] = np.argmax(transi)

    X_MAP[N-1] = np.argmax(delta[N-1,:])
    for n in range(N-2,-1,-1):
        X_MAP[n] = psi[n+1, int(X_MAP[n+1])] 

    return X_MAP

def getConfMat(K, N, X, X_MPM):
    # error rate computation
    ConfMatrix = np.zeros(shape=(K,K))
    ERbyClass  = np.zeros(shape=(K))
    ERGlobal   = 0.
    
    for n in range(N):
        ConfMatrix[int(X[n]), int(X_MPM[n])] += 1.
        # print('X[n]=', X[n], ', X_MPM[n]=', X_MPM[n])
        # input('pause')
        if X[n]!= X_MPM[n]: ERGlobal += 1.
    
    #ConfMatrix[] /= N
    ERGlobal /= N

    for k in range(K):
        ERbyClass[k] = 1. - ConfMatrix[k, k] / np.sum(ConfMatrix, axis=1)[k]
    
    return ConfMatrix, ERGlobal, ERbyClass


def getProbaMarkov(K, JProba):
    # get transition matric and stationary dutribution
    
    IProba = np.sum(JProba, axis=1).T
    TProba = np.zeros(shape = np.shape(JProba))
    
    for r in range(K):
        TProba[r, :] = JProba[r, :] / IProba[r]
    
    return TProba, IProba


def InitParam(K, N, Y):
    
    # init apram for EM
    mu    = np.zeros(shape=(K))
    sigma = np.zeros(shape=(K))
    c     = np.zeros(shape=(K, K))

    # compute the min, max, mean and var of the image
    minY  = int(np.min(Y))
    maxY  = int(np.max(Y))
    meanY = np.mean(Y) # comment: not necessary
    varY  = np.var(Y)

    # set the initial values for all the parameters
    for k in range(K):
        
        # Init Gaussian
        sigma[k] = np.sqrt(varY / 2.)
        mu[k]    = minY + (maxY-minY)/(2.*K) + k * (maxY-minY)/K
        
        # init stationarry joint Markov matrix
        c[k, k]  = 0.9/K
        for l in range(k+1, K):
            c[k, l] = 0.1/(K*(K-1))
            c[l, k] = c[k, l]
    
    return mu, sigma, c

def getCtilde(K, N, Y, alpha, beta, tTabIter, meanTabIter, sigmaTabIter, S):
    ctilde = np.zeros(shape=(N, K, K))
    
    # calculating ctilde
    for n in range(N-1):
        for xn in range(K):
            for xnp1 in range(K):
                ctilde[n, xn, xnp1] = tTabIter[xn, xnp1] * norm.pdf(Y[n+1], loc=meanTabIter[xnp1], scale=sigmaTabIter[xnp1]) * alpha[n, xn] * beta[n+1, xnp1] / S[n+1]
    # print('ctilde=', ctilde[n, :, :])
    # input('pause ctilde')
    return ctilde

def UpdateParameters(K, N, Y, alpha, beta, gamma, ctilde):

    mean  = np.zeros(shape=(K))
    sigma = np.zeros(shape=(K))
    c     = np.zeros(shape=(K, K))
    t     = np.zeros(shape=(K, K))
    I     = np.zeros(shape=(K))

    # TO DO

    # print(f'Alpha : {alpha}, Alpha shape : {alpha.shape}')
    # print(f'Beta : {beta}, Beta shape : {beta.shape}')
    # print(f'gamma : {gamma}, gamma shape : {gamma.shape}')
    # print(f'Ctilde : {ctilde}, Ctilde shape : {ctilde.shape}')

    for k in range(K):
        sumk = 0
        for n in range(N): # get mean 
            mean[k] += gamma[n,k]*Y[n]
            sumk += gamma[n,k]
        mean[k] /= sumk
        I[k] = sumk/N

        for l in range(K): # get c
            sumc = 0
            for n in range(N-1):
                sumc += ctilde[n,k,l]
            t[k,l] = sumc/(sumk - gamma[n,k])
            c[k,l] = t[k,l] * I[k]

    for k in range(K) :
        sumk = 0 
        for n in range(N): # get sigma
            sigma[k] += gamma[n,k]*(Y[n]-mean[k])**2
            sumk += gamma[n,k]
        sigma[k] = np.sqrt(sigma[k]/sumk)

    

    # print('c=', c)
    # print('t=', t)
    # print('I=', I)
    return mean, sigma, c, t, I


def EM_Iter(iteration, K, N, Y, meanTabIter, sigmaTabIter, cTabIter, tTabIter, ITabIter):

    # Proba computations
    alpha, S = getAlpha(K, N, Y, meanTabIter[iteration-1, :], sigmaTabIter[iteration-1, :], ITabIter[iteration-1, :], tTabIter[iteration-1, :, :])
    beta     = getBeta (K, N, Y, meanTabIter[iteration-1, :], sigmaTabIter[iteration-1, :], ITabIter[iteration-1, :], tTabIter[iteration-1, :, :], S)
    gamma    = getGamma(K, N, alpha, beta)
    ctilde   = getCtilde(K, N, Y, alpha, beta, tTabIter[iteration-1, :], meanTabIter[iteration-1, :], sigmaTabIter[iteration-1, :], S)
    
    meanTabIter[iteration, :], sigmaTabIter[iteration, :], cTabIter[iteration, :, :], tTabIter[iteration, :, :], ITabIter[iteration, :] = UpdateParameters(K, N, Y, alpha, beta, gamma, ctilde)
    
    return gamma

def DrawCurvesParam(nbIter, K, pathToSave, meanTabIter, sigmaTabIter, tTabIter):

    fig, [ax1, ax2, ax3] = plt.subplots(nrows=3, ncols=1)
    for k in range(K):
        ax1.plot(range(nbIter), meanTabIter[:, k], lw=1, alpha=0.9, color=colors[k], label='class ' + str(k))
        ax2.plot(range(nbIter), sigmaTabIter [:, k], lw=1, alpha=0.9, color=colors[k], label='class ' + str(k))
        ax3.plot(range(nbIter), tTabIter  [:, k, k], lw=1, alpha=0.9, color=colors[k], label='class ' + str(k))
    
    ax1.set_ylabel('mu',       fontsize=fontS)
    ax2.set_ylabel('sigma**2', fontsize=fontS)
    ax3.set_ylabel('t(k,k)',   fontsize=fontS)
    ax1.legend()

    # figure saving
    plt.xlabel('EM iterations', fontsize=fontS)
    plt.savefig(pathToSave + 'EvolParam.png', bbox_inches='tight', dpi=150)

def DrawCurvesError(nbIter, K, pathToSave, MeanErrorRateTabbyClass, MeanErrorRateTab):
    
    fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
    for k in range(K):
        ax1.plot(range(nbIter), MeanErrorRateTabbyClass[:, k], lw=1, alpha=0.9, color=colors[k], label='class ' + str(k))
    ax2.plot(range(nbIter), MeanErrorRateTab, lw=1, alpha=0.9, color='k', label='global')
    
    ax1.set_ylabel('% error', fontsize=fontS)
    ax2.set_ylabel('% error', fontsize=fontS)
    ax1.legend()

    # figure saving
    plt.xlabel('EM iterations', fontsize=fontS)
    plt.savefig(pathToSave + '_EvolError.png', bbox_inches='tight', dpi=150)