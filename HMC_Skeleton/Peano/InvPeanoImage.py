import numpy as np
from PIL.Image import *
import matplotlib.pyplot as plt
from hilbertcurve import HilbertCurve
from PeanoImage import getPowerOfTwo

def PeanoInverse(vecteur):

    # Prepare the inverse scan
    N = np.shape(vecteur)[0]
    L = int(math.sqrt(N))
    C = L
    p = getPowerOfTwo(L)
    if (L*C != N):
        print('pb !!!')
        exit(1)
    # print(N, L, C, p)
    # input('pause')

    # here we are!
    image = np.zeros(shape=(L, C), dtype=int)
    hilbert_curve = HilbertCurve(p, 2)
    for ii in range(N):
        l, c = hilbert_curve.coordinates_from_distance(ii)
        image[l, c] = vecteur[ii]

    return image

if __name__ == '__main__':

    vectname = './results/image_peano_EM_MPM.out'
    vecteur = np.loadtxt(vectname)
    image = PeanoInverse(vecteur)

    imS = fromarray(np.uint8(image))
    imS.save('imagereconstruite.png')


