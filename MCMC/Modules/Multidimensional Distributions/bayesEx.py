import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz

#import seaborn as sns
plt.style.use('ggplot')

def T(theta, prior='triangle'):
    # Prior
    if prior == 'flat':
        T = 1.0
    elif prior == 'triangle':
        # triangular
        if theta < 0.5:
            T = 4*theta
        else:
            T = 4*(1-theta)
    return T

def plotPostDist(N, nH):

    # get prior
    thetav   = np.linspace(1e-6, 1.-1e-6, 101) # protect from 0 and 1 where log terms go to inf
    priorD   = Tv(thetav, 'flat')

    # get unnormalized posterior
    LogPostD = nH * np.log(thetav) + (N - nH) * np.log(1.-thetav) + np.log(priorD)
    postD    = np.exp(LogPostD)

    # normalize posterior
    normFac = trapz(postD, thetav)
    postD   = postD/normFac

    # plot
    plt.plot(thetav, priorD, label='prior')
    plt.plot(thetav, postD, label='post')

    plt.title('# heads/total = ' + str(nH) + '/' + str(N), fontsize=20)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$p$')
    plt.legend()
    plt.tight_layout()
    plt.show()	

    return
	
Tv = np.vectorize(T)  # this lets me send scalar or vector theta to T()
plotPostDist(10, 7)
