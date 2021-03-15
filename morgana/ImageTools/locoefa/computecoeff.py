from skimage import transform
import numpy as np
import pandas as pd
import os

import sys
sys.path.append(os.path.join('..','..'))

from morgana.ImageTools.locoefa import initialize

def compute_EFA(contour, mode, DEBUG=False):
    if (DEBUG):
        print('Computing EFA coefficients...')
    N_points = len(contour.x)
    N_modes = len(mode.alpha)-2

    # Eq. 5: DeltaX, DeltaY, DeltaT, T
    contour.deltax[1:] = np.diff(contour.x)
    contour.deltay[1:] = np.diff(contour.y)
    contour.deltat[1:] = np.sqrt(contour.deltax[1:]**2+contour.deltay[1:]**2)
    contour.t = np.cumsum(contour.deltat)
    T = contour.t.values[-1]

    # extract info as numpy arrays for fast computations
    sumdeltaxj = contour.sumdeltaxj.values
    sumdeltayj = contour.sumdeltayj.values
    deltax = contour.deltax.values
    deltay = contour.deltay.values
    deltat = contour.deltat.values
    t = contour.t.values
    xi = contour.xi.values
    epsilon = contour.epsilon.values
    
    # Eq. 7: : sumDeltaxj, sumDeltayj, xi, epsilon
    for i in range(2,N_points):
        contour.sumdeltaxj[i] = contour.sumdeltaxj[i-1]+contour.deltax[i-1]
        contour.sumdeltayj[i] = contour.sumdeltayj[i-1]+contour.deltay[i-1]
        contour.xi[i] = contour.sumdeltaxj[i]-contour.deltax[i]/contour.deltat[i]*contour.t[i-1]
        contour.epsilon[i] = contour.sumdeltayj[i]-contour.deltay[i]/contour.deltat[i]*contour.t[i-1]

#    # Eq. 7: : sumDeltaxj, sumDeltayj, xi, epsilon ### does not work.... why???
#    contour.sumdeltaxj[2:] = sumdeltaxj[1:-1] + deltax[1:-1]
#    contour.sumdeltayj[2:] = sumdeltayj[1:-1] + deltay[1:-1]
#    contour.xi[2:] = sumdeltaxj[2:] - deltax[2:]/deltat[2:]*t[1:-1]
#    contour.epsilon[2:] = sumdeltayj[2:] - deltay[2:]/deltat[2:]*t[1:-1]

    #Equation 7: alpha0, gamma0
    mode.alpha[0] = contour.x[0]
    mode.gamma[0] = contour.y[0]
    mode.alpha[0] += np.sum((deltax[1:]/(2.*deltat[1:])*(t[1:]**2-t[:-1]**2)+xi[1:]*(t[1:]-t[:-1]))/T)
    mode.gamma[0] += np.sum((deltay[1:]/(2.*deltat[1:])*(t[1:]**2-t[:-1]**2)+epsilon[1:]*(t[1:]-t[:-1]))/T)

    for j in range(1,N_modes):
        mode.alpha[j] = np.sum( deltax[1:]/deltat[1:]*(np.cos(2.*j*np.pi*t[1:]/T)-np.cos(2*j*np.pi*t[:-1]/T)) ) * T/(2.*j**2*np.pi**2)
        mode.beta[j] = np.sum( deltax[1:]/deltat[1:]*(np.sin(2.*j*np.pi*t[1:]/T)-np.sin(2*j*np.pi*t[:-1]/T)) ) * T/(2.*j**2*np.pi**2)
        mode.gamma[j] = np.sum( deltay[1:]/deltat[1:]*(np.cos(2.*j*np.pi*t[1:]/T)-np.cos(2*j*np.pi*t[:-1]/T)) ) * T/(2.*j**2*np.pi**2)
        mode.delta[j] = np.sum( deltay[1:]/deltat[1:]*(np.sin(2.*j*np.pi*t[1:]/T)-np.sin(2*j*np.pi*t[:-1]/T)) ) * T/(2.*j**2*np.pi**2)
            
 
    if(DEBUG):
        print("\n\nEFA coefficients:\n=================\n\n")
        for j in range(N_modes):
            print("mode %d:\n"%(j))
            print("(%f\t%f)\n"%(mode.alpha[j],mode.beta[j]))
            print("(%f\t%f)\n\n"%(mode.gamma[j],mode.delta[j]))

    return mode, contour

def compute_LOCOEFA(mode, DEBUG=False):
    if (DEBUG):
        print('Computing LOCO-EFA coefficients...')
    N_modes = len(mode.alpha)-2

    # Equation 14: tau1
    mode.tau[1] = 0.5*np.arctan2(2.*(mode.alpha[1]*mode.beta[1]+mode.gamma[1]*mode.delta[1]), mode.alpha[1]**2+mode.gamma[1]**2-mode.beta[1]**2-mode.delta[1]**2) 

    # Below eq. 15: alpha1prime, gamma1prime
    mode.alphaprime[1] = mode.alpha[1] * np.cos(mode.tau[1]) + mode.beta[1] * np.sin(mode.tau[1])
    mode.gammaprime[1] = mode.gamma[1] * np.cos(mode.tau[1]) + mode.delta[1] * np.sin(mode.tau[1])

    # Equation 16: rho
    mode.rho[1] = np.arctan2(mode.gammaprime[1], mode.alphaprime[1])

    # Equation 17: tau1
    if(mode.rho[1]<0.):
        mode.tau[1] += np.pi

    #Equation 18: alphastar, betastar, gammastar, deltastar
    mode.alphastar[1:N_modes+1] = mode.alpha[1:N_modes+1]*np.cos(np.arange(1,N_modes+1)*mode.tau[1])+mode.beta[1:N_modes+1]*np.sin(np.arange(1,N_modes+1)*mode.tau[1])
    mode.betastar[1:N_modes+1] = -mode.alpha[1:N_modes+1]*np.sin(np.arange(1,N_modes+1)*mode.tau[1])+mode.beta[1:N_modes+1]*np.cos(np.arange(1,N_modes+1)*mode.tau[1])
    mode.gammastar[1:N_modes+1] = mode.gamma[1:N_modes+1]*np.cos(np.arange(1,N_modes+1)*mode.tau[1])+mode.delta[1:N_modes+1]*np.sin(np.arange(1,N_modes+1)*mode.tau[1])
    mode.deltastar[1:N_modes+1] = -mode.gamma[1:N_modes+1]*np.sin(np.arange(1,N_modes+1)*mode.tau[1])+mode.delta[1:N_modes+1]*np.cos(np.arange(1,N_modes+1)*mode.tau[1])

    # Equation 9: r
    mode.r[1] = mode.alphastar[1]*mode.deltastar[1]-mode.betastar[1]*mode.gammastar[1]

    # Equation 19: betastar, deltastar
    if(mode.r[1]<0.):
        mode.betastar[1:N_modes+1] = -mode.betastar[1:N_modes+1]
        mode.deltastar[1:N_modes+1] = -mode.deltastar[1:N_modes+1]

    # Equation 20: a, b, c, d
    mode.a[0] = mode.alpha[0]
    mode.c[0] = mode.gamma[0]

    mode.a[1:N_modes+1] = mode.alphastar[1:N_modes+1]
    mode.b[1:N_modes+1] = mode.betastar[1:N_modes+1]
    mode.c[1:N_modes+1] = mode.gammastar[1:N_modes+1]
    mode.d[1:N_modes+1] = mode.deltastar[1:N_modes+1]

    if(DEBUG):
        print("\n\nmodified EFA coefficients:\n==========================\n\n")
        for i in range(N_modes):
            print("mode %d:\n"%i)
            print("(%f\t%f)\n"%(mode.a[i],mode.b[i]))
            print("(%f\t%f)\n\n"%(mode.c[i],mode.d[i]))
  
    if(DEBUG):
        print("\n\nLambda matrices:\n================\n\n")

    ## this can all be optimized, but probably not worth it
    for i in range(1,N_modes+1):
        # Equation 26: phi
        mode.phi[i] = 0.5*np.arctan2( 2.*(mode.a[i]*mode.b[i]+mode.c[i]*mode.d[i]), mode.a[i]**2+mode.c[i]**2-mode.b[i]**2-mode.d[i]**2 )

        # Below eq. 27: aprime, bprime, cprime, dprime
        mode.aprime[i] = mode.a[i] * np.cos(mode.phi[i]) + mode.b[i] * np.sin(mode.phi[i])
        mode.bprime[i] = -mode.a[i] * np.sin(mode.phi[i]) + mode.b[i] * np.cos(mode.phi[i])
        mode.cprime[i] = mode.c[i] * np.cos(mode.phi[i]) + mode.d[i] * np.sin(mode.phi[i])
        mode.dprime[i] = -mode.c[i] * np.sin(mode.phi[i]) + mode.d[i] * np.cos(mode.phi[i])

        # Equation 27: theta
        mode.theta[i] = np.arctan2( mode.cprime[i], mode.aprime[i] )

        # Equation 25: Lambda
        mode.lambda1[i] = np.cos(mode.theta[i]) * mode.aprime[i] + np.sin(mode.theta[i]) * mode.cprime[i]
        mode.lambda12[i] = np.cos(mode.theta[i]) * mode.bprime[i] + np.sin(mode.theta[i]) * mode.dprime[i]
        mode.lambda21[i] = -np.sin(mode.theta[i]) * mode.aprime[i] + np.cos(mode.theta[i]) * mode.cprime[i]
        mode.lambda2[i] = -np.sin(mode.theta[i]) * mode.bprime[i] + np.cos(mode.theta[i]) * mode.dprime[i]

        # Equation 32: lambdaplus, lambdaminus 
        mode.lambdaplus[i] = (mode.lambda1[i]+mode.lambda2[i])/2.
        mode.lambdaminus[i] = (mode.lambda1[i]-mode.lambda2[i])/2.

        # Below eq. 37: zetaplus, zetaminus
        mode.zetaplus[i] = mode.theta[i]-mode.phi[i]
        mode.zetaminus[i] = -mode.theta[i]-mode.phi[i]

    # Below eq. 39: A0
    mode.locooffseta[0] = mode.a[0]
    mode.locooffsetc[0] = mode.c[0]

    if (DEBUG):
        print("\n\noffset:\n===============\n\n")
        print("LOCO-EFA A0 offset:\ta=%f\tc=%f\n"%(mode.locooffseta[0],mode.locooffsetc[0]))

    # Below eq. 41: A+(l=0)
    mode.locolambdaplus[0] = mode.lambdaplus[2]
    mode.locozetaplus[0] = mode.zetaplus[2]

    # Below eq. 41: A+(l=1)
    mode.locolambdaplus[1] = mode.lambdaplus[1]
    mode.locozetaplus[1] = mode.zetaplus[1]

    # Below eq. 41: A+(l>1)
    for i in range(2,N_modes):
        mode.locolambdaplus[i] = mode.lambdaplus[i+1]
        mode.locozetaplus[i] = mode.zetaplus[i+1]

    # Below eq. 41: A-(l>0)
    for i in range(2,N_modes+2):
        mode.locolambdaminus[i]=mode.lambdaminus[i-1]
        mode.locozetaminus[i] = mode.zetaminus[i-1]

    if (DEBUG):
        print("\n\nLn quadruplets:\n===============\n\n")
        for i in range(N_modes+2):
            print("LOCO-EFA mode %d:\tlambdaplus=%f\tlambdaminus=%f\tzetaplus=%ftzetaminus=%f\n"%(i,mode.locolambdaplus[i],mode.locolambdaminus[i],mode.locozetaplus[i],mode.locozetaminus[i]))

    # Equation 38: Lambda*Zeta
    mode.locoaplus = mode.locolambdaplus*np.cos(mode.locozetaplus)
    mode.locobplus = -mode.locolambdaplus*np.sin(mode.locozetaplus)
    mode.lococplus = mode.locolambdaplus*np.sin(mode.locozetaplus)
    mode.locodplus = mode.locolambdaplus*np.cos(mode.locozetaplus)
    mode.locoaminus = mode.locolambdaminus*np.cos(mode.locozetaminus)
    mode.locobminus = -mode.locolambdaminus*np.sin(mode.locozetaminus)
    mode.lococminus = -mode.locolambdaminus*np.sin(mode.locozetaminus)
    mode.locodminus = -mode.locolambdaminus*np.cos(mode.locozetaminus)

    if(DEBUG):
        print("\n\nLOCO coefficients:\n==================\n\n")
        for i in range(N_modes+2):
            print("mode %d, Aplus:\n"%i)
            print("(%f\t%f)\n"%(mode.locoaplus[i],mode.locobplus[i]))
            print("(%f\t%f)\n"%(mode.lococplus[i],mode.locodplus[i]))
            print("mode %d, Aminus:\n"%i)
            print("(%f\t%f)\n"%(mode.locoaminus[i],mode.locobminus[i]))
            print("(%f\t%f)\n"%(mode.lococminus[i],mode.locodminus[i]))

    # Equation 47: L
    mode.locoL[1:]=np.sqrt(mode.locolambdaplus[1:]*mode.locolambdaplus[1:]+mode.locolambdaminus[1:]*mode.locolambdaminus[1:]+2.*mode.locolambdaplus[1:]*mode.locolambdaminus[1:]*np.cos(mode.locozetaplus[1:]-mode.locozetaminus[1:]-2.*mode.locozetaplus[1]))

    if(DEBUG):
        print("\nLn scalar:\n==========\n")
        for i in range(N_modes+2):
            print("LOCO-EFA mode %d:\tLn=%f"%(i,mode.locoL[i]))

    return mode

def compute_LOCOEFA_Lcoeff(mask, down_shape=1., N_modes=50, DEBUG=False):
    
    ma_down = transform.resize(mask.astype(float), (int(mask.shape[0]*down_shape),int(mask.shape[1]*down_shape)), order=0, preserve_range=True)
    ma_down = np.rint(ma_down).astype(np.uint8)

    contour, mode = initialize.get_edge_points(ma_down)
    
    mode, contour = compute_EFA(contour, mode, DEBUG=DEBUG)
    mode = compute_LOCOEFA(mode, DEBUG=DEBUG)

    mode_save = pd.DataFrame({
        'alpha': mode.alpha,
        'beta': mode.beta,
        'gamma': mode.gamma,
        'delta': mode.delta,
        'locoefa_coeff': mode.locoL,
        'locooffseta': mode.locooffseta,
        'locooffsetc': mode.locooffsetc,
        'locolambdaplus': mode.locolambdaplus,
        'locolambdaminus': mode.locolambdaminus,
        'locozetaplus': mode.locozetaplus,
        'locozetaminus': mode.locozetaminus,
    })

    return mode_save

if __name__ == '__main__':
    import tqdm, os
    import pandas as pd
    # suppress warning for assignment
    pd.options.mode.chained_assignment = None
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.ndimage.morphology
    import skimage.io

    import ImageTools.locoefa.reconstruct
    import ImageTools.locoefa.initialize

    ################################################################################
    ### different ways of importing data

     ### read contour points
    fname = os.path.join('example','celloutline.csv')
    contour, mode = ImageTools.locoefa.initialize.read_example_data(fname)

    # ### create mask
    # mask = np.zeros((500,500))
    # for i, c in contour.iterrows():
    #     mask[int(c.x),int(c.y)] = 1
    # mask = scipy.ndimage.morphology.binary_fill_holes(mask)
    # skimage.io.imsave(os.path.join('example','mask.tif'),mask.astype(np.uint8))

#    ## extract contour points from a mask
#    fname = os.path.join('example','mask.tif')
# #   fname = os.path.join('..','..','Images/objectsparser_testData/splitObjects','result_segmentation','objectsparser_testData1_cropped00_finalMask.tif')
#    fname = os.path.join('..','..','Images\\objectsparser_testData\\splitObjects','result_segmentation','objectsparser_testData1_cropped00_finalMask.tif')
#    mask = skimage.io.imread(fname)
#    contour, mode = ImageTools.locoefa.initialize.get_edge_points(mask)

    ################################################################################

    mode, contour = compute_EFA(contour, mode, DEBUG=True)
    mode = compute_LOCOEFA(mode, DEBUG=True)

     ### optional: save/load from disk
    contour.to_json(os.path.join('example','contour.json'))
    mode.to_json(os.path.join('example','mode.json'))
    contour = pd.read_json(os.path.join('example','contour.json'))
    mode = pd.read_json(os.path.join('example','mode.json'))

    ### compute and make plots
    tp = np.linspace(0,1,100)
    fig, ax = plt.subplots(5,5,figsize=(8,8))
    ax = ax.flatten()
    max_mode=25
    for mm in tqdm.trange(max_mode, desc = 'Max_modes'):

        x_loco, y_loco = ImageTools.locoefa.reconstruct.reconstruct_contour(mode,tp=tp,rec_type='LOCOEFA',first_mode=0,last_mode=mm)
        x_efa, y_efa = ImageTools.locoefa.reconstruct.reconstruct_contour(mode,tp=tp,rec_type='EFA',first_mode=0,last_mode=mm)

        # ax[mm].set_xlim(-0,400)
        # ax[mm].set_ylim(-25,375)
        ax[mm].plot(contour.x,contour.y,'-b')
        ax[mm].plot(x_loco,y_loco,'-r')
        ax[mm].plot(x_efa,y_efa,'-',color='orange')
        ax[mm].set_title('Mode %d'%mm, fontsize=8)
        ax[mm].set_xticks([])
        ax[mm].set_yticks([])
    plt.show()

