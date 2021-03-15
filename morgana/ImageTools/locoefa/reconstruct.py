import numpy as np

def reconstruct_contour(mode, tp, rec_type='EFA', first_mode=0, last_mode=2):
    N_modes = len(mode.locoL.values)

    # timepoint -= cell->contour[cellnumber][cell->contourlength[cellnumber]].t*cell->locoefa[cellnumber][1].tau/(2.*M_PI)
    if mode.r[1]<0.:
        tp = -tp

    x = [ 0. for i in tp ]
    y = [ 0. for i in tp ]
    if rec_type=='EFA':
        if first_mode == 0:
            x += mode.alpha[0]
            y += mode.gamma[0]

        for p in range(np.max([1,first_mode]), np.min([last_mode,N_modes+1])):
            x += mode.alpha[p] * np.cos(2.*np.pi*p*tp) + mode.beta[p] * np.sin(2.*np.pi*p*tp)
            y += mode.gamma[p] * np.cos(2.*np.pi*p*tp) + mode.delta[p] * np.sin(2.*np.pi*p*tp)

    elif rec_type=='LOCOEFA':
        if first_mode == 0:
            x += mode.locooffseta[0]
            y += mode.locooffsetc[0]
            # L=0
            x += mode.locolambdaplus[0] * ( np.cos(mode.locozetaplus[0]) * np.cos(2.*np.pi*2.*tp) - np.sin(mode.locozetaplus[0]) * np.sin(2.*np.pi*2.*tp) )
            y += mode.locolambdaplus[0] * ( np.sin(mode.locozetaplus[0]) * np.cos(2.*np.pi*2.*tp) + np.cos(mode.locozetaplus[0]) * np.sin(2.*np.pi*2.*tp) )

        if first_mode <= 1:
            # L=1
            x += mode.locolambdaplus[1] * ( np.cos(mode.locozetaplus[1]) * np.cos(2.*np.pi*tp) - np.sin(mode.locozetaplus[1]) * np.sin(2.*np.pi*tp) )
            y += mode.locolambdaplus[1] * ( np.sin(mode.locozetaplus[1]) * np.cos(2.*np.pi*tp) + np.cos(mode.locozetaplus[1]) * np.sin(2.*np.pi*tp) )

        # L=2...N,+
        for p in range(np.max([2,first_mode]),np.min([last_mode+1,N_modes])):
            x += mode.locolambdaplus[p] * ( np.cos(mode.locozetaplus[p]) * np.cos(2.*np.pi*(p+1)*tp) - np.sin(mode.locozetaplus[p]) * np.sin(2.*np.pi*(p+1)*tp) )
            y += mode.locolambdaplus[p] * ( np.sin(mode.locozetaplus[p]) * np.cos(2.*np.pi*(p+1)*tp) + np.cos(mode.locozetaplus[p]) * np.sin(2.*np.pi*(p+1)*tp) )

        # L=2..N,-
        for p in range(np.max([2,first_mode]),np.min([last_mode+1,N_modes+2])):
            x += mode.locolambdaminus[p] * ( np.cos(mode.locozetaminus[p]) * np.cos(2.*np.pi*(p-1)*tp) - np.sin(mode.locozetaminus[p]) * np.sin(2.*np.pi*(p-1)*tp) )
            y -= mode.locolambdaminus[p] * ( np.sin(mode.locozetaminus[p]) * np.cos(2.*np.pi*(p-1)*tp) + np.cos(mode.locozetaminus[p]) * np.sin(2.*np.pi*(p-1)*tp) )

    return x, y
