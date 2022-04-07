import numpy as np

def mylog(mat, log=True, del_diag=True, del_subdiag=True):
    mat = np.where(np.isnan(mat) == True, 0, mat)
    matmin = mat.ravel()[np.nonzero(mat.ravel())].min()
    logmat = np.where(mat == 0, matmin / 2, mat)
    
    if log:
        logmat = np.log2(logmat)
    
    if del_diag:
        np.fill_diagonal(logmat, 0.) #zero main diagonal
    
    if del_subdiag:
        for i in range(logmat.shape[0]):
            if (i == 0):
                logmat[i, i+1] = 0
            elif (i == (logmat.shape[0] - 1)):
                logmat[i, i-1] = 0
            else:
                logmat[i, i+1] = 0
                logmat[i, i-1] = 0
    return logmat