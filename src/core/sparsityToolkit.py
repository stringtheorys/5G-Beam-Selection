import numpy as np

def to_sparse(weights,dtype=np.float32):
    
    indexed = list(np.ndenumerate(weights))
    
    spM = list()
    def isZero(elem):
        ind , val = elem
        if val != 0:
            spM.append(np.array([float(ind[0]),float(ind[1]),val]))
    
    sparsify = np.vectorize(isZero,signature = '(0)->()')
    sparsify(indexed)
    return np.array(spM,dtype)

def to_dense(sparseM, dtype=np.float32):
    
    x = int(np.max(sparseM[:,1])) + 1
    y = int(sparseM[-1][0]) + 1
    
    base = np.zeros((x,y),dtype)
    
    def setter(elem):
        xInd, yInd, val = elem
        base[int(xInd)][int(yInd)] = val
    arraySetter = np.vectorize(setter,signature = '(0)->()')
    arraySetter(sparseM)
    
    return base