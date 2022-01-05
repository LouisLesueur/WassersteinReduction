import torch


def cost(K,couplings):
    '''
    Compute the cost of a solution in the original dimension (d)

    - K (shape [k,n,T]) the cost matrix
    - couplings (shape [k, n, T]) the couplings
    '''

    return torch.multiply(couplings,K).sum()

def gen_K(X, support, eps):
    '''
    Generate the Cost matrix for Sinkhorn algorithm

    - X (shape [k,T,d]): supports of the distributions (point clouds)
    - support (shape [n]): support of the output barycenter
    - eps: regularization parameter
    '''
    
    n = support.shape[0]
    k,T,d = X.shape

    C = torch.zeros((k, n, T))
    for s in range(k):
        C[s] = torch.cdist(support, X[s])
    return torch.exp(-(C**2)/eps)
