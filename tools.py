import torch


def cost(K,couplings):
    '''
    Compute the cost of a solution in the original dimension (d)

    - K (shape [k,n,T]) the cost matrix
    - couplings (shape [k, n, T]) the couplings
    '''

    return (couplings*K).sum()

def gen_K(X, support, eps):
    
    n = support.shape[0]
    k,T,d = X.shape

    C = torch.zeros((k, n, T))
    for s in range(k):
        C[s] = torch.cdist(support, X[s])
    return torch.exp(-(C**2)/eps)
