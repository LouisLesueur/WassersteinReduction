import torch
from tqdm import tqdm
from tools import gen_K

class Sinkhorn:

    def __init__(self, niter=3000, eps=0.01, device="cuda"):
        self.niter = niter
        self.device = device
        self.eps = eps

    def solve(self, lambdas, X, a, supp):
        '''
        Computes the Wasserstein barycenter W_2 between the mus

        - K (shape [k,n,T]): cost matrix
        - lambdas (shape k): weights
        - X (shape [k, T, d]): points clouds
        - a (shape [k, T]): weights
        - supp (shape [n, d]): barycenter support
        '''

        lambdas = lambdas.to(self.device)
        X = X.to(self.device)
        a = a.to(self.device)
        supp = supp.to(self.device)

        # Save normalization factors
        factors = a.sum(axis=1)
        a = torch.div(a.T, factors).T

        k,T,d = X.shape
        n = supp.shape[0]

        K = gen_K(X, supp, self.eps).to(self.device)

        couplings = torch.zeros((k,n,T))
        v = torch.ones((k,T)).to(self.device)
        u = torch.ones((k,n)).to(self.device)


        for l in tqdm(range(self.niter)):

            for s in range(k):
                v[s] = a[s] / (K[s].T@u[s])

            b = torch.zeros(n).to(self.device)
            for s in range(k):
                b = b + lambdas[s]*torch.log(K[s]@v[s])
            b = torch.exp(b)

            for s in range(k):
                u[s] = b / (K[s]@v[s])

        for s in range(k):
            couplings[s] = torch.diag(u[s])@K[s]@torch.diag(v[s])

        #Cancel normalization
        norm_fact = (lambdas*factors).sum()
        b = b*norm_fact


        return b, couplings, K
