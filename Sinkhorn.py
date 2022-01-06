import torch
from tqdm import tqdm

class Sinkhorn:

    def __init__(self, thresh=1e-10, eps=0.01, device="cuda"):
        self.thresh = thresh
        self.device = device
        self.eps = eps
        self.C = None

    def solve(self, lambdas, a, supp, compute_couplings=True):
        '''
        Computes the Wasserstein barycenter W_2 between the mus

        - lambdas (shape k): weights
        - a (shape [k, n]): weights
        - supp (shape [n, d]): barycenter support
        '''

        lambdas = lambdas.to(self.device)
        a = a.to(self.device)
        supp = supp.to(self.device)


        k = a.shape[0]
        n, d = supp.shape

        self.C = torch.cdist(supp, supp)
        K = torch.exp(-(self.C**2)/self.eps)


        v = torch.ones((k,n)).to(self.device)
        u = torch.ones((k,n)).to(self.device)

        old_bary = torch.ones(n).to(self.device)
        b = torch.zeros(n).to(self.device)

        while torch.linalg.norm(b-old_bary) > self.thresh:

            old_bary = b.clone()

            for s in range(k):
                v[s] = a[s] / (K.T@u[s])

            b = torch.zeros(n).to(self.device)
            for s in range(k):
                b = b + lambdas[s]*torch.log(K@v[s])
            b = torch.exp(b)

            for s in range(k):
                u[s] = b / (K@v[s])


        couplings = None
        if compute_couplings:
            couplings = torch.zeros_like(K)
            for s in range(k):
                couplings[s] = torch.diag(u[s])@K[s]@torch.diag(v[s])

        return b, couplings
