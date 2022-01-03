import torch
import math
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

class Sinkhorn:

    def __init__(self, niter=3000, eps=0.005, device="cuda"):
        self.niter = niter
        self.eps = eps
        self.device = device

    def solve(self, lambdas, mus):
        '''
        Computes the Wasserstein barycenter W_2 between the mus

        - lambdas (shape k): weights
        - mus (shape [k, n]): discrete distributions
        - Cs (shape [k, n, n]): cost matrices
        '''

        lambdas = lambdas.to(self.device)
        mus = mus.to(self.device)

        k,n = mus.shape
        v = torch.ones((k,n)).to(device)
        u = v.clone()

        t = torch.linspace(0,1,n).to(device)
        [Y,X] = torch.meshgrid(t,t).to(device)
        K = torch.exp(-(X-Y)**2/self.eps)

        for l in tqdm(range(self.niter)):

            for s in range(k):
                v[s] = mus[s] / (K.T@u[s])

            a = torch.zeros(n).to(device)
            for s in range(k):
                a = a + lambdas[s]*torch.log(K@v[s])
            a = torch.exp(a)

            for s in range(k):
                u[s] = a / (K@v[s])
        return a

def gauss(x, sigma, mu):
    factor = 1/(sigma*math.sqrt(2*torch.pi))
    in_exp = ((x-mu)**2) / (2*sigma**2)
    return factor*torch.exp(-in_exp)

# Just for testing
if __name__ == "__main__":
    X = torch.arange(0,10,0.1)
    a = gauss(X,1,4)
    a = a/a.sum()
    b = gauss(X,1,8)
    b = b/b.sum()

    lambdas = torch.tensor([0.5,0.5])
    mus = torch.stack((a,b))

    solver = Sinkhorn()
    bary = solver.solve(lambdas, mus)

    plt.plot(X,a)
    plt.plot(X,b)
    plt.plot(X,bary)
    plt.show()

    mus = torch.stack((a,b))
