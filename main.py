import torch
from Sinkhorn import Sinkhorn
from JLWasserstein import JLProjWasserstein
import math
import matplotlib.pyplot as plt
import time
from tools import *

if __name__ == "__main__":

    # Parameters
    k = 10 #input distributions
    T = 20 #support for input distributions
    n = T #support for barycenter
    d = 1000 #dimension 
    m = 5 #dimension for reduction

    eps = 10
    niter = 1000

    device = "cpu"

    support = torch.zeros((T,d))
    for ind in range(T):
        support[ind,ind] = 1
    #support += 0.01*torch.randn((T,d))

    # Point clouds
    X = torch.zeros((k,T,d))
    for s in range(k):
        X[s] = support.clone()

    t = torch.arange(0,T)
    Gaussian = lambda t0,sigma: torch.exp(-(t-t0)**2/(2*sigma**2))

    a = torch.zeros((k,T)) # weights
    for s in range(k):
        a[s] = Gaussian(s-4, 1)

    lambdas = torch.ones(k)/k # lambdas

    solver = Sinkhorn(niter=niter, eps=eps, device=device)
    t0 = time.time()
    bary, couplings, K = solver.solve(lambdas, X, a, support)
    t1 = time.time()
    cost1 = cost(K, couplings.to(device))
    print(f"bary 1: computed in {t1-t0}s, cost={cost1}")

    M = torch.arange(1, 100, 1)
    costs = torch.zeros(M.shape[0])

    for i,m in enumerate(M):

        dimsolver = JLProjWasserstein(solver, m, device=device)
        t2 = time.time()
        bary2, couplings2, K2 = dimsolver.solve(lambdas, X, a, support )
        t3 = time.time()
        cost2 = cost(K2, couplings2.to(device))
        print(f"bary 2: computed in {t3-t2}s, cost={cost2}")

        print("cost ratio: ", cost2/cost1)
        costs[i] = cost2/cost1

    plt.plot(M.cpu(), costs.cpu())
    plt.show()
