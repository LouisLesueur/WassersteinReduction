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

    eps = 0.0005
    niter = 1000

    device = "cpu"

    support = torch.zeros((T,d))
    for ind in range(T):
        support[ind,ind] = 1/T
    #support += 0.01*torch.randn((T,d))

    # Point clouds
    X = torch.zeros((k,T,d))
    for s in range(k):
        X[s] = support.clone()

    t = torch.arange(0,T)
    Gaussian = lambda t0,sigma: torch.exp(-(t-t0)**2/(2*sigma**2))

    a = torch.zeros((k,T)) # weights
    for s in range(k):
        a[s] = Gaussian(s, 1)

    lambdas = torch.ones(k)/k # lambdas

    solver = Sinkhorn(niter=niter, eps=eps, device=device)
    t0 = time.time()
    bary, couplings, K = solver.solve(lambdas, X, a, support)
    t1 = time.time()
    cost1 = cost(K, couplings.to(device))
    print(f"bary 1: computed in {t1-t0}s, cost={cost1}")

    ref_time = t1-t0

    max_m = 20
    n_trials = 5
    
    M = torch.arange(max_m)
    
    costs = torch.zeros((n_trials, max_m))
    bary_dif = torch.zeros((n_trials, max_m))
    speedup = torch.zeros((n_trials, max_m))

    loss = torch.nn.MSELoss()

    for trial in range(n_trials):

        for i,m in enumerate(M):

            dimsolver = JLProjWasserstein(solver, m, device=device)
            t2 = time.time()
            bary2, couplings2, K2 = dimsolver.solve(lambdas, X, a, support )
            t3 = time.time()
            cost2 = cost(K2, couplings.to(device))
            print(f"bary 2: computed in {t3-t2}s, cost={cost2}")

            red_time = t3 - t2

            print("cost ratio: ", cost2/cost1)
            
            costs[trial, i] = cost2/cost1
            bary_dif[trial, i] = loss(bary, bary2)
            speedup[trial, i] = red_time/ref_time


    plt.errorbar(M.cpu(), torch.mean(bary_dif, axis=0), torch.std(bary_dif, axis=0), ecolor='lightgray')
    plt.title("L2 difference between the barycenter weights")
    plt.show()
    plt.errorbar(M.cpu(), torch.mean(costs, axis=0), torch.std(costs, axis=0), ecolor='lightgray')
    plt.title("Cost ratio")
    plt.show()
    plt.errorbar(M.cpu(), torch.mean(speedup, axis=0), torch.std(speedup, axis=0), ecolor='lightgray')
    plt.title("Speedup")
    plt.show()
