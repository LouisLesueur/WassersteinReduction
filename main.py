import torch
from Sinkhorn import Sinkhorn
from JLWasserstein import JLProjWasserstein
import math
import matplotlib.pyplot as plt
import time
from tools import *

if __name__ == "__main__":

    # Parameters
    k = 10  #input distributions
    d = 1000 #dimension 

    eps = 0.001
    thresh = 1e-5
    n = 1000
    max_m = 30
    n_trials = 10

    device = "cpu"
    
    support = torch.rand((n,d))/n
    print(f"{n} points in support !")

    t = torch.arange(0,n)
    Gaussian = lambda t0,sigma: torch.exp(-(t-t0)**2/(2*sigma**2))

    #a = torch.randn((k,n)) # weights

    a = torch.zeros((k,n)) # weights
    for s in range(k):
        a[s] = Gaussian(s, 1)

    lambdas = torch.ones(k)/k # lambdas

    solver = Sinkhorn(thresh, eps=eps, device=device)

    # To get couplings
    bary, couplings = solver.solve(lambdas, a, support, True)
    
    t0 = time.time()
    bary, _ = solver.solve(lambdas, a, support, False)
    cost1 = cost(couplings, solver.C)
    t1 = time.time()
    print(f"bary 1 {bary}: computed in {t1-t0}s, cost={cost1}")
    ref_time = t1-t0

    
    M = torch.arange(1,max_m+1,1)
    
    costs = torch.zeros((n_trials, max_m))
    bary_dif = torch.zeros((n_trials, max_m))
    speedup = torch.zeros((n_trials, max_m))

    loss = torch.nn.MSELoss()

    for trial in range(n_trials):

        for i,m in enumerate(M):

            dimsolver = JLProjWasserstein(solver, m, device=device)
            t2 = time.time()
            bary2, couplings2 = dimsolver.solve(lambdas, a, support )
            t3 = time.time()
            cost2 = cost(couplings, dimsolver.solver.C)
            print(f"bary 2: computed in {t3-t2}s, cost={cost2}")

            red_time = t3 - t2

            print("cost ratio: ", cost1/cost2)
            
            costs[trial, i] = cost1/cost2
            bary_dif[trial, i] = loss(bary, bary2)
            speedup[trial, i] = ref_time/red_time


    plt.errorbar(M.cpu(), torch.mean(bary_dif, axis=0), torch.std(bary_dif, axis=0), ecolor='lightgray')
    plt.title("L2 difference between the barycenter weights")
    plt.show()
    plt.errorbar(M.cpu(), torch.mean(costs, axis=0), torch.std(costs, axis=0), ecolor='lightgray')
    plt.title("Cost ratio")
    plt.show()
    plt.errorbar(M.cpu(), torch.mean(speedup, axis=0), torch.std(speedup, axis=0), ecolor='lightgray')
    plt.title("Speedup")
    plt.show()
