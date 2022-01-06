import torch
from Sinkhorn import Sinkhorn
from JLWasserstein import JLProjWasserstein
import math
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

if __name__ == "__main__":

    # Parameters
    k = 10  #input distributions
    d = 4096 # original dimension 

    eps = 0.1 #epsilon for Sinkhorn
    thresh = 1e-5 #threshold for Sinkhorn
    n = 1000 #support size
    max_m = 100 
    n_trials = 20

    device = "cpu"
    
    support = torch.rand((n,d))/n

    t = torch.arange(0,n)
    Gaussian = lambda t0: torch.exp(-(t-t0)**2)

    a = torch.zeros((k,n)) # weights
    for s in range(k):
        a[s] = Gaussian(s)
        #normalization
        a[s] = a[s]/a[s].sum()

    lambdas = torch.ones(k)/k # lambdas
    solver = Sinkhorn(thresh, eps=eps, device=device)

    # To get couplings
    bary, couplings = solver.solve(lambdas, a, support, True)
    cost1 = torch.multiply(couplings,solver.C).sum()

    t = torch.arange(n)
    plt.figure(figsize=(10,5))
    plt.bar(t, bary, width = 1, color = "darkblue")
    plt.title("barycenter weights")
    plt.xlabel("j")
    plt.ylabel("b_j")
    plt.savefig("histo.png")
    plt.show()


    M = torch.arange(1,max_m+1,1)
    
    costs = torch.zeros((n_trials, max_m))
    bary_dif = torch.zeros((n_trials, max_m))
    speedup = torch.zeros((n_trials, max_m))

    loss = torch.nn.MSELoss()

    for trial in tqdm(range(n_trials)):
        
        t0 = time.time()
        solver.solve(lambdas, a, support, False)
        t1 = time.time()
        ref_time = t1-t0

        for i,m in enumerate(M):

            dimsolver = JLProjWasserstein(solver, m, device=device)
            t2 = time.time()
            bary2, couplings2 = dimsolver.solve(lambdas, a, support )
            t3 = time.time()
            cost2 = torch.multiply(couplings,dimsolver.solver.C).sum() 

            red_time = t3 - t2
            
            costs[trial, i] = cost1/cost2
            bary_dif[trial, i] = loss(bary, bary2)
            speedup[trial, i] = ref_time/red_time

    plt.figure(figsize=(10,5))
    plt.errorbar(M.cpu(), torch.mean(bary_dif, axis=0), torch.std(bary_dif, axis=0), ecolor='lightgray')
    plt.title("L2 difference between the barycenter weights")
    plt.xlabel("dimension")
    plt.ylabel("$||b_j - b_j^*||$")
    plt.savefig("l2.png")
    plt.show()
    
    plt.figure(figsize=(10,5))
    plt.errorbar(M.cpu(), torch.mean(costs, axis=0), torch.std(costs, axis=0), ecolor='lightgray')
    plt.title("Cost ratio")
    plt.xlabel("dimension")
    plt.ylabel("cost ratio")
    plt.savefig("ratio.png")
    plt.show()
    
    plt.figure(figsize=(10,5))
    plt.errorbar(M.cpu(), torch.mean(speedup, axis=0), torch.std(speedup, axis=0), ecolor='lightgray')
    plt.title("Speedup")
    plt.xlabel("dimension")
    plt.ylabel("speedup factor")
    plt.savefig("speedup.png")
    plt.show()
