import torch
from Sinkhorn import Sinkhorn
from JLWasserstein import JLProjWasserstein
import math
import matplotlib.pyplot as plt
import time

if __name__ == "__main__":

    # Parameters
    k = 2  #input distributions
    d = 1000 #dimension 

    eps = 0.01
    thresh = 1e-5
    n0 = 4
    n_trials = 10

    device = "cpu"

    D = torch.arange(8)


    times = torch.zeros((n_trials, D.shape[0]))
    
    for trial in range(n_trials):

        print(f"trial {trial+1}/{n_trials}")

        for i,d in enumerate(D):
            d = int(d)
            n = n0**d
            support = torch.rand((n,d))/n

            a = torch.randn((k,n)) # weights
            lambdas = torch.ones(k)/k # lambdas

            solver = Sinkhorn(thresh, eps=eps, device=device)

            t0 = time.time()
            bary, _ = solver.solve(lambdas, a, support, False)
            t1 = time.time()
            times[trial, i] = t1-t0

    plt.figure(figsize=(10,5))
    plt.errorbar(D.cpu(), torch.mean(times, axis=0), torch.std(times, axis=0), ecolor='lightgray')
    plt.title("Mean execution time")
    plt.xlabel("Dimension d")
    plt.ylabel("time to compute barycenter (s)")
    plt.savefig("curse.png")
    plt.show()
