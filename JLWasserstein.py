import torch
import math

class JLProjWasserstein:

    def __init__(self, solver, m, device="cuda"):
        self.m = m
        self.solver = solver
        self.device = device

    def solve(self, lambdas, X, a, supp):
        '''
        Compute the Wasserstein barrycenter W_2 between point clouds, with dimensionality reduction

        - lambdas (shape k): weights
        - X (shape [k, T, d]): points clouds
        - a (shape [k, T]): weights
        - supp (shape [n, d]): barycenter support
        '''

        k,T,d = X.shape
        n = supp.shape[0]
        
        JLProj = torch.randn((self.m,d)).to(self.device)/math.sqrt(self.m)
        
        # 1. Projection on IR^m, with normalization
        X, supp = X.to(self.device), supp.to(self.device)
        lambdas = lambdas.to(self.device)
        reduced_X = torch.einsum('kTd,md->kTm', X, JLProj)
        reduced_supp = torch.einsum('nd,md->nm', supp, JLProj)

        # 2. solve with solver
        bary, couplings, K = self.solver.solve(lambdas, reduced_X, a, reduced_supp)
        
        return bary, couplings, K
