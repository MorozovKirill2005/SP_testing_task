import torch

class Adam:
    def __init__(self, param, alpha = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        self.param = param
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [torch.zeros_likes(p) for p in param]
        self.v = [torch.zeros_likes(p) for p in param]

    def step(self):
        self.t += 1
        for i, par in enumerate(self.param):
            gt = self.par.grad.data
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * gt
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (gt ** 2)
            mt = self.m[i] / (1 - self.beta1 ** self.t)
            vt = self.v[i] / (1 - self.beta2 ** self.t)
            par.data = par.data - self.alpha * mt / (vt ** 0.5 + self.eps)