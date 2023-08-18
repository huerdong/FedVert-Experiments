import torch
import copy

def next_moment(moment, gradient, decay=0):
    return moment * (decay) + (1-decay) * gradient

def adagrad(gradient, prev, decay=0):
    return prev.addcmul(gradient, gradient.conj(), 1)

def adam(gradient, prev, decay=0):
    return prev.mul(decay).addcmul(gradient, gradient.conj(), 1-decay)

def yogi(gradient, prev, decay=0):
    second_moment = gradient.mul(gradient.conj())
    sign = torch.sign(prev - second_moment)
    return prev.mul(decay).addcmul_(sign, second_moment, 1-decay)

alg_table = {"adam": adam, "adagrad": adagrad, "yogi": yogi}

# Implements Averaging aggregation
# Modify this for different algorithms
# For FedOPT need memory of previous iterations and some extra parameters
class OPTAggregator:
    def __init__(self, server_lr, tau, w_init, m=None, v=None, beta1=0, beta2=0, alg="adagrad"):
        # Initial parameters, not sure about
        self.s_lr = server_lr
        self.tau = tau
        self.beta1 = beta1
        self.beta2 = beta2
        self.w = w_init

        if self.m is None:
            self.m = copy.deepcopy(self.w_init)
        else:
            self.m = m

        if self.v is None:
            self.v = {}
            for k in self.w.keys():
                self.v[k] = torch.full(self.w[k].shape, tau * tau)
        else:
            self.v = v

        self.func = alg_table[alg]

    def aggregate(self, w):
        grad = copy.deepcopy(w[0])
        w_opt = copy.deepcopy(self.w_init)
        for k in grad.keys():
            for i in range(1, len(w)):
                grad[k] += w[i][k]
            grad[k] = torch.div(grad[k], len(w))
        
            grad[k] = grad[k] - self.w[k]
            
            self.m[k] = next_moment(grad[k], decay=self.beta1)
            self.v[k] = self.func(grad[k], self.v[k], decay=self.beta2)

            denom = torch.sqrt(self.v[k]).add_(self.tau)
            self.w[k].addcdiv_(self.m[k], denom, self.s_lr) 

        return self.w
