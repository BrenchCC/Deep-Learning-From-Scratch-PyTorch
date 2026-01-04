import os
import sys
import logging

import torch
from torch.optim import Optimizer

logger = logging.getLogger("Optimizers")

sys.path.append(os.getcwd())

class StochasticGradientDescent(Optimizer):
    """
    Implementation of SGD with Momentum.
    """
    def __init__(self, params, lr = 1e-3, momentum = 0.0):
        defaults = dict(lr = lr, momentum = momentum)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad

                # state initialization
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    buf = state['momentum_buffer'] = torch.clone(grad).detach()
                else:
                    buf = state['momentum_buffer']
                    # velocity update: v = momentum * v + g
                    buf.mul_(momentum).add_(grad)
                
                # update: p = p - lr * v
                p.add_(buf, alpha = -lr)
        return loss
                
class AdaptiveMomentEstimationW(Optimizer):
    """
    Implementation of AdamW (Adam with decoupled weight decay)
    """ 
    def __init__(self, params, lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 1e-2):
        defaults = dict(lr = lr, betas = betas, eps = eps, weight_decay = weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad

                # 1.decoupled weight decay p = p * (1 - lr * weight_decay)
                p.mul_(1 - lr * weight_decay)

                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                # 2.momentum mean
                exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)
                
                # 3.rms variance
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)

                # 4. bias correction
                bias_corr1 = 1 - beta1 ** t
                bias_corr2 = 1 - beta2 ** t

                step_size = lr / bias_corr1

                denom = (exp_avg_sq.sqrt() / bias_corr2 ** 0.5).add_(eps)
        
        return loss

if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()],
        datefmt = '%Y-%m-%d %H:%M:%S'
    )
    
    # Simple quadratic optimization test: y = (x - 5)^2
    # Goal: x should converge to 5.0
    
    # 1. Test SGD
    x = torch.tensor([0.0], requires_grad = True)
    # Compare calling implementation vs PyTorch interface
    # Note: Our implementation matches torch.optim.SGD structure
    opt = StochasticGradientDescent([x], lr = 0.1, momentum = 0.9)
    
    logger.info("Testing SGD convergence (Target: 5.0)...")
    for i in range(50):
        opt.zero_grad()
        loss = (x - 5.0) ** 2
        loss.backward()
        opt.step()
    
    logger.info(f"Final x value: {x.item():.4f}")
    
    # 2. Test AdamW
    x_adam = torch.tensor([0.0], requires_grad = True)
    # Official PyTorch Interface usage for reference:
    # opt_torch = torch.optim.AdamW([x_adam], lr=0.1)
    
    opt_adam = AdaptiveMomentEstimationW([x_adam], lr = 0.1, weight_decay = 0.0)
    logger.info("Testing AdamW convergence (Target: 5.0)...")
    for i in range(50):
        opt_adam.zero_grad()
        loss = (x_adam - 5.0) ** 2
        loss.backward()
        opt_adam.step()
        
    logger.info(f"Final x_adam value: {x_adam.item():.4f}")