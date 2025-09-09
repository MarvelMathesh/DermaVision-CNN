"""
Optimizers and Schedulers Module
===============================

Advanced optimization strategies for training skin disease detection models.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, CosineAnnealingWarmRestarts, 
    ReduceLROnPlateau, OneCycleLR, StepLR
)
import math
from typing import Union, Dict, Any


class AdamW(optim.AdamW):
    """Enhanced AdamW optimizer with custom defaults for medical imaging."""
    
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=1e-2, amsgrad=False):
        super().__init__(params, lr=lr, betas=betas, eps=eps, 
                        weight_decay=weight_decay, amsgrad=amsgrad)


class LAMB(optim.Optimizer):
    """
    Layer-wise Adaptive Moments optimizer for large batch training.
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.01, clamp_value=10.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       clamp_value=clamp_value)
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('LAMB does not support sparse gradients')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Apply bias correction
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                
                # Compute update
                update = exp_avg / denom
                
                # Add weight decay
                if group['weight_decay'] != 0:
                    update.add_(p.data, alpha=group['weight_decay'])
                
                # Apply trust ratio
                weight_norm = p.data.norm()
                update_norm = update.norm()
                
                if weight_norm > 0 and update_norm > 0:
                    trust_ratio = min(group['clamp_value'], weight_norm / update_norm)
                    step_size *= trust_ratio
                
                p.data.add_(update, alpha=-step_size)
        
        return loss


class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with warm restarts and warmup.
    """
    
    def __init__(self, optimizer, T_0, T_mult=1, eta_min=0, warmup_epochs=0, 
                 warmup_start_lr=1e-8, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Warmup phase
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * 
                   (self.last_epoch / self.warmup_epochs) for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            epoch = self.last_epoch - self.warmup_epochs
            T_cur = epoch % self.T_0
            return [self.eta_min + (base_lr - self.eta_min) * 
                   (1 + math.cos(math.pi * T_cur / self.T_0)) / 2 
                   for base_lr in self.base_lrs]


def create_optimizer(parameters, 
                    optimizer_type: str = 'adamw',
                    lr: float = 1e-4,
                    weight_decay: float = 1e-2,
                    **kwargs) -> optim.Optimizer:
    """
    Factory function to create optimizers.
    
    Args:
        parameters: Model parameters
        optimizer_type: Type of optimizer ('adamw', 'sgd', 'adam', 'lamb')
        lr: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer-specific arguments
    
    Returns:
        Initialized optimizer
    """
    if optimizer_type.lower() == 'adamw':
        return AdamW(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    
    elif optimizer_type.lower() == 'adam':
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    
    elif optimizer_type.lower() == 'sgd':
        momentum = kwargs.get('momentum', 0.9)
        nesterov = kwargs.get('nesterov', True)
        return optim.SGD(parameters, lr=lr, weight_decay=weight_decay,
                        momentum=momentum, nesterov=nesterov)
    
    elif optimizer_type.lower() == 'lamb':
        return LAMB(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    
    elif optimizer_type.lower() == 'rmsprop':
        return optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay, **kwargs)
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def create_scheduler(optimizer: optim.Optimizer,
                    scheduler_type: str = 'cosine',
                    num_epochs: int = 200,
                    warmup_epochs: int = 10,
                    **kwargs) -> Union[optim.lr_scheduler._LRScheduler, None]:
    """
    Factory function to create learning rate schedulers.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
        num_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        **kwargs: Additional scheduler-specific arguments
    
    Returns:
        Initialized scheduler or None
    """
    if scheduler_type.lower() == 'cosine':
        return CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs,
                               eta_min=kwargs.get('eta_min', 1e-7))
    
    elif scheduler_type.lower() == 'cosine_warmup':
        T_0 = kwargs.get('T_0', num_epochs // 4)
        return CosineAnnealingWarmupRestarts(
            optimizer, T_0=T_0, warmup_epochs=warmup_epochs,
            eta_min=kwargs.get('eta_min', 1e-7)
        )
    
    elif scheduler_type.lower() == 'cosine_restart':
        T_0 = kwargs.get('T_0', num_epochs // 4)
        return CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=kwargs.get('T_mult', 2),
            eta_min=kwargs.get('eta_min', 1e-7)
        )
    
    elif scheduler_type.lower() == 'plateau':
        return ReduceLROnPlateau(
            optimizer, mode='min', factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10), verbose=True,
            min_lr=kwargs.get('min_lr', 1e-7)
        )
    
    elif scheduler_type.lower() == 'onecycle':
        return OneCycleLR(
            optimizer, max_lr=kwargs.get('max_lr', 1e-3),
            total_steps=num_epochs, pct_start=kwargs.get('pct_start', 0.1),
            anneal_strategy='cos'
        )
    
    elif scheduler_type.lower() == 'step':
        step_size = kwargs.get('step_size', num_epochs // 3)
        gamma = kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type.lower() == 'none':
        return None
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


# Export functions
__all__ = [
    'AdamW',
    'LAMB',
    'CosineAnnealingWarmupRestarts',
    'create_optimizer',
    'create_scheduler'
]


if __name__ == "__main__":
    # Test optimizer and scheduler creation
    print("Testing Optimizers and Schedulers...")
    
    import torch.nn as nn
    
    # Create a simple model for testing
    model = nn.Linear(10, 5)
    
    # Test optimizer creation
    optimizer_types = ['adamw', 'sgd', 'adam', 'lamb']
    
    for opt_type in optimizer_types:
        try:
            optimizer = create_optimizer(model.parameters(), opt_type, lr=1e-4)
            print(f"{opt_type.upper()} optimizer created successfully")
        except Exception as e:
            print(f"Error creating {opt_type} optimizer: {e}")
    
    # Test scheduler creation
    scheduler_types = ['cosine', 'cosine_warmup', 'plateau', 'onecycle']
    
    optimizer = create_optimizer(model.parameters(), 'adamw')
    
    for sched_type in scheduler_types:
        try:
            scheduler = create_scheduler(optimizer, sched_type, num_epochs=100)
            print(f"{sched_type.upper()} scheduler created successfully")
        except Exception as e:
            print(f"Error creating {sched_type} scheduler: {e}")
    
    print("Optimizer and scheduler testing completed!")