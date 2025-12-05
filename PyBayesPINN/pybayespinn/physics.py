import torch

class PhysicsUtils:
    """
    Toolkit for computing high-order derivatives automatically using PyTorch Autograd.
    """
    @staticmethod
    def first_derivative(u, x):
        """Computes du/dx"""
        grads = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]
        return grads

    @staticmethod
    def second_derivative(u, x):
        """Computes d^2u/dx^2"""
        du_dx = PhysicsUtils.first_derivative(u, x)
        d2u_dx2 = torch.autograd.grad(
            du_dx, x,
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True,
            retain_graph=True
        )[0]
        return d2u_dx2

# --- Model 1: Wave Equation (Linear) ---
def wave_equation_loss(model, xt_collocation, c=1.0):
    """
    Physics Residual for the 1D Wave Equation: u_tt - c^2 * u_xx = 0
    """
    xt_collocation.requires_grad = True
    u = model(xt_collocation)
    
    # Gradients
    grads = torch.autograd.grad(u, xt_collocation, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]
    
    # Second Derivatives
    u_xx = torch.autograd.grad(u_x, xt_collocation, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0:1]
    u_tt = torch.autograd.grad(u_t, xt_collocation, torch.ones_like(u_t), create_graph=True, retain_graph=True)[0][:, 1:2]
    
    residual = u_tt - (c**2 * u_xx)
    return torch.mean(residual**2)

# --- Model 2: Burgers' Equation (Non-Linear) ---
def burgers_equation_loss(model, xt_collocation, nu=0.01/3.14159):
    """
    Physics Residual for Viscous Burgers' Equation: u_t + u*u_x - nu*u_xx = 0
    """
    xt_collocation.requires_grad = True
    u = model(xt_collocation)
    
    # Gradients
    grads = torch.autograd.grad(u, xt_collocation, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]
    
    # Second Derivatives
    u_xx = torch.autograd.grad(u_x, xt_collocation, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0:1]
    
    # The Physics: u_t + u*u_x - nu*u_xx
    residual = u_t + (u * u_x) - (nu * u_xx)
    return torch.mean(residual**2)
