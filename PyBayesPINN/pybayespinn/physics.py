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

def wave_equation_loss(model, xt_collocation, c=1.0):
    """
    Calculates the Physics Residual for the 1D Wave Equation:
    u_tt - c^2 * u_xx = 0
    
    Args:
        model: The neural network
        xt_collocation: Tensor of shape [N, 2] representing (x, t) points
        c: Wave speed constant
    """
    # Enable gradient tracking for inputs
    xt_collocation.requires_grad = True
    
    # Forward pass
    u = model(xt_collocation)
    
    # Compute gradients w.r.t input (x, t)
    grads = torch.autograd.grad(u, xt_collocation, torch.ones_like(u), create_graph=True, retain_graph=True)[0]
    
    # Split gradients: column 0 is x, column 1 is t
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    
    # Compute second derivatives
    u_xx = torch.autograd.grad(u_x, xt_collocation, torch.ones_like(u_x), create_graph=True, retain_graph=True)[0][:, 0:1]
    u_tt = torch.autograd.grad(u_t, xt_collocation, torch.ones_like(u_t), create_graph=True, retain_graph=True)[0][:, 1:2]
    
    # The Physics Residual (Should be 0)
    residual = u_tt - (c**2 * u_xx)
    
    return torch.mean(residual**2)