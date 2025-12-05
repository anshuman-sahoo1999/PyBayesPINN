import torch
from pybayespinn import BayesianNetwork, PhysicsUtils

def test_model_initialization():
    model = BayesianNetwork(input_dim=2, output_dim=1)
    x = torch.randn(10, 2)
    y = model(x)
    assert y.shape == (10, 1), "Output shape mismatch"

def test_derivatives():
    # Test if derivative of x^2 is 2x
    x = torch.tensor([[3.0]], requires_grad=True)
    u = x**2
    du_dx = PhysicsUtils.first_derivative(u, x)
    assert torch.isclose(du_dx, torch.tensor([[6.0]])), "Derivative calculation failed"

if __name__ == "__main__":
    test_model_initialization()
    test_derivatives()
    print("All tests passed!")