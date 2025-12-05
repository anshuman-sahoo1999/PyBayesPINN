import matplotlib.pyplot as plt
import numpy as np

def plot_results(x_test, u_true, u_mean, u_std, title="Bayesian PINN Result"):
    """
    Generates the standard 'JSS-style' plot with confidence intervals.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot Mean Prediction
    plt.plot(x_test, u_mean, 'b-', label='Predicted Mean', linewidth=2)
    
    # Plot Exact Solution (if available)
    if u_true is not None:
        plt.plot(x_test, u_true, 'r--', label='Exact Solution', alpha=0.7)
        
    # Plot Uncertainty (95% Confidence Interval: Mean +/- 2*Std)
    plt.fill_between(
        x_test.flatten(), 
        (u_mean - 2*u_std).flatten(), 
        (u_mean + 2*u_std).flatten(), 
        color='blue', alpha=0.2, label='95% Confidence Interval'
    )
    
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    return plt