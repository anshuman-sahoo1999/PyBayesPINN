# PyBayesPINN: Bayesian Physics-Informed Neural Networks

PyBayesPINN is a comprehensive Python library designed to bridge the gap between Scientific Computing and Probabilistic Deep Learning.

Unlike standard Physics-Informed Neural Networks (PINNs) which provide deterministic point estimates, PyBayesPINN introduces rigorous uncertainty quantification using Bayesian Variational Inference (Monte Carlo Dropout). It allows researchers to solve partial differential equations (PDEs) while visualizing the "error bars" of the simulation, essential for high-stakes engineering applications.

## 🚀 Key Features

### 🧠 Bayesian Uncertainty
Automatically estimates epistemic uncertainty in physics simulations using Monte Carlo sampling.

### 🌊 Multi-Physics Support
Built-in solvers for:
- Wave Equation (Linear wave propagation)
- Viscous Burgers' Equation (Non-linear fluid dynamics & shockwaves)

### ⚙️ Hybrid Optimization
Implements a two-stage training strategy:
- **Adam**: For fast global convergence.
- **L-BFGS**: For high-precision fine-tuning (essential for capturing shock fronts).

### 🔌 Scikit-Learn Style API
Designed for ease of use. If you know `model.fit()`, you can use PyBayesPINN.

## 📦 Installation

This package is currently hosted on GitHub to facilitate open research review. You can install it directly into Google Colab or your local environment via pip:

```bash
pip install git+https://github.com/anshuman-sahoo1999/PyBayesPINN.git
