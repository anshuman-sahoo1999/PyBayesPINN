import torch
import numpy as np
import time

class BayesianPINNSolver:
    """
    Solver that trains the B-PINN and estimates uncertainty.
    """
    def __init__(self, model, physics_loss_function):
        self.model = model
        self.physics_loss_func = physics_loss_function
        # Initialize Adam optimizer
        self.optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.history = {'loss': []}

    def train(self, x_data, u_data, x_collocation, epochs=1000, use_lbfgs=True):
        """
        Trains the model using Adam (fast convergence) followed by L-BFGS (high precision).
        """
        self.model.train() # Enable Dropout during training
        
        print(f"--- Starting Training (Adam: {epochs} epochs) ---")
        start_time = time.time()

        # Phase 1: Adam Optimization
        for epoch in range(epochs):
            self.optimizer_adam.zero_grad()
            
            # Data Loss
            u_pred = self.model(x_data)
            loss_data = torch.mean((u_pred - u_data)**2)
            
            # Physics Loss
            loss_physics = self.physics_loss_func(self.model, x_collocation)
            
            loss = loss_data + loss_physics
            loss.backward()
            self.optimizer_adam.step()
            
            self.history['loss'].append(loss.item())
            
            if epoch % 200 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.5f}")

        # Phase 2: L-BFGS Optimization (Optional but recommended for JSS)
        if use_lbfgs:
            print("--- Starting L-BFGS Fine-Tuning ---")
            self.optimizer_lbfgs = torch.optim.LBFGS(
                self.model.parameters(), 
                lr=1.0, 
                max_iter=50000, 
                max_eval=50000, 
                history_size=50,
                line_search_fn="strong_wolfe"
            )

            def closure():
                self.optimizer_lbfgs.zero_grad()
                u_pred = self.model(x_data)
                loss_data = torch.mean((u_pred - u_data)**2)
                loss_physics = self.physics_loss_func(self.model, x_collocation)
                loss = loss_data + loss_physics
                loss.backward()
                return loss

            self.optimizer_lbfgs.step(closure)
            
        print(f"Training Completed in {time.time() - start_time:.2f} seconds.")

    def predict_with_uncertainty(self, x_test, n_samples=100):
        """
        Performs Monte Carlo Sampling to estimate mean and uncertainty.
        
        Returns:
            mean_pred: The average prediction.
            std_pred: The standard deviation (uncertainty).
        """
        self.model.train() # KEEP Dropout ON during inference
        
        preds = []
        # Convert test data to tensor if it isn't already
        if not torch.is_tensor(x_test):
            x_test = torch.tensor(x_test, dtype=torch.float32)

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(x_test)
                preds.append(pred.cpu().numpy())
        
        preds = np.array(preds)
        
        mean_pred = np.mean(preds, axis=0)
        std_pred = np.std(preds, axis=0) # This is the "Error Bar"
        
        return mean_pred, std_pred