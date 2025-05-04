# -*- coding: utf-8 -*-
"""Python script for LP generation, solving, NN modeling, training, and evaluation.

This script provides functions to support the experiments outlined in the project
proposal for using Neural Networks to solve Linear Programming problems.

Note: This script requires numpy, scipy, and tensorflow to be installed.
      `pip install numpy scipy tensorflow`
"""

import numpy as np
from scipy.optimize import linprog
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

def generate_lp(n, m, sparsity=0.1, ensure_feasible_bounded=True):
    """Generates a random standard form LP problem (min c'x s.t. Ax=b, x>=0).

    Args:
        n (int): Number of variables.
        m (int): Number of constraints.
        sparsity (float): Approximate fraction of non-zero elements in A.
        ensure_feasible_bounded (bool): If True, attempts to generate a problem
                                       that is likely feasible and bounded.
                                       (Note: This is heuristic and not guaranteed).

    Returns:
        tuple: (c, A, b) representing the LP problem, or None if generation fails
               to meet feasibility/boundedness criteria heuristically.
    """
    if m >= n:
        print("Warning: Number of constraints m should ideally be less than n.")

    # Generate c: Coefficients for the objective function
    c = np.random.rand(n) * 10 # Coefficients between 0 and 10

    # Generate A: Constraint matrix with specified sparsity
    A = np.zeros((m, n))
    num_non_zero = int(m * n * sparsity)
    rows = np.random.randint(0, m, size=num_non_zero)
    cols = np.random.randint(0, n, size=num_non_zero)
    vals = np.random.randn(num_non_zero) # Standard normal values
    A[rows, cols] = vals
    # Ensure A has full row rank (heuristic)
    if np.linalg.matrix_rank(A) < m:
        # Regenerate A if rank deficient (simple approach)
        print("Regenerating A due to rank deficiency.")
        return generate_lp(n, m, sparsity, ensure_feasible_bounded)

    # Generate a feasible solution x_feasible and corresponding b
    if ensure_feasible_bounded:
        # Generate a feasible point x_feasible > 0
        x_feasible = np.random.rand(n) + 0.1 # Strictly positive
        # Calculate b = Ax_feasible
        b = A @ x_feasible

        # Heuristic check for boundedness: ensure c is not orthogonal to null space of A
        # (A more robust check involves duality, but this is simpler for generation)
        # If c is in the row space of A, the problem might be unbounded below if feasible.
        # We want c to have a component outside the row space.
        # This check is not perfect.
        null_space_basis = np.linalg.svd(A)[2][m:] # Basis for null space
        if null_space_basis.shape[0] > 0:
             projection_onto_null = null_space_basis.T @ (null_space_basis @ c)
             if np.linalg.norm(c - projection_onto_null) < 1e-6: # c approx in row space
                 print("Regenerating c for potential boundedness.")
                 # Try adding a component orthogonal to row space (in null space of A')
                 # This part is complex, simpler to just regenerate c
                 c = np.random.rand(n) * 10
                 # Re-run check (simplistic)
                 projection_onto_null = null_space_basis.T @ (null_space_basis @ c)
                 if np.linalg.norm(c - projection_onto_null) < 1e-6:
                     print("Warning: Could not easily ensure boundedness heuristically.")

    else:
        # Generate b directly (may lead to infeasible problems)
        b = np.random.randn(m)

    return c, A, b

def solve_lp(c, A, b):
    """Solves the LP problem using scipy.optimize.linprog.

    Args:
        c (np.ndarray): Objective function coefficients.
        A (np.ndarray): Constraint matrix for equality constraints (A_eq).
        b (np.ndarray): Right-hand side for equality constraints (b_eq).

    Returns:
        tuple: (x_optimal, fun_optimal, success, message, solve_time)
               Returns (None, None, False, message, solve_time) if solver fails.
    """
    start_time = time.time()
    # Using 'highs' method which is generally robust for LP
    # Bounds are x_i >= 0 for all i
    bounds = [(0, None)] * len(c)
    try:
        result = linprog(c, A_eq=A, b_eq=b, bounds=bounds, method='highs')
        solve_time = time.time() - start_time
        if result.success:
            return result.x, result.fun, True, result.message, solve_time
        else:
            return None, None, False, result.message, solve_time
    except Exception as e:
        solve_time = time.time() - start_time
        return None, None, False, f"Solver error: {e}", solve_time

def build_nn_model(input_dim, output_dim, hidden_layers=[128, 64]):
    """Builds a Keras MLP model.

    Args:
        input_dim (int): Dimension of the input layer (flattened c, A, b).
        output_dim (int): Dimension of the output layer (dimension n of x).
        hidden_layers (list[int]): List of neuron counts for each hidden layer.

    Returns:
        keras.Model: The compiled Keras model.
    """
    model = keras.Sequential(name="LP_Solver_NN")
    model.add(layers.Input(shape=(input_dim,), name="input_layer"))

    # Hidden layers
    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(units, activation='relu', name=f"hidden_{i+1}"))
        # Optional: Add Batch Normalization or Dropout
        # model.add(layers.BatchNormalization())
        # model.add(layers.Dropout(0.2))

    # Output layer - ReLU activation to encourage non-negativity (x >= 0)
    model.add(layers.Dense(output_dim, activation='relu', name="output_layer"))

    # Compile the model - Using Mean Squared Error as the primary loss
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])
    return model

def train_nn(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """Trains the Keras model.

    Args:
        model (keras.Model): The model to train.
        X_train (np.ndarray): Training input data (flattened LP problems).
        y_train (np.ndarray): Training target data (optimal x*).
        X_val (np.ndarray): Validation input data.
        y_val (np.ndarray): Validation target data.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.

    Returns:
        keras.callbacks.History: Training history object.
    """
    print(f"Starting training for {epochs} epochs...")
    # Optional: Add callbacks like EarlyStopping or ModelCheckpoint
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=1) # Set verbose=2 for less output per epoch
    print("Training finished.")
    return history

def evaluate_nn(model, X_test, y_test, lp_params_test):
    """Evaluates the trained model on the test set.

    Args:
        model (keras.Model): The trained model.
        X_test (np.ndarray): Test input data (flattened LP problems).
        y_test (np.ndarray): Test target data (true optimal x*).
        lp_params_test (list[tuple]): List of original (c, A, b) tuples for the test set.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    print("Evaluating model on test set...")
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time_nn = (time.time() - start_time) / len(X_test) # Avg inference time

    # Basic metrics from model.evaluate
    loss, mae = model.evaluate(X_test, y_test, verbose=0)

    # Calculate additional metrics
    mse = np.mean(np.square(y_test - y_pred))
    feasibility_violations = []
    optimality_gaps = []
    solver_times = []

    for i in range(len(lp_params_test)):
        c, A, b = lp_params_test[i]
        x_pred_i = y_pred[i]
        x_true_i = y_test[i]

        # 1. Feasibility Check (Ax=b violation)
        # Ensure x_pred is non-negative (model output activation helps, but clip just in case)
        x_pred_i = np.maximum(x_pred_i, 0)
        constraint_violation = np.linalg.norm(A @ x_pred_i - b)
        feasibility_violations.append(constraint_violation)

        # 2. Optimality Gap Check (c'x_pred vs c'x_true)
        obj_pred = c @ x_pred_i
        obj_true = c @ x_true_i # Assumes y_test contains true optimal solutions
        optimality_gap = abs(obj_pred - obj_true)
        optimality_gaps.append(optimality_gap)

        # 3. Solver time (for comparison - requires solving again, or storing previous times)
        # _, _, _, _, solve_time = solve_lp(c, A, b) # Re-solve or use stored time
        # solver_times.append(solve_time)

    avg_feasibility_violation = np.mean(feasibility_violations)
    avg_optimality_gap = np.mean(optimality_gaps)
    # avg_solver_time = np.mean(solver_times) if solver_times else 0

    results = {
        'test_loss (MSE)': loss,
        'test_mae': mae,
        'avg_feasibility_violation (||Ax_pred - b||)': avg_feasibility_violation,
        'avg_optimality_gap (|cT(x_pred - x*)|)': avg_optimality_gap,
        'avg_nn_inference_time_per_instance': inference_time_nn,
        # 'avg_solver_time_per_instance': avg_solver_time # Add if solver times are available
    }
    print("Evaluation finished.")
    print(results)
    return results

def run_experiment(n, m, sparsity, num_samples, nn_hidden_layers):
    """Runs a single experiment: generate data, build, train, evaluate.

    Args:
        n (int): Number of variables.
        m (int): Number of constraints.
        sparsity (float): Sparsity of matrix A.
        num_samples (int): Total number of LP instances to generate.
        nn_hidden_layers (list[int]): NN hidden layer configuration.
    """
    print(f"\n--- Running Experiment: n={n}, m={m}, sparsity={sparsity}, samples={num_samples} ---")

    # 1. Generate Data
    print("Generating LP data...")
    lp_params = []
    solutions = []
    generated_count = 0
    failed_count = 0
    while generated_count < num_samples:
        c, A, b = generate_lp(n, m, sparsity)
        if c is None: # Generation failed heuristic checks
            failed_count += 1
            if failed_count > num_samples * 2: # Avoid infinite loop
                 print("Error: Too many failures generating feasible/bounded LPs.")
                 return
            continue

        x_opt, _, success, msg, _ = solve_lp(c, A, b)
        if success:
            lp_params.append((c, A, b))
            solutions.append(x_opt)
            generated_count += 1
            if generated_count % (num_samples // 10) == 0:
                print(f"Generated {generated_count}/{num_samples} valid LP instances...")
        else:
            # print(f"Solver failed for generated LP: {msg}") # Optional: Log failures
            failed_count += 1
            if failed_count > num_samples * 2:
                 print("Error: Too many solver failures on generated LPs.")
                 return

    print(f"Data generation complete. {generated_count} instances.")

    # Flatten data for NN input
    X_data = np.array([np.concatenate([c, A.flatten(), b]) for c, A, b in lp_params])
    y_data = np.array(solutions)

    # Split data (e.g., 70% train, 15% val, 15% test)
    num_train = int(num_samples * 0.7)
    num_val = int(num_samples * 0.15)
    num_test = num_samples - num_train - num_val

    X_train, X_val, X_test = X_data[:num_train], X_data[num_train:num_train+num_val], X_data[num_train+num_val:]
    y_train, y_val, y_test = y_data[:num_train], y_data[num_train:num_train+num_val], y_data[num_train+num_val:]
    lp_params_test = lp_params[num_train+num_val:]

    # 2. Build Model
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    model = build_nn_model(input_dim, output_dim, hidden_layers=nn_hidden_layers)
    model.summary()

    # 3. Train Model
    history = train_nn(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

    # 4. Evaluate Model
    eval_results = evaluate_nn(model, X_test, y_test, lp_params_test)

    print(f"--- Experiment Finished: n={n}, m={m} ---")
    return model, history, eval_results

# Example of how to structure the experiments (do not run automatically)
if __name__ == "__main__":
    print("This script defines functions for the LP NN project.")
    print("It does not run the full experiments automatically.")
    print("To run an experiment, call run_experiment() with desired parameters.")

    # Example parameters for Experiment 1 (adjust as needed)
    # n1, m1 = 20, 10
    # sparsity1 = 0.2
    # samples1 = 1000 # Use more samples for actual training (e.g., 10000+)
    # hidden1 = [64, 32]
    # print("\nExample call for Experiment 1 (commented out):")
    # print(f"# run_experiment(n={n1}, m={m1}, sparsity={sparsity1}, num_samples={samples1}, nn_hidden_layers={hidden1})")

    # Example parameters for Experiment 2 (increasing dimensions)
    # n2, m2 = 50, 25
    # sparsity2 = 0.2
    # samples2 = 2000
    # hidden2 = [128, 64]
    # print("\nExample call for Experiment 2 (commented out):")
    # print(f"# run_experiment(n={n2}, m={m2}, sparsity={sparsity2}, num_samples={samples2}, nn_hidden_layers={hidden2})")

    # You would call run_experiment for each configuration outlined in the proposal.




# --- Polynomial Optimization Functions ---

from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
import itertools

def generate_polynomial(degree, num_vars, domain_range=(-5, 5)):
    """Generates a random polynomial function and its representation.

    Args:
        degree (int): The maximum degree of the polynomial.
        num_vars (int): The number of variables (dimension d).
        domain_range (tuple): The approximate range (min, max) for coefficients and optima search.

    Returns:
        tuple: (poly_func, coeff_vector, true_minimum_x, true_minimum_val)
               Returns None if finding minimum fails.
    """
    # Generate coefficients for all terms up to the specified degree
    poly = PolynomialFeatures(degree=degree)
    # Fit to dummy data to establish the feature mapping
    dummy_input = np.zeros((1, num_vars))
    poly.fit(dummy_input)
    num_coeffs = poly.n_output_features_

    # Generate random coefficients (excluding the constant term for simplicity in optimization focus)
    # Coeffs range can be adjusted
    coeffs = (np.random.rand(num_coeffs) - 0.5) * 4 # Range approx -2 to 2
    coeffs[0] = 0 # Set constant term to 0, or handle separately if needed

    # Define the polynomial function using the generated coefficients
    def poly_func(x):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        features = poly.transform(x)
        return (features @ coeffs)[0] # Calculate P(x)

    # Find the minimum using a numerical solver (e.g., BFGS)
    # Start search from multiple points within the domain range
    best_min_val = float("inf")
    best_min_x = None
    solver_success = False
    for _ in range(5): # Try multiple starting points
        x0 = np.random.uniform(domain_range[0], domain_range[1], size=num_vars)
        try:
            result = minimize(poly_func, x0, method="BFGS", options={"maxiter": 200})
            if result.success and result.fun < best_min_val:
                best_min_val = result.fun
                best_min_x = result.x
                solver_success = True
        except Exception as e:
            # print(f"Solver failed for polynomial: {e}") # Optional logging
            pass # Continue trying other start points

    if solver_success:
        # The input to the NN will be the coefficient vector
        coeff_vector = coeffs
        return poly_func, coeff_vector, best_min_x, best_min_val
    else:
        # print("Warning: Failed to find minimum reliably for generated polynomial.")
        return None # Indicate failure

def solve_polynomial(poly_func, num_vars, domain_range=(-5, 5)):
    """Finds the minimum of a given polynomial function numerically.

    Args:
        poly_func (callable): The polynomial function P(x).
        num_vars (int): The number of variables.
        domain_range (tuple): The range for starting points.

    Returns:
        tuple: (x_optimal, fun_optimal, success)
    """
    best_min_val = float("inf")
    best_min_x = None
    solver_success = False
    for _ in range(10): # More starting points for robustness
        x0 = np.random.uniform(domain_range[0], domain_range[1], size=num_vars)
        try:
            result = minimize(poly_func, x0, method="BFGS", options={"maxiter": 200})
            if result.success and result.fun < best_min_val:
                best_min_val = result.fun
                best_min_x = result.x
                solver_success = True
        except Exception:
            pass

    if solver_success:
        return best_min_x, best_min_val, True
    else:
        return None, None, False

def evaluate_nn_poly(model, X_test_coeffs, y_test_optima, poly_funcs_test):
    """Evaluates the trained model on the polynomial test set.

    Args:
        model (keras.Model): The trained model.
        X_test_coeffs (np.ndarray): Test input data (polynomial coefficient vectors).
        y_test_optima (np.ndarray): Test target data (true optimal x*).
        poly_funcs_test (list[callable]): List of polynomial functions for the test set.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    print("Evaluating model on polynomial test set...")
    start_time = time.time()
    y_pred_optima = model.predict(X_test_coeffs)
    inference_time_nn = (time.time() - start_time) / len(X_test_coeffs)

    # Basic metrics (distance between predicted and true optima)
    loss, mae = model.evaluate(X_test_coeffs, y_test_optima, verbose=0)
    mse = np.mean(np.square(y_test_optima - y_pred_optima))

    # Calculate difference in function values
    value_diffs = []
    for i in range(len(poly_funcs_test)):
        poly_func = poly_funcs_test[i]
        x_pred_i = y_pred_optima[i]
        x_true_i = y_test_optima[i]

        try:
            val_pred = poly_func(x_pred_i)
            val_true = poly_func(x_true_i) # Assumes y_test_optima is the true minimum point
            value_diffs.append(abs(val_pred - val_true))
        except Exception as e:
            # print(f"Error evaluating polynomial {i}: {e}")
            value_diffs.append(float('nan')) # Mark as invalid

    avg_value_diff = np.nanmean(value_diffs)

    results = {
        'test_loss_optima (MSE)': loss,
        'test_mae_optima': mae,
        'avg_value_difference |P(x_pred) - P(x*)|': avg_value_diff,
        'avg_nn_inference_time_per_instance': inference_time_nn,
    }
    print("Evaluation finished.")
    print(results)
    return results

def run_poly_experiment(degree, num_vars, num_samples, nn_hidden_layers):
    """Runs a single polynomial optimization experiment.

    Args:
        degree (int): Max degree of polynomials.
        num_vars (int): Number of variables.
        num_samples (int): Number of polynomial instances to generate.
        nn_hidden_layers (list[int]): NN hidden layer configuration.
    """
    print(f"\n--- Running Polynomial Experiment: degree={degree}, vars={num_vars}, samples={num_samples} ---")

    # 1. Generate Data
    print("Generating Polynomial data...")
    poly_coeffs = []
    poly_optima = []
    poly_funcs = []
    generated_count = 0
    failed_count = 0
    while generated_count < num_samples:
        gen_result = generate_polynomial(degree, num_vars)
        if gen_result:
            p_func, p_coeffs, p_opt_x, _ = gen_result
            poly_coeffs.append(p_coeffs)
            poly_optima.append(p_opt_x)
            poly_funcs.append(p_func)
            generated_count += 1
            if generated_count % (num_samples // 10) == 0:
                print(f"Generated {generated_count}/{num_samples} valid polynomial instances...")
        else:
            failed_count += 1
            if failed_count > num_samples * 2: # Avoid infinite loop
                 print("Error: Too many failures generating/solving polynomials.")
                 return

    print(f"Data generation complete. {generated_count} instances.")

    X_data = np.array(poly_coeffs)
    y_data = np.array(poly_optima)

    # Split data
    num_train = int(num_samples * 0.7)
    num_val = int(num_samples * 0.15)
    num_test = num_samples - num_train - num_val

    X_train, X_val, X_test = X_data[:num_train], X_data[num_train:num_train+num_val], X_data[num_train+num_val:]
    y_train, y_val, y_test = y_data[:num_train], y_data[num_train:num_train+num_val], y_data[num_train+num_val:]
    poly_funcs_test = poly_funcs[num_train+num_val:]

    # 2. Build Model
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]
    # Can reuse the same build_nn_model function
    model = build_nn_model(input_dim, output_dim, hidden_layers=nn_hidden_layers)
    model.summary()

    # 3. Train Model
    # Can reuse the same train_nn function
    history = train_nn(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32)

    # 4. Evaluate Model
    eval_results = evaluate_nn_poly(model, X_test, y_test, poly_funcs_test)

    print(f"--- Polynomial Experiment Finished: degree={degree}, vars={num_vars} ---")
    return model, history, eval_results

# Update main guard
if __name__ == "__main__":
    print("This script defines functions for the LP and Polynomial NN projects.")
    print("It does not run the full experiments automatically.")
    print("To run an experiment, call run_experiment() or run_poly_experiment().")

    # Example parameters for Polynomial Experiment (adjust as needed)
    # degree1 = 2 # Quadratic
    # vars1 = 2
    # poly_samples1 = 1000
    # poly_hidden1 = [32, 16]
    # print("\nExample call for Polynomial Experiment 1 (commented out):")
    # print(f"# run_poly_experiment(degree={degree1}, num_vars={vars1}, num_samples={poly_samples1}, nn_hidden_layers={poly_hidden1})")


