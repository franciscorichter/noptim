## 1.3. Methodology

The proposed methodology involves several key steps:

1.  **Data Generation:**
    *   Generate a large dataset of diverse, high-dimensional standard-form LP problems (`c`, `A`, `b`).
    *   Parameters like the dimensions (`m`, `n`), the sparsity of `A`, and the distribution of coefficients in `c`, `A`, and `b` will be varied to ensure diversity.
    *   Crucially, generated problems must be feasible and bounded. Techniques to ensure this (e.g., specific structures for `A`, careful generation of `b`) will be employed.
    *   For each generated LP instance, a reliable traditional LP solver (e.g., `scipy.optimize.linprog` with the 'highs' method) will be used to compute the ground truth optimal solution `x*` and the optimal objective value `z* = cᵀx*`.
    *   The dataset will consist of tuples: `(c, A, b) -> x*`.

2.  **Input/Output Representation:**
    *   The input to the NN will be a flattened representation of the LP parameters (`c`, `A`, `b`). For fixed dimensions `m` and `n`, this results in a fixed-size input vector.
    *   The output of the NN will be a vector of the same dimension as `x` (`n`), representing the predicted optimal solution `x_pred`.

3.  **Neural Network Architecture:**
    *   A Multi-Layer Perceptron (MLP) architecture is proposed as a starting point due to its general function approximation capabilities.
    *   The input layer size will match the flattened dimension of (`c`, `A`, `b`).
    *   The output layer size will match the dimension of `x` (`n`), likely using a linear or ReLU activation to allow for non-negative outputs (since `x ≥ 0`).
    *   Several hidden layers with non-linear activation functions (e.g., ReLU) will be used. The exact number of layers and neurons per layer will be hyperparameters to tune during experimentation.

4.  **Training:**
    *   The NN will be trained in a supervised manner using the generated dataset.
    *   The primary loss function will measure the discrepancy between the predicted solution `x_pred` and the true optimal solution `x*`. Mean Squared Error (MSE) is a common choice: `Loss = ||x* - x_pred||₂²`.
    *   Alternative or supplementary loss terms could be considered, such as penalizing constraint violations (`||Ax_pred - b||₂²`) or the difference in objective values (`|cᵀx_pred - cᵀx*|`).
    *   Standard optimization algorithms like Adam or RMSprop will be used.
    *   The dataset will be split into training, validation, and test sets to monitor performance and prevent overfitting.

5.  **Evaluation:**
    *   The performance of the trained NN will be evaluated on the unseen test set using several metrics:
        *   **Solution Accuracy:** Distance between predicted and true solutions (e.g., Mean Absolute Error or MSE).
        *   **Feasibility:** How well the predicted solution `x_pred` satisfies the constraints (`Ax = b`, `x ≥ 0`). This might involve measuring the magnitude of constraint violations.
        *   **Optimality Gap:** The difference between the objective function value achieved by the predicted solution (`cᵀx_pred`) and the true optimal value (`cᵀx*`).
        *   **Inference Time:** Compare the time taken by the NN to predict a solution versus the time taken by the traditional solver.
