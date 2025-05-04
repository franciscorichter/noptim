## 1.1. Introduction

This project explores the potential of using trained Neural Networks (NNs) as surrogate optimizers for high-dimensional Linear Programming (LP) problems. Traditional LP solvers, while exact, can be computationally intensive for very large-scale problems. Inspired by the concepts of statistical learning and stochastic programs discussed in the provided reference material (Week13.pdf), this project investigates whether NNs can learn the mapping from LP problem parameters (objective function coefficients, constraint matrix, constraint bounds) to optimal solutions. The core idea is to generate a dataset of diverse LP instances and their corresponding optimal solutions obtained via a standard solver. This dataset will then be used to train NNs. The project aims to define a series of experiments, increasing in complexity, to evaluate the feasibility, accuracy, and generalization capabilities of this NN-based approach. The ultimate goal is to provide a detailed project definition, including methodology, mathematical formulation, experiment code, and a webpage summarizing the findings, serving as a foundation for further research and development in this area.
## 1.2. Problem Definition

The core problem addressed is solving standard form Linear Programming (LP) problems, particularly focusing on high-dimensional instances. A standard form LP problem is defined as:

Minimize: cᵀx
Subject to: Ax = b
            x ≥ 0

Where:
- `x` is the vector of decision variables (dimension `n`).
- `c` is the vector of objective function coefficients (dimension `n`).
- `A` is the constraint matrix (dimension `m x n`).
- `b` is the vector of constraint right-hand side values (dimension `m`).

The challenge arises when `n` (the number of variables) becomes very large (high-dimensional). Traditional algorithms like the Simplex method or interior-point methods, while guaranteed to find the optimal solution, can face significant computational burdens in terms of time and memory as `n` and `m` grow.

This project aims to frame the LP solving process as a learning problem. The input to the learning system (the NN) will be the parameters defining a specific LP instance: the objective vector `c`, the constraint matrix `A`, and the right-hand side vector `b`. The desired output is the optimal solution vector `x*` (or a close approximation thereof).

The specific focus is on generating diverse, high-dimensional LP instances and training an NN to predict `x*` given (`c`, `A`, `b`). The 'program' mentioned by the user refers to this trained NN, which acts as a function mapping LP problem instances to their solutions.
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
## 1.4. Experiment Outline (Increasing Complexity)

The project will involve a series of experiments designed to progressively test the capabilities and limitations of the NN-based LP solver.

1.  **Experiment 1: Fixed Low Dimensions & Simple Structure**
    *   **Goal:** Establish a baseline and verify the basic learning capability.
    *   **Setup:** Generate LP problems with fixed, relatively small dimensions (e.g., n=20, m=10). Use a simple structure for matrix `A` (e.g., dense, well-conditioned).
    *   **NN:** Train a basic MLP architecture.
    *   **Evaluation:** Assess accuracy, feasibility, and optimality gap on a test set of similar problems.

2.  **Experiment 2: Increasing Dimensions**
    *   **Goal:** Evaluate scalability with respect to problem size.
    *   **Setup:** Gradually increase `n` and `m` (e.g., n=50, m=25; n=100, m=50; n=200, m=100), keeping the structure relatively simple.
    *   **NN:** Potentially adjust NN capacity (layers/neurons) as dimensions increase.
    *   **Evaluation:** Track how performance metrics (accuracy, feasibility, gap, inference time) change with increasing dimensions. Compare NN inference time against the traditional solver's time.

3.  **Experiment 3: Varying Sparsity**
    *   **Goal:** Test robustness to different problem structures, common in real-world LPs.
    *   **Setup:** Fix dimensions (e.g., n=100, m=50) but vary the sparsity of the constraint matrix `A` (e.g., 10% non-zero, 5% non-zero, 1% non-zero).
    *   **NN:** Train separate models or a single model on mixed-sparsity data.
    *   **Evaluation:** Analyze performance across different sparsity levels.

4.  **Experiment 4: Exploring NN Architectures & Loss Functions**
    *   **Goal:** Investigate the impact of NN design choices.
    *   **Setup:** Use a fixed problem setting (e.g., n=100, m=50, moderate sparsity).
    *   **NN:** Compare different MLP depths/widths. Experiment with alternative loss functions (e.g., including feasibility penalties).
    *   **Evaluation:** Determine which architectures and loss functions yield the best trade-offs between accuracy, feasibility, and training stability.

5.  **Experiment 5: Generalization Test**
    *   **Goal:** Assess how well a trained model generalizes to LP instances with slightly different statistical properties than the training set.
    *   **Setup:** Train on data generated with one set of parameters (e.g., specific distributions for coefficients) and test on data generated with slightly perturbed parameters.
    *   **NN:** Use the best performing architecture from previous experiments.
    *   **Evaluation:** Measure the drop in performance on the out-of-distribution test set compared to the in-distribution test set.
## 1.5. Deliverables

The primary deliverables for this defined project phase will be:

1.  **Project Proposal Document:** A comprehensive document (this document, compiled) detailing the project's introduction, problem definition, methodology, and experiment outline.
2.  **Mathematical Documentation:** A LaTeX file (`math_docs.tex`) containing the formal mathematical definitions for:
    *   The standard form Linear Programming problem.
    *   A conceptual description of the proposed Neural Network architecture (input/output layers, activation functions).
    *   The primary loss function (e.g., MSE between predicted and true solutions).
3.  **Python Experiment Code:** A Python script (`lp_nn_experiments.py`) containing functions for:
    *   Generating feasible and bounded high-dimensional LP instances (`generate_lp`).
    *   Solving LP instances using a standard solver (`solve_lp`).
    *   Defining the NN model architecture using TensorFlow/Keras (`build_nn_model`).
    *   Training the NN model (`train_nn`).
    *   Evaluating the trained NN model (`evaluate_nn`).
    *   A basic structure to run the outlined experiments (Note: The code will be provided but not executed by the AI agent).
4.  **Static Webpage:** An HTML file (`index.html`) with associated CSS (`style.css`) presenting the project information. This webpage will include:
    *   Sections mirroring the project proposal (Introduction, Problem, Methodology, Experiments).
    *   An overview or link to the mathematical formulations.
    *   Code snippets or a link to the full Python script.
    *   A clear presentation of the project's goals and approach.

These deliverables collectively define the project scope, methodology, and provide the necessary tools (code) to begin the experimental phase.

## 1.6. Additional Problem: Polynomial Optimization

As a second benchmark, this project will also investigate the use of NNs for optimizing polynomial functions. This involves finding the minimum (or maximum) value of a given polynomial function, potentially subject to constraints (though we will start with unconstrained optimization).

**Problem Definition:**
Find $\mathbf{x}^*$ such that $P(\mathbf{x}^*)$ is minimized (or maximized), where $P(\mathbf{x})$ is a polynomial function of variables $\mathbf{x} = (x_1, x_2, ..., x_d)$.

Example (2D): Minimize $P(x_1, x_2) = (x_1 - 2)^2 + (x_2 + 1)^2 + 0.5 x_1 x_2$

**Methodology:**
1.  **Polynomial Generation:** Define a way to generate polynomial functions of varying degrees and number of variables (e.g., by generating coefficients). Start with simple, low-degree polynomials in few variables.
2.  **Data Generation:** Generate training data. This could involve:
    *   Sampling points $\mathbf{x}$ in a domain and evaluating $P(\mathbf{x})$. The NN could learn the function landscape.
    *   Alternatively, for a set of generated polynomials, find their true optima $\mathbf{x}^*$ (analytically or numerically). The NN input would represent the polynomial (e.g., its coefficients), and the output would be the predicted optimum $\mathbf{x}^*_{pred}$. This aligns more closely with the LP approach.
    *   Monte Carlo methods can be used for sampling the domain or potentially for evaluating complex polynomials or finding approximate optima for training data generation.
3.  **NN Architecture:** An MLP similar to the one used for LP can be employed. The input would represent the polynomial (e.g., coefficients), and the output would be the predicted location of the optimum $\mathbf{x}^*_{pred}$.
4.  **Training:** Train the NN using a suitable loss function, such as the MSE between the predicted optimum $\mathbf{x}^*_{pred}$ and the true optimum $\mathbf{x}^*$, or potentially a loss based on the polynomial value at the predicted optimum $P(\mathbf{x}^*_{pred})$.
5.  **Evaluation:** Evaluate the NN's ability to find optima for unseen polynomials. Metrics include the distance between predicted and true optima ($||\mathbf{x}^* - \mathbf{x}^*_{pred}||$) and the difference in the objective function values ($|P(\mathbf{x}^*) - P(\mathbf{x}^*_{pred})|$).

**Experiment Outline (Polynomials):**
1.  **Simple Polynomials:** Start with low-degree (e.g., quadratic) polynomials in 1 or 2 variables.
2.  **Increasing Degree/Variables:** Gradually increase the polynomial degree and the number of variables.
3.  **Complex Landscapes:** Test polynomials known to have multiple local minima/maxima.
4.  **Visualization:** Use plots (e.g., contour plots for 2D) to visualize the polynomial landscape, the true optimum, and the NN's predicted optimum, demonstrating the NN's learning capability.
