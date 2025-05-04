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
