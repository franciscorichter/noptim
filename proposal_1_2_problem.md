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
