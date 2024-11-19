import cvxpy as cp
import numpy as np

# Define your initial float position sizes
float_positions = np.array([2.3, 4.7, 1.9, 3.6, 5.2])  # Example float positions
n = len(float_positions)

# Define the integer variables for the position sizes
integer_positions = cp.Variable(n, integer=True)

# Objective: Minimize tracking error (squared difference from the float positions)
tracking_error = cp.sum_squares(integer_positions - float_positions)
objective = cp.Minimize(tracking_error)

# Constraints
capital_limit = 10  # Example: max total number of contracts
diversity_constraint = 1  # Example: minimum contracts per asset
volatility_limit = 5  # Example: arbitrary volatility constraint

constraints = [
    cp.sum(integer_positions) <= capital_limit,  # Capital constraint
    integer_positions >= diversity_constraint,  # Diversity constraint (e.g., at least 1 contract per asset)
    cp.norm(integer_positions, 2) <= volatility_limit  # Volatility constraint
]

# Define and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve()

# Output the results
print("Optimal Integer Positions:")
print(np.round(integer_positions.value))
print("Tracking Error:", problem.value)
