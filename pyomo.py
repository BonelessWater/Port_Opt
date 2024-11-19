import pyomo.environ as pyo
import numpy as np

# Initialize example values
x_b = np.array([10, 5, 8])  # Target positions
cov_matrix = np.array([[0.02, 0.01, 0.015], [0.01, 0.03, 0.02], [0.015, 0.02, 0.025]])
prices = np.array([5000, 2000, 3000])
capital = 100000
U = [20, 10, 15]  # Upper bounds on contracts
max_volatility = 0.15  # Volatility constraint

# Define the Pyomo model
model = pyo.ConcreteModel()

# Number of assets
n_assets = len(x_b)

# Define decision variables (integer values for contracts)
model.x = pyo.Var(range(n_assets), domain=pyo.Integers, bounds=(0, None))

# Objective function for tracking error
def objective_function(model):
    diff = np.array([model.x[i] - x_b[i] for i in range(n_assets)])
    return pyo.sqrt(sum(diff[i] * cov_matrix[i, j] * diff[j] for i in range(n_assets) for j in range(n_assets)))

model.objective = pyo.Objective(rule=objective_function, sense=pyo.minimize)

# Capital constraint
def capital_constraint(model):
    return sum(prices[i] * model.x[i] for i in range(n_assets)) <= capital

model.capital_constraint = pyo.Constraint(rule=capital_constraint)

# Volatility constraint
def volatility_constraint(model):
    return pyo.sqrt(sum(model.x[i] * cov_matrix[i, j] * model.x[j] for i in range(n_assets) for j in range(n_assets))) <= max_volatility

model.volatility_constraint = pyo.Constraint(rule=volatility_constraint)

# Upper bounds for each asset
for i in range(n_assets):
    model.add_component(f'bound_{i}', pyo.Constraint(expr=model.x[i] <= U[i]))

# Solve the model
solver = pyo.SolverFactory('glpk')  # Use 'cbc' if you have installed CBC solver
result = solver.solve(model)

# Output the results
optimal_solution = [pyo.value(model.x[i]) for i in range(n_assets)]
objective_value = pyo.value(model.objective)

print("Optimal Solution (Integer Contracts):", optimal_solution)
print("Objective Value:", objective_value)
print("Solver Status:", result.solver.status)
print("Solver Termination Condition:", result.solver.termination_condition)

