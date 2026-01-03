"""
Supply Chain Optimization - Capacitated Plant Location Model

This script optimizes manufacturing plant locations and production allocation
to minimize total costs (fixed + variable) while meeting demand constraints.
"""

import pandas as pd
from pulp import (
    LpProblem,
    LpMinimize,
    LpVariable,
    lpSum,
    LpStatus,
    value,
)


def load_data():
    """Load all input data from Excel files."""
    manvar_costs = pd.read_excel("data/variable_costs.xlsx", index_col=0)
    freight_costs = pd.read_excel("data/freight_costs.xlsx", index_col=0)
    fixed_costs = pd.read_excel("data/fixed_cost.xlsx", index_col=0)
    capacity = pd.read_excel("data/capacity.xlsx", index_col=0)
    demand = pd.read_excel("data/demand.xlsx", index_col=0)

    # Calculate variable costs (freight + manufacturing)
    var_cost = freight_costs / 1000 + manvar_costs

    return var_cost, fixed_costs, capacity, demand


def build_model(var_cost, fixed_costs, capacity, demand):
    """Build the capacitated plant location optimization model."""
    loc = ["USA", "Germany", "Japan", "Brazil", "India"]
    size = ["Low", "High"]

    # Initialize model
    model = LpProblem("Capacitated_Plant_Location_Model", LpMinimize)

    # Decision Variables
    # x: production quantity from location i to location j
    x = LpVariable.dicts(
        "production",
        [(i, j) for i in loc for j in loc],
        lowBound=0,
        upBound=None,
        cat="continuous",
    )

    # y: binary variable for plant type at each location
    y = LpVariable.dicts("plant", [(i, s) for s in size for i in loc], cat="Binary")

    # Objective Function: Minimize total costs (fixed + variable)
    model += lpSum(
        [fixed_costs.loc[i, s] * y[(i, s)] * 1000 for s in size for i in loc]
    ) + lpSum([var_cost.loc[i, j] * x[(i, j)] for i in loc for j in loc])

    # Constraints
    # Demand constraint: total production to each location must meet demand
    for j in loc:
        model += lpSum([x[(i, j)] for i in loc]) == demand.loc[j, "Demand"]

    # Capacity constraint: production from each location limited by plant capacity
    for i in loc:
        model += lpSum([x[(i, j)] for j in loc]) <= lpSum(
            [capacity.loc[i, s] * y[(i, s)] * 1000 for s in size]
        )

    return model, x, y, loc, size


def solve_and_display_results(model, x, y, loc, size):
    """Solve the model and display results."""
    model.solve()

    print("=" * 60)
    print("SUPPLY CHAIN OPTIMIZATION RESULTS")
    print("=" * 60)
    print(f"\nTotal Costs = {int(value(model.objective)):,} ($/Month)")
    print(f"Status: {LpStatus[model.status]}")

    # Extract plant decisions
    print("\n" + "-" * 60)
    print("PLANT DECISIONS")
    print("-" * 60)

    plant_decisions = {"Location": loc, "Low": [], "High": []}
    for location in loc:
        for cap_type in ["Low", "High"]:
            val = int(y[(location, cap_type)].varValue)
            plant_decisions[cap_type].append(val)
            if val == 1:
                print(f"  {location}: {cap_type} Capacity Plant OPEN")

    df_capacity = pd.DataFrame(plant_decisions).set_index("Location")

    # Extract production decisions
    print("\n" + "-" * 60)
    print("PRODUCTION ALLOCATION (units)")
    print("-" * 60)

    production_data = []
    for i in loc:
        row = {"From": i}
        for j in loc:
            qty = x[(i, j)].varValue
            row[j] = int(qty) if qty else 0
        production_data.append(row)

    df_production = pd.DataFrame(production_data).set_index("From")
    print(df_production.to_string())

    return df_capacity, df_production


def main():
    """Main function to run the optimization."""
    # Load data
    var_cost, fixed_costs, capacity, demand = load_data()

    # Build model
    model, x, y, loc, size = build_model(var_cost, fixed_costs, capacity, demand)

    # Solve and display results
    df_capacity, df_production = solve_and_display_results(model, x, y, loc, size)

    return df_capacity, df_production


if __name__ == "__main__":
    main()
