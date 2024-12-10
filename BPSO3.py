import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def BPSO3(func, N=30, D=28, Tmax=1000, step=25):
    positions = np.random.randint(2, size=(N, D))
    velocities = np.random.uniform(-6, 6, (N, D))
    personal_best_positions = positions.copy()
    personal_best_costs = np.array([func(pos) for pos in positions])

    global_best_index = np.argmin(personal_best_costs)
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_cost = personal_best_costs[global_best_index]

    results = []

    for iteration in range(Tmax):
        inertia_weight = 0.9 - 0.4 * (iteration / Tmax)  # Décroît avec le temps
        cognitive_factor = 2.0 - 1.5 * (iteration / Tmax)  # Plus d'exploration au début
        social_factor = 0.5 + 1.5 * (iteration / Tmax)  # Plus d'exploitation vers la fin

        r1 = np.random.uniform(0, 1, (N, D))
        r2 = np.random.uniform(0, 1, (N, D))

        cognitive = cognitive_factor * r1 * (personal_best_positions - positions)
        social = social_factor * r2 * (global_best_position - positions)
        velocities = inertia_weight * velocities + cognitive + social

        probabilities = sigmoid(velocities)
        random_values = np.random.uniform(0, 1, (N, D))
        positions = (probabilities >= random_values).astype(int)

        costs = np.array([func(pos) for pos in positions])
        better_mask = costs < personal_best_costs
        personal_best_costs[better_mask] = costs[better_mask]
        personal_best_positions[better_mask] = positions[better_mask].copy()

        new_global_best_index = np.argmin(personal_best_costs)
        if personal_best_costs[new_global_best_index] < global_best_cost:
            global_best_cost = personal_best_costs[new_global_best_index]
            global_best_position = personal_best_positions[new_global_best_index].copy()

        if (iteration + 1) % step == 0:
            results.append(-global_best_cost)

    return results
