import numpy as np

def GA3(func, N=30, D=28, Tmax=1000, step=25):
    # Initial random population
    population = np.random.randint(0, 2, (N, D))
    fitnesses = np.array([func(individual) for individual in population])
    results = []

    for t in range(Tmax):
        # Calculate selection probabilities
        total_fitness = np.sum(fitnesses)
        if total_fitness == 0:
            # Avoid division by zero by assigning equal probabilities
            selection_probs = np.ones(N) / N
        else:
            # Use normalized inverse fitness to handle minimization problems
            selection_probs = fitnesses / total_fitness
        
        # Ensure selection_probs is a valid probability distribution
        selection_probs = selection_probs.astype(float)  # Cast to float
        selection_probs = np.clip(selection_probs, 0, None)  # Ensure non-negative
        selection_probs /= selection_probs.sum()  # Normalize to sum to 1

        # Select parents
        selected_indices = np.random.choice(np.arange(N), size=N, p=selection_probs)
        parents = population[selected_indices]

        # Crossover
        crossover_point = np.random.randint(1, D - 1, size=N // 2)
        children = []
        for i in range(0, N, 2):
            p1, p2 = parents[i], parents[i + 1]
            cross = crossover_point[i // 2]
            child1 = np.hstack((p1[:cross], p2[cross:]))
            child2 = np.hstack((p2[:cross], p1[cross:]))
            children.extend([child1, child2])

        children = np.array(children)

        # Mutation
        mutation_rate = 0.1
        mutation_mask = np.random.rand(N, D) < mutation_rate
        children = np.where(mutation_mask, 1 - children, children)

        # Evaluate children
        child_fitnesses = np.array([func(child) for child in children])

        # Combine population and select best individuals
        combined_population = np.vstack((population, children))
        combined_fitnesses = np.hstack((fitnesses, child_fitnesses))
        best_indices = np.argsort(combined_fitnesses)[:N]
        population = combined_population[best_indices]
        fitnesses = combined_fitnesses[best_indices]

        # Save best result every 'step' iterations
        if (t + 1) % step == 0:
            results.append(-np.min(fitnesses))

    return results
