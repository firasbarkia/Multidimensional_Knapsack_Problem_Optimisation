import numpy as np

def GA2(func, N=30, D=28, Tmax=1000, step=25, local_iters=5):
    # Initialize population
    parents = np.random.randint(0, 2, (N, D))
    fitnesses = np.array([func(parent) for parent in parents])
    results = []

    for t in range(Tmax):
        # Dynamic mutation rate
        mutation_rate = 0.1 - (0.1 - 0.01) * (t / Tmax)

        # Crossover: Random pairing
        j = np.random.permutation(N)
        k = np.random.permutation(N)
        cross_point = np.random.randint(1, D - 1)
        enfants = np.hstack((parents[j, :cross_point], parents[k, cross_point:]))

        # Mutation
        mutation_mask = np.random.rand(*enfants.shape) < mutation_rate
        enfants = (enfants + mutation_mask) % 2  # Flip bits with mutation_mask

        # Evaluate fitness of children
        enfants_fitnesses = np.array([func(enfant) for enfant in enfants])

        # Combine parents and children
        combined_population = np.vstack((parents, enfants))
        combined_fitnesses = np.hstack((fitnesses, enfants_fitnesses))

        # Select top N individuals (elitism)
        best_indices = np.argsort(combined_fitnesses)[:N]
        parents = combined_population[best_indices]
        fitnesses = combined_fitnesses[best_indices]

        # Local search on the top individuals
        for idx in best_indices:
            individual = combined_population[idx]
            original_fitness = combined_fitnesses[idx]
            for _ in range(local_iters):
                # Randomly flip a single bit
                local_individual = individual.copy()
                random_bit = np.random.randint(0, D)
                local_individual[random_bit] = 1 - local_individual[random_bit]

                # Evaluate local fitness
                local_fitness = func(local_individual)
                if local_fitness < original_fitness:  # Minimize cost
                    combined_population[idx] = local_individual
                    combined_fitnesses[idx] = local_fitness
                    original_fitness = local_fitness  # Update the best fitness for this individual

        # Update parents and fitnesses after local search
        best_indices = np.argsort(combined_fitnesses)[:N]
        parents = combined_population[best_indices]
        fitnesses = combined_fitnesses[best_indices]

        # Log the best result every step iterations
        if (t + 1) % step == 0:
            results.append(-fitnesses[0])  # Assuming lower cost is better

    return results
