# ga_with_local_search.py
import numpy as np

def GA2(func, N=30, D=28, Tmax=1000, step=25):
    parents = np.random.randint(0, 2, (N, D))
    fitnesses = np.array([func(parent) for parent in parents])
    results = []

    for t in range(Tmax):
        mutation_rate = 0.1 - (0.1 - 0.01) * (t / Tmax)
        j, k = np.random.randint(0, N, (2, N))
        cross_point = np.random.randint(1, D - 1)
        enfants = np.hstack((parents[j, :cross_point], parents[k, cross_point:]))

        for i in range(N):
            mutation_mask = np.random.rand(D) < mutation_rate
            enfants[i] = np.where(mutation_mask, 1 - enfants[i], enfants[i])

        enfants_fitnesses = np.array([func(enfant) for enfant in enfants])
        combined_population = np.vstack((parents, enfants))
        combined_fitnesses = np.hstack((fitnesses, enfants_fitnesses))

        best_indices = np.argsort(combined_fitnesses)[:N]
        parents = combined_population[best_indices]
        fitnesses = combined_fitnesses[best_indices]

        # Local search on the best individuals
        for idx in best_indices:
            individual = combined_population[idx]
            for _ in range(5):
                local_individual = individual.copy()
                random_bit = np.random.randint(0, D)
                local_individual[random_bit] = 1 - local_individual[random_bit]
                local_cost = func(local_individual)
                if local_cost < combined_fitnesses[idx]:
                    combined_population[idx] = local_individual
                    combined_fitnesses[idx] = local_cost

        if (t + 1) % step == 0:
            results.append(-fitnesses[0])

    return results