import numpy as np

def GA2(func, N=30, D=28, Tmax=1000, step=25, local_iters=5):
    # Initialiser la population avec des individus binaires aléatoires
    parents = np.random.randint(0, 2, (N, D))  # N individus, chacun de taille D
    fitnesses = np.array([func(parent) for parent in parents])  # Calculer les fitness des individus
    results = []  # Liste pour stocker les meilleurs résultats au fur et à mesure des générations

    # Boucle sur chaque génération
    for t in range(Tmax):
        # Calcul de la probabilité de mutation dynamique
        mutation_rate = 0.1 - (0.1 - 0.01) * (t / Tmax)  # La mutation diminue avec les générations

        # Opération de croisement (crossover) : appariement aléatoire de la population
        j = np.random.permutation(N)  # Permutation aléatoire des indices de parents
        k = np.random.permutation(N)  # Permutation aléatoire supplémentaire pour obtenir des paires distinctes
        cross_point = np.random.randint(1, D - 1)  # Choisir un point de croisement aléatoire
        # Créer les enfants en échangeant les gènes entre les deux parents à partir du point de croisement
        enfants = np.hstack((parents[j, :cross_point], parents[k, cross_point:]))

        # Mutation : Appliquer la mutation sur les enfants avec une probabilité mutation_rate
        mutation_mask = np.random.rand(*enfants.shape) < mutation_rate  # Créer un masque de mutation aléatoire
        enfants = (enfants + mutation_mask) % 2  # Flipper les bits des enfants avec le masque de mutation

        # Évaluer les fitness des enfants
        enfants_fitnesses = np.array([func(enfant) for enfant in enfants])

        # Combiner les parents et les enfants dans une seule population
        combined_population = np.vstack((parents, enfants))
        combined_fitnesses = np.hstack((fitnesses, enfants_fitnesses))

        # Sélectionner les N meilleurs individus (élitisme) en fonction de la fitness
        best_indices = np.argsort(combined_fitnesses)[:N]  # Indices des N meilleurs individus
        parents = combined_population[best_indices]  # Mettre à jour la population des parents
        fitnesses = combined_fitnesses[best_indices]  # Mettre à jour les valeurs de fitness

        # Recherche locale sur les meilleurs individus pour améliorer la solution
        for idx in best_indices:
            individual = combined_population[idx]  # Sélectionner un individu
            original_fitness = combined_fitnesses[idx]  # Enregistrer sa fitness initiale
            for _ in range(local_iters):  # Effectuer plusieurs itérations de recherche locale
                # Créer une copie de l'individu pour effectuer une modification
                local_individual = individual.copy()
                random_bit = np.random.randint(0, D)  # Choisir un bit aléatoire à inverser
                local_individual[random_bit] = 1 - local_individual[random_bit]  # Inverser ce bit

                # Calculer la fitness de l'individu modifié
                local_fitness = func(local_individual)
                # Si la fitness locale est meilleure (coût minimisé), mettre à jour l'individu
                if local_fitness < original_fitness:
                    combined_population[idx] = local_individual
                    combined_fitnesses[idx] = local_fitness
                    original_fitness = local_fitness  # Mettre à jour la meilleure fitness pour cet individu

        # Mise à jour finale des parents et des fitness après la recherche locale
        best_indices = np.argsort(combined_fitnesses)[:N]
        parents = combined_population[best_indices]
        fitnesses = combined_fitnesses[best_indices]

        # Enregistrer le meilleur résultat tous les 'step' itérations
        if (t + 1) % step == 0:
            results.append(-fitnesses[0])  # Supposer que plus la fitness est faible, mieux c'est (minimisation)

    return results  # Retourner les résultats à chaque intervalle de 'step' générations
