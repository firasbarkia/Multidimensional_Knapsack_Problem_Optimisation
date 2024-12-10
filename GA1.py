import numpy as np

def GA1(func, N=30, D=28, Tmax=1000, step=25):
    # Initialisation de la population avec des individus binaires aléatoires
    parents = np.random.randint(0, 2, (N, D))  # Générer N individus, chacun avec D gènes binaires (0 ou 1)
    fitnesses = np.array([func(parent) for parent in parents])  # Calculer la fitness pour chaque individu
    results = []  # Liste pour stocker les meilleurs résultats à chaque étape spécifiée par 'step'

    # Boucle principale sur les générations
    for t in range(Tmax):
        # Calcul de la probabilité de mutation dynamique qui diminue avec les générations
        mutation_rate = 0.1 - (0.1 - 0.01) * (t / Tmax)  # Le taux de mutation commence à 0.1 et diminue au fil des générations
        
        # Sélection de paires de parents pour le crossover
        j, k = np.random.randint(0, N, (2, N))  # Sélectionner N indices de parents aléatoires (2 pour chaque parent)
        cross_point = np.random.randint(1, D - 1)  # Point de croisement aléatoire entre 1 et D-1
        
        # Crossover : combiner les gènes des parents j et k pour générer des enfants
        enfants = np.hstack((parents[j, :cross_point], parents[k, cross_point:]))  # Croiser les parents à la position 'cross_point'

        # Mutation : appliquer la mutation sur les enfants avec une probabilité mutation_rate
        for i in range(N):
            mutation_mask = np.random.rand(D) < mutation_rate  # Créer un masque de mutation aléatoire
            enfants[i] = np.where(mutation_mask, 1 - enfants[i], enfants[i])  # Appliquer la mutation (inverser les bits)

        # Calculer la fitness des enfants
        enfants_fitnesses = np.array([func(enfant) for enfant in enfants])

        # Combiner la population actuelle (parents) avec les nouveaux enfants
        combined_population = np.vstack((parents, enfants))  # Combiner parents et enfants dans une seule population
        combined_fitnesses = np.hstack((fitnesses, enfants_fitnesses))  # Combiner les fitness des parents et enfants

        # Sélectionner les N meilleurs individus (élitisme)
        best_indices = np.argsort(combined_fitnesses)[:N]  # Sélectionner les indices des N meilleurs individus (minimiser la fitness)
        parents = combined_population[best_indices]  # Mettre à jour la population des parents
        fitnesses = combined_fitnesses[best_indices]  # Mettre à jour les valeurs de fitness des parents sélectionnés

        # Enregistrer le meilleur résultat tous les 'step' itérations
        if (t + 1) % step == 0:
            results.append(-fitnesses[0])  # Ajouter la meilleure fitness à la liste (en supposant que plus faible est mieux)

    return results  # Retourner les meilleurs résultats enregistrés
