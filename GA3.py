import numpy as np

def GA3(func, N=30, D=28, Tmax=1000, step=25):
    # Initialisation de la population avec des individus binaires aléatoires
    population = np.random.randint(0, 2, (N, D))  # N individus, chacun de taille D
    fitnesses = np.array([func(individual) for individual in population])  # Calcul de la fitness de chaque individu
    results = []  # Liste pour stocker les meilleurs résultats au fur et à mesure des générations

    # Boucle principale sur chaque génération
    for t in range(Tmax):
        # Calcul des probabilités de sélection des parents basées sur leur fitness
        total_fitness = np.sum(fitnesses)  # Somme des fitness des individus
        if total_fitness == 0:
            # Si la somme des fitness est nulle (éviter la division par zéro), on attribue une probabilité égale
            selection_probs = np.ones(N) / N
        else:
            # Normalisation de la fitness inverse pour gérer les problèmes de minimisation
            selection_probs = fitnesses / total_fitness
        
        # Assurer que selection_probs est une distribution de probabilité valide
        selection_probs = selection_probs.astype(float)  # Convertir en type float
        selection_probs = np.clip(selection_probs, 0, None)  # S'assurer que les probabilités sont non négatives
        selection_probs /= selection_probs.sum()  # Normaliser pour que la somme soit égale à 1

        # Sélection des parents par échantillonnage stochastique
        selected_indices = np.random.choice(np.arange(N), size=N, p=selection_probs)
        parents = population[selected_indices]

        # Crossover : Croisement des parents pour créer des enfants
        crossover_point = np.random.randint(1, D - 1, size=N // 2)  # Points de croisement aléatoires
        children = []  # Liste pour stocker les enfants créés
        for i in range(0, N, 2):
            p1, p2 = parents[i], parents[i + 1]  # Sélectionner deux parents
            cross = crossover_point[i // 2]  # Point de croisement pour cette paire
            child1 = np.hstack((p1[:cross], p2[cross:]))  # Créer le premier enfant en croisant les gènes
            child2 = np.hstack((p2[:cross], p1[cross:]))  # Créer le deuxième enfant
            children.extend([child1, child2])  # Ajouter les enfants à la liste

        children = np.array(children)  # Convertir la liste des enfants en tableau numpy

        # Mutation : Mutation aléatoire des enfants
        mutation_rate = 0.1  # Taux de mutation
        mutation_mask = np.random.rand(N, D) < mutation_rate  # Masque de mutation basé sur une probabilité
        children = np.where(mutation_mask, 1 - children, children)  # Appliquer la mutation en inversant les bits

        # Évaluation des fitness des enfants
        child_fitnesses = np.array([func(child) for child in children])

        # Combiner la population actuelle et les enfants, puis sélectionner les meilleurs individus
        combined_population = np.vstack((population, children))  # Combiner les parents et les enfants
        combined_fitnesses = np.hstack((fitnesses, child_fitnesses))  # Combiner les fitness des parents et des enfants
        best_indices = np.argsort(combined_fitnesses)[:N]  # Sélectionner les N meilleurs individus
        population = combined_population[best_indices]  # Mettre à jour la population avec les meilleurs individus
        fitnesses = combined_fitnesses[best_indices]  # Mettre à jour les fitness des meilleurs individus

        # Sauvegarder le meilleur résultat tous les 'step' itérations
        if (t + 1) % step == 0:
            results.append(-np.min(fitnesses))  # Enregistrer la meilleure fitness (en supposant que la minimisation est recherchée)

    return results  # Retourner les meilleurs résultats à chaque intervalle de 'step'
