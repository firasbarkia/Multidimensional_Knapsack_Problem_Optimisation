import numpy as np

# Fonction sigmoïde utilisée pour transformer les vitesses en probabilités binaires
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def BPSO1(func, N=30, D=28, Tmax=1000, step=25):
    # Initialisation des positions binaires des particules
    positions = np.random.randint(2, size=(N, D))  # Positions aléatoires binaires (0 ou 1)
    # Initialisation des vitesses des particules, avec des valeurs continues entre -6 et 6
    velocities = np.random.uniform(-6, 6, (N, D))  
    # Enregistrer les meilleures positions et coûts pour chaque particule
    personal_best_positions = positions.copy()  
    personal_best_costs = np.array([func(pos) for pos in positions])  # Calcul des coûts pour chaque particule

    # Initialisation de la meilleure position globale
    global_best_index = np.argmin(personal_best_costs)  # Trouver la particule avec le meilleur coût
    global_best_position = personal_best_positions[global_best_index].copy()  # Position associée au meilleur coût
    global_best_cost = personal_best_costs[global_best_index]  # Coût associé à la meilleure position globale

    results = []  # Liste pour stocker les résultats à chaque itération

    # Boucle principale de l'algorithme
    for iteration in range(Tmax):
        # Calcul du poids d'inertie, qui diminue au fur et à mesure des itérations
        inertia_weight = 0.9 - 0.5 * iteration / Tmax
        
        # Génération de valeurs aléatoires r1 et r2 pour la mise à jour des vitesses
        r1 = np.random.uniform(0, 1, (N, D))  # Valeurs aléatoires pour la partie cognitive
        r2 = np.random.uniform(0, 1, (N, D))  # Valeurs aléatoires pour la partie sociale

        # Calcul des termes cognitifs et sociaux
        cognitive = 1.5 * r1 * (personal_best_positions - positions)  # Influence de la meilleure position personnelle
        social = 1.5 * r2 * (global_best_position - positions)  # Influence de la meilleure position globale

        # Mise à jour des vitesses des particules
        velocities = inertia_weight * velocities + cognitive + social

        # Transformation des vitesses en probabilités binaires
        probabilities = sigmoid(velocities)

        # Génération de valeurs aléatoires pour comparer aux probabilités et déterminer la nouvelle position
        random_values = np.random.uniform(0, 1, (N, D))
        positions = (probabilities >= random_values).astype(int)  # Mise à jour des positions binaires (0 ou 1)

        # Calcul des coûts pour les nouvelles positions
        costs = np.array([func(pos) for pos in positions])

        # Mise à jour des meilleures positions personnelles
        better_mask = costs < personal_best_costs  # Masque indiquant quelles particules ont amélioré leur coût
        personal_best_costs[better_mask] = costs[better_mask]
        personal_best_positions[better_mask] = positions[better_mask].copy()

        # Mise à jour de la meilleure position globale
        new_global_best_index = np.argmin(personal_best_costs)
        if personal_best_costs[new_global_best_index] < global_best_cost:
            global_best_cost = personal_best_costs[new_global_best_index]
            global_best_position = personal_best_positions[new_global_best_index].copy()

        # Enregistrer le meilleur coût global tous les 'step' itérations
        if (iteration + 1) % step == 0:
            results.append(-global_best_cost)  # Enregistrer la valeur du meilleur coût global (négatif car on minimise)

    return results  # Retourner la liste des résultats
