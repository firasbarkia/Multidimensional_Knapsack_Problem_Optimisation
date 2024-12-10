import numpy as np

# Fonction sigmoïde utilisée pour la mise à jour des positions binaires des particules
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def BPSO3(func, N=30, D=28, Tmax=1000, step=25):
    # Initialisation aléatoire des positions et vitesses des particules
    positions = np.random.randint(2, size=(N, D))  # Positions binaires (0 ou 1) pour N particules
    velocities = np.random.uniform(-6, 6, (N, D))  # Vitesses initiales aléatoires des particules
    personal_best_positions = positions.copy()  # Copier les positions initiales des particules
    personal_best_costs = np.array([func(pos) for pos in positions])  # Calculer les coûts (fitness) pour chaque particule

    # Initialiser la meilleure position globale avec la meilleure particule
    global_best_index = np.argmin(personal_best_costs)  # Trouver l'indice de la meilleure particule
    global_best_position = personal_best_positions[global_best_index].copy()  # Meilleure position globale
    global_best_cost = personal_best_costs[global_best_index]  # Meilleur coût global

    results = []  # Liste pour stocker les meilleurs résultats à chaque itération

    # Boucle sur les itérations (générations)
    for iteration in range(Tmax):
        # Calcul des facteurs d'inertie, cognitif et social, qui varient au fil du temps
        inertia_weight = 0.9 - 0.4 * (iteration / Tmax)  # Poids d'inertie décroissant
        cognitive_factor = 2.0 - 1.5 * (iteration / Tmax)  # Facteur cognitif qui favorise l'exploration au début
        social_factor = 0.5 + 1.5 * (iteration / Tmax)  # Facteur social qui favorise l'exploitation vers la fin

        # Générer des valeurs aléatoires r1 et r2 pour la mise à jour des vitesses
        r1 = np.random.uniform(0, 1, (N, D))  # Valeurs aléatoires pour la partie cognitive
        r2 = np.random.uniform(0, 1, (N, D))  # Valeurs aléatoires pour la partie sociale

        # Calcul des termes cognitifs et sociaux pour chaque particule
        cognitive = cognitive_factor * r1 * (personal_best_positions - positions)
        social = social_factor * r2 * (global_best_position - positions)

        # Mise à jour des vitesses des particules
        velocities = inertia_weight * velocities + cognitive + social

        # Calcul des probabilités pour chaque position binaire en fonction des vitesses
        probabilities = sigmoid(velocities)

        # Générer des valeurs aléatoires pour chaque position et décider si le bit doit être 0 ou 1
        random_values = np.random.uniform(0, 1, (N, D))
        positions = (probabilities >= random_values).astype(int)  # Mise à jour des positions binaires

        # Calculer les coûts (fitness) des nouvelles positions
        costs = np.array([func(pos) for pos in positions])

        # Mettre à jour les meilleures positions personnelles si la nouvelle position est meilleure
        better_mask = costs < personal_best_costs
        personal_best_costs[better_mask] = costs[better_mask]
        personal_best_positions[better_mask] = positions[better_mask].copy()

        # Mise à jour de la meilleure position globale si une meilleure solution est trouvée
        new_global_best_index = np.argmin(personal_best_costs)
        if personal_best_costs[new_global_best_index] < global_best_cost:
            global_best_cost = personal_best_costs[new_global_best_index]
            global_best_position = personal_best_positions[new_global_best_index].copy()

        # Sauvegarder la meilleure performance globale tous les 'step' itérations
        if (iteration + 1) % step == 0:
            results.append(-global_best_cost)  # Ajouter le coût de la meilleure solution globale (inversé)

    return results  # Retourner les résultats à chaque 'step' itérations
