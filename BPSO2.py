import numpy as np

# Fonction sigmoïde pour mettre à jour les positions binaires des particules
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def BPSO2(func, N=30, D=28, Tmax=1000, tabu_size=50, step=25):
    # Initialisation des positions et vitesses des particules
    positions = np.random.randint(2, size=(N, D))  # Positions binaires des particules
    velocities = np.random.uniform(-6, 6, (N, D))  # Vitesses initiales des particules
    personal_best_positions = positions.copy()  # Copies des meilleures positions de chaque particule
    personal_best_costs = np.array([func(pos) for pos in positions])  # Calcul des coûts (fitness) pour chaque particule

    # Initialisation de la meilleure position globale
    global_best_index = np.argmin(personal_best_costs)  # Trouver l'indice de la meilleure particule
    global_best_position = personal_best_positions[global_best_index].copy()  # Meilleure position globale
    global_best_cost = personal_best_costs[global_best_index]  # Meilleur coût global

    # Initialisation de l'ensemble tabou (ensemble pour stocker les solutions déjà explorées)
    tabu_set = set()
    results = []  # Liste pour enregistrer les meilleurs résultats au fil des itérations

    # Fonction pour vérifier si une position est dans le tabou
    def is_tabu(position):
        return tuple(position) in tabu_set

    # Fonction pour ajouter une position au tabou et maintenir la taille de l'ensemble
    def add_to_tabu(position):
        tabu_set.add(tuple(position))
        if len(tabu_set) > tabu_size:  # Si la taille du tabou dépasse la taille maximale
            tabu_set.pop()  # Supprimer la plus ancienne position du tabou

    # Boucle principale de l'algorithme
    for iteration in range(Tmax):
        # Calcul de l'inertie, qui diminue avec les générations
        inertia_weight = 0.9 - 0.5 * iteration / Tmax
        
        # Générer des valeurs aléatoires r1 et r2 pour la mise à jour des vitesses
        r1 = np.random.uniform(0, 1, (N, D))  # Valeurs aléatoires pour la partie cognitive
        r2 = np.random.uniform(0, 1, (N, D))  # Valeurs aléatoires pour la partie sociale

        # Calcul des termes cognitifs et sociaux
        cognitive = 1.5 * r1 * (personal_best_positions - positions)
        social = 1.5 * r2 * (global_best_position - positions)

        # Mise à jour des vitesses des particules
        velocities = inertia_weight * velocities + cognitive + social

        # Calcul des probabilités pour chaque position binaire
        probabilities = sigmoid(velocities)

        # Mettre à jour les positions en fonction des probabilités
        random_values = np.random.uniform(0, 1, (N, D))  # Générer des valeurs aléatoires pour comparer aux probabilités
        positions = (probabilities >= random_values).astype(int)  # Mise à jour des positions binaires

        # Vérifier si la position est dans le tabou et appliquer une mutation si c'est le cas
        for i in range(N):
            if is_tabu(positions[i]):  # Si la position est tabou
                # Appliquer une mutation aléatoire avec probabilité de 0.1
                mutation = np.random.choice([-1, 1], size=D) * (np.random.rand(D) < 0.1)
                positions[i] = np.clip(positions[i] + mutation, 0, 1).astype(int)  # Mettre à jour la position en respectant les bornes

        # Calculer les coûts pour les nouvelles positions
        costs = np.array([func(pos) for pos in positions])

        # Mettre à jour les meilleures positions personnelles
        better_mask = costs < personal_best_costs  # Mask des meilleures positions
        personal_best_costs[better_mask] = costs[better_mask]
        personal_best_positions[better_mask] = positions[better_mask].copy()

        # Mise à jour de la meilleure position globale si nécessaire
        new_global_best_index = np.argmin(personal_best_costs)
        if personal_best_costs[new_global_best_index] < global_best_cost:
            global_best_cost = personal_best_costs[new_global_best_index]
            global_best_position = personal_best_positions[new_global_best_index].copy()

        # Ajouter la meilleure position globale à l'ensemble tabou
        add_to_tabu(global_best_position)

        # Enregistrer le meilleur coût global tous les 'step' itérations
        if (iteration + 1) % step == 0:
            results.append(-global_best_cost)  # Inverser le coût car on minimise

    return results  # Retourner les résultats de l'algorithme
