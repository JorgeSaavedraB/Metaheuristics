import numpy as np
import random
import copy

## Selection
### Tournament Operator
def tournament(list_pop: list, fitness_funct, tournament_size, n_tournaments, minimize=False):
    selected = []
    for _ in range(n_tournaments):
        rand_inds = random.sample(list_pop, tournament_size)
        best_ind = rand_inds[0]
        fitness_val = fitness_funct(best_ind)
        for ind in rand_inds[1:]:
            new_fitness = fitness_funct(ind)
            if minimize:
                if new_fitness <= fitness_val:
                    best_ind = ind
                    fitness_val = new_fitness
            else:
                if new_fitness >= fitness_val:
                    best_ind = ind
                    fitness_val = new_fitness
        selected.append(best_ind)
    return selected

### Stochastic Universal Selection
def SUS(list_pop: list, fitness_funct, k, minimize=False):
    fitness_list = [fitness_funct(ind) for ind in list_pop]

     # Si estamos minimizando, invertimos los valores
    if minimize:
        max_fitness = max(fitness_list)
        fitness_list = [max_fitness - fit for fit in fitness_list]
    
    # Evita división entre cero
    total_fitness = np.sum(fitness_list)
    if total_fitness == 0:
        probs_list = np.array([1/len(fitness_list) for _ in fitness_list])
    else:
        probs_list = np.array(fitness_list)/total_fitness
    
    # Probabilidades acumuladas
    cum_probs = np.cumsum(probs_list)
    
    # Iniciar desde un punto aleatorio en [0, 1/k)
    start = random.uniform(0, 1 / k)
    pointers = [start + i / k for i in range(k)]
    
    sols = []
    idx = 0
    for p in pointers:
        while p > cum_probs[idx]:
            idx += 1
        sols.append(list_pop[idx])
    
    return sols

### Proportional sampling
def RWS(list_pop: list, fitness_funct, k, minimize=False):
    fitness_list = [fitness_funct(ind) for ind in list_pop]

    # Si estamos minimizando, invertimos los valores
    if minimize:
        max_fitness = max(fitness_list)
        fitness_list = [max_fitness - fit for fit in fitness_list]

    # Evita división entre cero
    total_fitness = np.sum(fitness_list)
    if total_fitness == 0:
        probs_list = np.array([1/len(fitness_list) for _ in fitness_list])
    else:
        probs_list = np.array(fitness_list)/total_fitness
    
    # Probabilidades acumuladas
    cum_probs = np.cumsum(probs_list)
    
    pointers = [random.uniform(0, 1) for i in range(k)]
    
    sols = []
    for p in pointers:
        idx = 0
        while p > cum_probs[idx]:
            idx += 1
        sols.append(list_pop[idx])
    
    return sols

## Crossover
### One point crossover
def one_point_crossover(ind_1: list, ind_2: list):
    c1, c2 = ind_1.copy(), ind_2.copy()
    n = len(ind_1)
    if n <= 2:
        return c1, c2
    point = random.randint(1, n-1)
    for i in range(point):
        c1[i], c2[i] = c2[i], c1[i]
    return c1, c2

### Two points crossover
def two_point_crossover(ind_1: list, ind_2: list):
    c1, c2 = ind_1.copy(), ind_2.copy()
    n = len(ind_1)
    if n <= 2:
        return c1, c2
    point1 = random.randint(1, n-2)
    point2 = random.randint(point1 + 1, n-1)
    for i in range(point1, point2):
        c1[i], c2[i] = c2[i], c1[i]
    return c1, c2

### Uniform crossover
def uniform_crossover(ind_1: list, ind_2: list):
    c1, c2 = ind_1.copy(), ind_2.copy()
    n = len(ind_1)
    for i in range(n):
        if random.random() <= 0.5:
            c1[i], c2[i] = c2[i], c1[i]
    return c1, c2

## Replacement
### Random based replacements
def random_based_replacement(parents, offspring):
    n_offspring = len(offspring)
    n_parents = len(parents)
    new_pop = []
    for i in range(n_parents):
        if i < n_offspring and random.random() > 0.5:
            new_pop.append(offspring[i])
        else:
            new_pop.append(parents[i])
    return new_pop

### Elitism replacement
def elitism_replacement(parents, 
                        offspring, 
                        fitness_funct, 
                        num_parents_selected, 
                        minimize=False):
    n = len(parents)
    if num_parents_selected > n:
        raise ValueError("num_parents_selected cannot exceed population size")
        
    parents_fit = [fitness_funct(ind) for ind in parents]
    top_parent_indices = [idx for _, idx in sorted([(val, idx) for idx, val in enumerate(parents_fit)], reverse=not minimize)[:num_parents_selected]]
    
    offspring_fit = [fitness_funct(ind) for ind in offspring]
    top_offspring_indices = [idx for _, idx in sorted([(val, idx) for idx, val in enumerate(offspring_fit)], reverse=not minimize)[:n - num_parents_selected]]
    
    new_pop = [parents[i] for i in top_parent_indices] + [offspring[i] for i in top_offspring_indices]
        
    return new_pop

### Tournament similar replacement
def tournament_similar_replacement(parents, 
                                   offspring, 
                                   fitness_funct, 
                                   minimize=False):
    new_pop = []
    offspring_used = []

    for parent in parents:
        closest_idx = None
        closest_dist = np.inf

        for i, child in enumerate(offspring):
            if i in offspring_used:
                continue
            eucl_dist = np.linalg.norm(np.array(parent) - np.array(child))
            if eucl_dist <= closest_dist:
                closest_idx = i
                closest_dist = eucl_dist
        if closest_idx is not None:
            offspring_used .append(closest_idx)
            fitness_parent = fitness_funct(parent)
            fitness_child = fitness_funct(offspring[closest_idx])

            if (minimize and fitness_parent <= fitness_child) or (not minimize and fitness_parent >= fitness_child):
                new_pop.append(parent)
            else:
                new_pop.append(offspring[closest_idx])
            
        else:
            new_pop.append(parent)

    return new_pop