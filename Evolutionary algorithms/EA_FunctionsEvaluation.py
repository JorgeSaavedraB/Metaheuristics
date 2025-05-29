import numpy as np
import time 
import random
import copy
from EA_Operators import tournament, SUS, RWS # Selection operators
from EA_Operators import one_point_crossover, two_point_crossover, uniform_crossover # Crossover operators
from EA_Operators import random_based_replacement, elitism_replacement, tournament_similar_replacement # Replacements operators
from cec2017.basic import bent_cigar, sum_diff_pow, zakharov, rosenbrock, rastrigin, expanded_schaffers_f6, lunacek_bi_rastrigin, non_cont_rastrigin, levy, modified_schwefel, high_conditioned_elliptic

# Mutation
def mutation(ind, 
             mutation_rate=0.1):
    mutant = ind.copy()
    for i in range(len(mutant)):
        if random.random() < mutation_rate:
            mutant[i] = np.clip(mutant[i] + np.random.normal(0, 1), -100, 100)
    return mutant

# Functions
functs = [bent_cigar, zakharov, rosenbrock, rastrigin, expanded_schaffers_f6, lunacek_bi_rastrigin, non_cont_rastrigin, levy, modified_schwefel, high_conditioned_elliptic]

# Evolution algorithms
def EA1(parents, 
        fitness_funct, 
        mutation_funct, 
        num_generations, 
        num_evaluations, 
        size_offspring, 
        mutation_rate, 
        minimize=False):
    fitness_cache = {}
    eval_counter = {'count': 0}
    
    def counted_fitness(ind):
        key = tuple(ind.flatten())  # O cualquier forma de hash válida
        if key not in fitness_cache:
            fitness_cache[key] = fitness_funct(ind)
            eval_counter['count'] += 1
        return fitness_cache[key][0]

    for _ in range(num_generations):
        # Selección
        selection = tournament(parents, counted_fitness, 2, size_offspring, minimize=minimize)
        # Cruza y mutación
        offspring = []
        for i in range(0, size_offspring, 2):
            p1 = selection[i % len(selection)][0]
            p2 = selection[(i + 1) % len(selection)][0]
            crossover = two_point_crossover(p1, p2)
            for ind in crossover:
                offspring.append(np.array([mutation_funct(ind, mutation_rate)])) # mutación de los individuos
        offspring = offspring[:size_offspring]  # Limitar a tamaño de población
        # Reeplazo
        parents = random_based_replacement(parents, offspring)

        if eval_counter['count'] >= num_evaluations:
            break

    final_fitness = [fitness_funct(ind)[0] for ind in parents]
    best = parents[np.argmin(final_fitness)] # Mejor individuo final
    
    return best, fitness_funct(best)[0]

def EA2(parents, 
        fitness_funct, 
        mutation_funct, 
        num_generations, 
        num_evaluations, 
        size_offspring, 
        mutation_rate, 
        num_arrows, 
        minimize=False):
    fitness_cache = {}
    eval_counter = {'count': 0}
    
    def counted_fitness(ind):
        key = tuple(ind.flatten())  # O cualquier forma de hash válida
        if key not in fitness_cache:
            fitness_cache[key] = fitness_funct(ind)
            eval_counter['count'] += 1
        return fitness_cache[key][0]
    
    popsize = len(parents)
    for _ in range(num_generations):
        # Selección
        selection = RWS(parents, counted_fitness, num_arrows, minimize=minimize)
        # Cruza y mutación
        offspring = []
        for i in range(0, size_offspring, 2):
            p1 = selection[i % len(selection)][0]
            p2 = selection[(i + 1) % len(selection)][0]
            crossover = one_point_crossover(p1, p2)
            for ind in crossover:
                offspring.append(np.array([mutation_funct(ind, mutation_rate)])) # mutación de los individuos
        offspring = offspring[:size_offspring]  # Limitar a tamaño de población
        # Reemplazo
        parents = elitism_replacement(parents, offspring, counted_fitness, popsize-1, minimize=minimize)

        if eval_counter['count'] >= num_evaluations:
            break
    
    final_fitness = [fitness_funct(ind)[0] for ind in parents]
    best = parents[np.argmin(final_fitness)] # Mejor individuo final
    
    return best, fitness_funct(best)[0]

def EA3(parents, 
        fitness_funct, 
        mutation_funct, 
        num_generations, 
        num_evaluations, 
        size_offspring, 
        mutation_rate, 
        num_arrows, 
        minimize=False):
    fitness_cache = {}
    eval_counter = {'count': 0}
    
    def counted_fitness(ind):
        key = tuple(ind.flatten())  # O cualquier forma de hash válida
        if key not in fitness_cache:
            fitness_cache[key] = fitness_funct(ind)
            eval_counter['count'] += 1
        return fitness_cache[key][0]
        
    for _ in range(num_generations):
        # Selección
        selection = SUS(parents, counted_fitness, num_arrows, minimize=minimize)
        # Cruza y mutación
        offspring = []
        for i in range(0, size_offspring, 2):
            p1 = selection[i % len(selection)][0]
            p2 = selection[(i + 1) % len(selection)][0]
            crossover = uniform_crossover(p1, p2)
            for ind in crossover:
                offspring.append(np.array([mutation_funct(ind, mutation_rate)])) # mutación de los individuos
        offspring = offspring[:size_offspring]  # Limitar a tamaño de población
        # Reemplazo
        parents = tournament_similar_replacement(parents, offspring, counted_fitness, minimize=minimize)

        if eval_counter['count'] >= num_evaluations:
            break
    
    final_fitness = [fitness_funct(ind)[0] for ind in parents]
    best = parents[np.argmin(final_fitness)] # Mejor individuo final
    
    return best, fitness_funct(best)[0]

def evaluations(EA: str, 
                dimension: int, 
                popsize: int, 
                size_offspring: int,
                num_generations: int, 
                num_evaluations: int,
                n_arrows: int,
                mutation_rate=0.2):
    all_results = []
    all_times = []
    for exc in range(20):
        print('Ejecución: ', exc)
        results = []
        times = []
        for num, f in enumerate(functs):
            initial_pop = [np.random.uniform(-100, 100, size=(1, dimension)) for _ in range(popsize)]
            initial_fits = [f(ind)[0] for ind in initial_pop]
            i_time = time.time()
            print('Función: ', num)
            if EA == 'EA1':
                f_state = EA1(initial_pop, 
                              f, 
                              mutation, 
                              num_generations, 
                              num_evaluations,
                              size_offspring,
                              mutation_rate, 
                              minimize=True)[0]
            elif EA == 'EA2':
                f_state = EA2(initial_pop, 
                              f, 
                              mutation, 
                              num_generations,
                              num_evaluations, 
                              size_offspring,
                              mutation_rate, 
                              n_arrows, 
                              minimize=True)[0]
            elif EA == 'EA3':
                f_state = EA3(initial_pop, 
                              f, 
                              mutation, 
                              num_generations, 
                              num_evaluations, 
                              size_offspring,  
                              mutation_rate, 
                              n_arrows, 
                              minimize=True)[0]
            else:
                raise ValueError("Not valid Genetic Algorithm")
                
            f_time = time.time()
            ex_time = round(f_time - i_time, 4)
            print('Evaluación inicial (mejor de la población): ', min(initial_fits))
            print('Evaluación final (mejor de la población): ', f(f_state)[0])
            print('Tiempo de evaluación: ', ex_time)
            results.append((f(f_state)[0]))
            times.append(ex_time)
        print('------------------------------------------------------')
        all_results.append(results)
        all_times.append(times)
    return all_results, all_times