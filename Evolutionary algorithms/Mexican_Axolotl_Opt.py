import numpy as np
import random
import time
import copy
from EA_Operators import tournament
import matplotlib.pyplot as plt

def transition(MorFcache, fitness_cache, dimension, Lb, Ub, lmbd, fitness_counter, eval_counter, minimization):
    r = [random.random() for _ in range(dimension)]
    bestMorF = min(MorFcache, key=MorFcache.get)
    sum_MorF = np.sum(list(MorFcache.values()))
    new_MorFcache = {}
    for ind in MorFcache:
        pm = MorFcache[ind] / max(0.00001, sum_MorF)
        tran_ind = np.array(copy.deepcopy(ind))
        if pm < random.random():
            for i in range(dimension):
                tran_ind[i] = np.clip(round(ind[i] + (bestMorF[i] - ind[i]) * lmbd, 4), Lb, Ub)
        else:
            for i in range(dimension):
                min_i = Lb - ind[i]
                max_i = Ub - ind[i]
                tran_ind[i] = np.clip(round(min_i + (max_i - min_i) * r[i], 4), Lb, Ub)
                
        old_fitness = MorFcache[ind]
        new_fitness = fitness_counter(tran_ind, fitness_cache, None, eval_counter)
        if (new_fitness <= old_fitness and minimization) or (new_fitness >= old_fitness and not minimization):
            new_MorFcache[tuple(tran_ind)] = new_fitness
        else:
            new_MorFcache[ind] = old_fitness
            
    return new_MorFcache

def accidents(MorFcache, fitness_cache, dimension, Lb, Ub, dp, rp, fitness_counter, eval_counter, minimization):
    r = [random.random() for _ in range(dimension)]
    bestMorF = min(MorFcache, key=MorFcache.get)
    new_MorFcache = {}
    for ind in MorFcache:
        tran_ind = np.array(copy.deepcopy(ind))
        if random.random() <= dp:
            for i in range(dimension):
                if random.random() <= rp:
                    min_i = Lb - ind[i]
                    max_i = Ub - ind[i]
                    tran_ind[i] = np.clip(round(min_i + (max_i - min_i) * r[i], 4), Lb, Ub)
        old_fitness = MorFcache[ind]
        new_fitness = fitness_counter(tran_ind, fitness_cache, None, eval_counter)
        if (new_fitness <= old_fitness and minimization) or (new_fitness >= old_fitness and not minimization):
            new_MorFcache[tuple(tran_ind)] = new_fitness
        else:
            new_MorFcache[ind] = old_fitness
    return new_MorFcache

def newlife(Mcache, 
            Fcache, 
            fitness_cache, 
            dimension, 
            tournament_size, 
            fitness_counter, 
            fitness_funct, 
            eval_counter, 
            minimization):

    male_inds  = list(Mcache.keys())      
    male_fits  = [Mcache[ind] for ind in male_inds]

    female_inds  = list(Fcache.keys())    
    female_fits  = [Fcache[ind] for ind in female_inds]

    M = len(male_inds)
    F = len(female_inds)

    for j in range(F):
        Minds = [np.array(ind) for ind in male_inds]
        fj = female_inds[j]   
        ofj = female_fits[j]    

        # tournament() returns the winner
        best_m_ind_array = tournament(
            Minds,
            fitness_funct=fitness_funct,
            tournament_size=tournament_size,
            n_tournaments=1,
            minimize=True
        )[0]
        best_Mind = tuple(best_m_ind_array.flatten())
        id_win = male_inds.index(best_Mind)
        omj    = male_fits[id_win]  
        
        egg1 = list(best_Mind)
        egg2 = list(fj)
        for i in range(dimension):
            if random.random() < 0.5:
                None
            else:
                egg1[i] = fj[i]
                egg2[i] = best_Mind[i]

        fit1 = fitness_counter(np.array(egg1), fitness_cache, None, eval_counter)
        fit2 = fitness_counter(np.array(egg2), fitness_cache, None, eval_counter)

        candidates = [
            (tuple(egg1), fit1),
            (tuple(egg2), fit2),
            (best_Mind,    omj),
            (fj,           ofj)
        ]
        candidates.sort(key=lambda x: x[1], reverse=not minimization)

        mejor_ind, mejor_fit   = candidates[0]
        segundo_ind, segundo_fit = candidates[1]

        female_inds[j] = mejor_ind
        female_fits[j] = mejor_fit

        male_inds[id_win] = segundo_ind
        male_fits[id_win] = segundo_fit

    new_Mcache = { male_inds[i]: male_fits[i] for i in range(M) }
    new_Fcache = { female_inds[i]: female_fits[i] for i in range(F) }

    return new_Mcache, new_Fcache

def MAO(initialpop: list,
        popsize: int,
        dimension: int,
        Lb: int, 
        Ub: int,
        num_evaluations: int,
        fitness_funct,
        lmbd: int,
        dp: int,
        rp: int,
        tournament_size: int,
        minimization=True):
    
    def counted_fitness(ind, fitness_cache, pop_cache, eval_counter):
        key = tuple(ind.flatten()) 
        if key not in fitness_cache:
            fit_val = fitness_funct(np.array([ind]))[0]
            fitness_cache[key] = fit_val
            eval_counter['count'] += 1

        if pop_cache is not None:
            pop_cache[key] = fitness_cache[key]
        return fitness_cache[key]

    def fitness_function(ind):
        return fitness_funct(np.array([ind]))

    fit_cache = {}
    eval_count = {'count': 0}
    Fcache = {}
    Mcache = {}

    # Divide the population in males and females
    for i, ind in enumerate(initialpop):
        ind = np.array(ind).flatten()
        if i%2 == 0:
            # Evaluate each male individual
            counted_fitness(ind, fit_cache, Mcache, eval_count)
        else:
            # Evaluate each male individual
            counted_fitness(ind, fit_cache, Fcache, eval_count)
    total_fitness = []
    while eval_count['count'] < num_evaluations:
        # Transition phase
        Fcache = transition(Fcache, fit_cache, dimension, Lb, Ub, lmbd, counted_fitness, eval_count, minimization)
        Mcache = transition(Mcache, fit_cache, dimension, Lb, Ub, lmbd, counted_fitness, eval_count, minimization)
        
        # Accidents phase
        Fcache = accidents(Fcache, fit_cache, dimension, Lb, Ub, dp, rp, counted_fitness, eval_count, minimization)
        Mcache = accidents(Mcache, fit_cache, dimension, Lb, Ub, dp, rp, counted_fitness, eval_count, minimization)

        # NewLife phase
        Mcache , Fcache = newlife(Mcache, 
                                Fcache, 
                                fit_cache, 
                                dimension, 
                                tournament_size, 
                                counted_fitness, 
                                fitness_function, 
                                eval_count, 
                                minimization)

    totalpop = list(Mcache.keys()) + list(Fcache.keys())
    totalpop_values = list(Mcache.values()) + list(Fcache.values())
    best_fit_indx = np.argmin(totalpop_values)
    best_fit = totalpop_values[best_fit_indx]
    best = totalpop[best_fit_indx]
    best = np.array(best)
    return best, best_fit