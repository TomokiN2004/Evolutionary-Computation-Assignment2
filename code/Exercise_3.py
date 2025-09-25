"""
Exercise 3 – Genetic Algorithm
-----------------------------------------------------------------------------

This script implements a Genetic Algorithm (GA) for Pseudo-Boolean Optimisation 
(PBO) problems using the IOHexperimenter framework. 

The GA uses:
  - A parent population of size >= 10
  - Tournament selection
  - Uniform crossover
  - Bit-flip mutation
  - Elitism (optional, for preserving the best solutions)

The algorithm is run on benchmark problems (F1, F2, F3, F18, F23, F24, F25) 
with dimension = 100 and a maximum budget of 100,000 evaluations. Results 
are logged for later analysis with IOHanalyzer.

Author:
    Nethmi Ranathunga (a1895261)

Date:
    September 2025
"""
#inside algorithm loop replace random generation with GA loop (population , selection, crossover, mutation, elitism)
from ioh import get_problem, ProblemClass, logger
import numpy as np
import random

class Individual:
    def __init__(self, bitstring, problem):
        self.bitstring = bitstring #actual solution
        self.problem = problem  #IOH experimenter problem 
        self.fitness = problem(bitstring) #score returned by the problem for bitstring
    
    #take in the bistring and problem
    #save them as attributes
    #evaluate the fitness immediately
    def evaluate(self): #recalculate the fitness from the 
        fitness = self.problem(self.bitstring) 
        self.fitness = fitness
        return self.fitness
    
    @classmethod
    def random(cls, problem): #make random bitstring of length
        variables = problem.meta_data.n_variables
        bitstring = np.random.randint(2, size=variables).tolist()
        new_individual = cls(bitstring, problem)
        return new_individual
        
    def print_summary(self): #show bitstring and 
        print(self.bitstring, self.fitness)

class Population:
    def __init__(self, problem , size):
        #store problem and size
        #create size number of random individuals
        #store them in self.individuals (list)
        self.problem = problem 
        self.size = size #how many individuals are inside it
        #have list of individuals
        #loop size times and create random individual 
        #add it to the list
        #save the list as self.individuals
        list_Individual=[]
        for i in range(self.size):
            list_Individual.append(Individual.random(problem))
        
        self.individuals = list_Individual
        print(self.individuals)

        
    def best(self):
        #go through self.individuals
        #return the one with highest fitness
        best_individual = max(self.individuals, key=lambda individual:individual.fitness)

        return best_individual

def tournament_selection(population, k, new_size): 
    #for each new parent pick k random individuals from the population 
    #choose the one with best fitness
    #repear until you have new_size parents
    #output list of individuals (mating pool)
    mating_pool=[]
    for i in range(new_size):
        #pick k random competitors from the population
        competitors = random.sample(population.individuals, k)
        winner = max(competitors, key=lambda individual:individual.fitness)
        mating_pool.append(winner)
    return mating_pool
    
    
# def uniform_crossover:
#     #

# # def bit_flip_mutation:



# def genetic_algorithm(func, budget=100000, trials = 10):
#     return best_fitness, best_solution

# logger_ga = logger.Analzer(root="data/exercise-3", folder_name = "Run-GA", algoithm_name= "GeneticAlgorithm", algorithm_info ="GA With Uniform Crossover and Mutation")

# def main():