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
    
    
def uniform_crossover(parent1, parent2):
    """
        Perform uniform crossover between two parents.
        
        Args:
            parent1(Individual): First Parent
            parent2(Individual): Second Parent
            
        Returns:
            Individual: A new offspring individual
    """
    child_bits=[]
    for i in range(len(parent1.bitstring)):

        if random.random() < 0.5:
            child_bits.append(parent1.bitstring[i])
        else:
            child_bits.append(parent2.bitstring[i])
    return Individual(child_bits, parent1.problem)
        
        
def bit_flip_mutation(individual):
    """
    Perform bit flip mutation on an individual 
    Each bit flips with probability 1/n
    
    Args:
        Individual (Individual): The individual to mutate
    
    Returns:
        Individual: A new mutated individual
    """
    n = len(individual.bitstring)
    mutated_bits = individual.bitstring.copy()
    
    for i in range(n):
        if random.random() < 1/n:
            mutated_bits[i] = 1 - mutated_bits[i]
        
    new_Individual = Individual(mutated_bits, individual.problem)
    
    return new_Individual
    
def genetic_algorithm(problem, budget=100_000, trials=10, pop_size=20):
    #Handle cases such as F18 with n = 32
    if problem.meta_data.problem_id == 18 and problem.meta_data.n_variables == 32:
        optimum = 8
    elif problem.meta_data.problem_id == 25:
        optimum= None
    else:
        optimum = problem.optimum.y
    
    print(f"Target Optimum For {problem.meta_data.name}: {optimum}")
    
    for run in range(trials):
        #intialise the population 
        population = Population(problem, pop_size)
        best_so_far = population.best()
        evaluations  = pop_size
        
        #main GA loop
        while evaluations < budget:
            #selection
            parents = tournament_selection(population, k=3, new_size=pop_size)
            
            #crossover + mutation 
            offspring = []
            for i in range(0, pop_size, 2):
                parent1 = parents[i]
                parent2 = parents[(i + 1) % pop_size]
                
                child1 = uniform_crossover(parent1, parent2)
                child2 = uniform_crossover(parent2, parent1)
                
                child1 = bit_flip_mutation(child1)
                child2 = bit_flip_mutation(child2)
                
                offspring.append(child1)
                offspring.append(child2)
                
            #elitism 
            elite = population.best()
            offspring[random.randrange(len(offspring))] = elite
            
            #replace population 
            population.individuals = offspring[:pop_size]
            
            #update evaluations and best 
            evaluations += pop_size
            if population.best().fitness > best_so_far.fitness:
                best_so_far = population.best()
                
            #stop if optimum found
            if optimum is not None and best_so_far.fitness >= optimum:
                break
            
        print(f"Run {run +1} Best For {problem.meta_data.name}: f = {best_so_far.fitness}")
        problem.reset()
        
    return best_so_far.fitness, best_so_far.bitstring


def main():
    log_ga = logger.Analyzer(root="data/exercise-3",
                             folder_name="Run-GA",
                             algorithm_name="GeneticAlgorithm",
                             algorithm_info="Genetic Algorithm with Crossover and Mutation")
        
    problems = [
        get_problem(fid=1, dimension=100, instance=1, problem_class=ProblemClass.PBO),   # F1: OneMax
        get_problem(fid=2, dimension=100, instance=1, problem_class=ProblemClass.PBO),   # F2: LeadingOnes
        get_problem(fid=3, dimension=100, instance=1, problem_class=ProblemClass.PBO),   # F3: Linear Function
        get_problem(fid=18, dimension=100, instance=1, problem_class=ProblemClass.PBO),  # F18: LABS
        get_problem(fid=23, dimension=100, instance=1, problem_class=ProblemClass.PBO),  # F23: N-Queens
        get_problem(fid=24, dimension=100, instance=1, problem_class=ProblemClass.PBO),  # F24: Concatenated Trap
        get_problem(fid=25, dimension=100, instance=1, problem_class=ProblemClass.PBO)   # F25: NK Landscapes
    ]

    for problem in problems:
        problem.attach_logger(log_ga)
        genetic_algorithm(problem, budget=100_000, trials=10, pop_size=20)

    del log_ga


if __name__ == "__main__":
    main()