"""
Exercise 3 – Genetic Algorithm
-----------------------------------------------------------------------------


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
        print(list_Individual)
        
        self.individuals = list_Individual
        
    def best(self):
        #go through self.individuals
        #return the one with highest fitness
    
def individuals:
    
def population:
    
def tournament_selection: 
    
def uniform_crossover:

def bit_flip_mutation:



def genetic_algorithm(func, budget=100000, trials = 10):
    return best_fitness, best_solution

logger_ga = logger.Analzer(root="data/exercise-3", folder_name = "Run-GA", algoithm_name= "GeneticAlgorithm", algorithm_info ="GA With Uniform Crossover and Mutation")

def main():