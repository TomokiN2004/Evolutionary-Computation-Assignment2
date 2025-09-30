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
# Import Libraries
import random
import numpy as np
from ioh import get_problem, ProblemClass, logger


class Individual:
    """
    Represents a single candidate solution (bitstring) for a given problem.

    Attributes:
        bitstring (list[int]): The binary representation of the solution.
        problem (callable): The fitness evaluation function provided by the problem.
        fitness (float): The fitness score of the bitstring evaluated by the problem.
    """
    def __init__(self, bitstring, problem):
        """
        Initialise an Individual with a bitstring and problem.

        Args:
            bitstring (list[int]): A binary list representing the solution.
            problem (callable): The problem function that evaluates fitness.

        Returns:
            None
        """
        self.bitstring = bitstring # Actual Solution
        self.problem = problem  # IOH Experimenter Problem 
        self.fitness = problem(bitstring) # Score Returned by the Problem for Bitstring
    

    def evaluate(self):
        """
        Recalculate the fitness score for the current bitstring.

        Args:
            None

        Returns:
            float: The updated fitness score.
        """
        # Evaluate Fitness Based on Current Bitstring
        fitness = self.problem(self.bitstring) 

        # Update Stored Fitness
        self.fitness = fitness

        # Return the Recalculated Fitness
        return self.fitness
    
    @classmethod
    def random(cls, problem):
        """
        Create a new Individual with a randomly generated bitstring.

        Args:
            problem (callable): The problem function that evaluates fitness.
                                Must have meta_data.n_variables to define length.

        Returns:
            Individual: A new randomly initialised individual.
        """
        # Number of Variables in Problem
        variables = problem.meta_data.n_variables

        # Generate Random 0/1 Bitstring
        bitstring = np.random.randint(2, size=variables).tolist()

        # Create Individual with Generated Bitstring
        new_individual = cls(bitstring, problem)

        # Return New Instance
        return new_individual
        
    def print_summary(self):
        """
        Print the individual's bitstring and its fitness score.

        Args:
            None

        Returns:
            None
        """
        print(self.bitstring, self.fitness)


class Population:
    """
    Represents a collection of Individuals for evolutionary algorithms.

    Attributes:
        problem (callable): The problem function used to evaluate individuals.
        size (int): Number of individuals in the population.
        individuals (list[Individual]): List of individuals in the population.
    """
    def __init__(self, problem , size):
        """
        Initialise a Population with random individuals.

        Args:
            problem (callable): The problem function for evaluation.
            size (int): Number of individuals in the population.

        Returns:
            None
        """
        # Store the Problem
        self.problem = problem 

        # Store Population Size
        self.size = size

        # Temporary List to Store Created Individuals
        list_Individual=[]
        
        # Create 'size' Individuals
        for i in range(self.size):
            # Add Random Individual Each Loop
            list_Individual.append(Individual.random(problem))

         # Store Individuals in Population
        self.individuals = list_Individual
        
    def best(self):
        """
        Get the best individual (highest fitness) in the population.

        Args:
            None

        Returns:
            Individual: The best individual in the population.
        """
        # Select the Individual with Maximum Fitness
        best_individual = max(self.individuals, key=lambda individual:individual.fitness)

        # Return the Best One
        return best_individual


# Implement Tournament Selection Method
def tournament_selection(population, k, new_size): 
    """
    Perform tournament selection to create a mating pool.

    Args:
        population (Population): The population to select from.
        k (int): Number of individuals to sample per tournament.
        new_size (int): Number of parents to select (mating pool size).

    Returns:
        list[Individual]: The selected individuals forming the mating pool.
    """
    # Initialise Empty Mating Pool
    mating_pool=[]

    # Repeat Selection Until Mating Pool Reaches Desired Size
    for i in range(new_size):
        # Randomly Pick k Competitors
        competitors = random.sample(population.individuals, k)
         # Choose Best Competitor
        winner = max(competitors, key=lambda individual:individual.fitness)
        # Add Winner to Mating Pool
        mating_pool.append(winner)

    # Return Full Mating Pool
    return mating_pool
    

# Implement Uniform Crossover Method
def uniform_crossover(parent1, parent2):
    """
    Perform uniform crossover between two parents.
        
    Args:
        parent1 (Individual): The first parent individual.
        parent2 (Individual): The second parent individual.
            
    Returns:
        Individual: A new offspring created by combining bits from both parents.
    """
    # Empty List to Store the Child's Bitstring
    child_bits=[]

    # Loop Over all Bit Positions
    for i in range(len(parent1.bitstring)):
        # With 50% Probability, take the Bit from parent1
        if random.random() < 0.5:
            child_bits.append(parent1.bitstring[i])
        # Otherwise take from parent2
        else:
            child_bits.append(parent2.bitstring[i])

    # Create a New Individual with the Child Bitstring
    return Individual(child_bits, parent1.problem)
        

# Implement Bit Flip Mutation Method
def bit_flip_mutation(individual):
    """
    Perform bit flip mutation on an individual.
    Each bit flips with probability 1/n, where n is the bitstring length.
    
    Args:
        individual (Individual): The individual to mutate.
    
    Returns:
        Individual: A new mutated individual.
    """
    # Length of Bitstring (Number of Variables)
    n = len(individual.bitstring)

    # Copy the Bitstring so Original Remains Unchanged
    mutated_bits = individual.bitstring.copy()
    
    # Loop through Each Bit
    for i in range(n):
        # With Probability 1/n, Flip the Bit (0 -> 1, 1 -> 0)
        if random.random() < 1/n:
            mutated_bits[i] = 1 - mutated_bits[i]
    
    # Create a New Individual with the Mutated Bitstring
    new_Individual = Individual(mutated_bits, individual.problem)
    
    # Return the Mutated Individual
    return new_Individual
    

# Construct a Genetic Algorithm 
def genetic_algorithm(problem, budget=100_000, trials=10, pop_size=20):
    """
    Run a Genetic Algorithm (GA) on a given problem.

    Args:
        problem (callable): The problem instance with evaluation and metadata.
        budget (int, optional): Maximum number of fitness evaluations allowed.
                                Defaults to 100,000.
        trials (int, optional): Number of independent GA runs to perform.
                                Defaults to 10.
        pop_size (int, optional): Number of individuals in the population.
                                  Defaults to 20.

    Returns:
        tuple:
            - float: Fitness of the best individual found.
            - list[int]: Bitstring of the best individual.
    """
    # Handle Special Cases For Defining the Optimum
    if problem.meta_data.problem_id == 18 and problem.meta_data.n_variables == 32:
        optimum = 8
    elif problem.meta_data.problem_id == 25:
        optimum= None
    else:
        optimum = problem.optimum.y
    
    # Print the Target Optimum for Reference
    print(f"Target Optimum For {problem.meta_data.name}: {optimum}")
    
    # Run the GA for the Specified Number of Trials
    for run in range(trials):
        # Intialise the Population 
        population = Population(problem, pop_size)

        # Track the Best Solution so far
        best_so_far = population.best()

        # Count Initial Evaluations
        evaluations  = pop_size
        
        # Main Genetic Algorithm Loop
        while evaluations < budget:
            # Selection
            parents = tournament_selection(population, k=3, new_size=pop_size)
            
            # Crossover + Mutation 
            offspring = [] # Store New Offspring

            # Process Parents in Pairs
            for i in range(0, pop_size, 2):
                parent1 = parents[i]
                parent2 = parents[(i + 1) % pop_size]
                
                # Perform Crossover to Create Two Children
                child1 = uniform_crossover(parent1, parent2)
                child2 = uniform_crossover(parent2, parent1)
                
                # Apply Mutation to Both Children
                child1 = bit_flip_mutation(child1)
                child2 = bit_flip_mutation(child2)
                
                # Add Children to Offspring List
                offspring.append(child1)
                offspring.append(child2)
                
            # Elitism 
            elite = population.best()
            # Replace a Random Offspring with the Elite to Preserve Best Solution
            offspring[random.randrange(len(offspring))] = elite
            
            # Replace Population 
            population.individuals = offspring[:pop_size]
            
            # Update Evaluations & Best so far 
            evaluations += pop_size
            if population.best().fitness > best_so_far.fitness:
                best_so_far = population.best()
                
            # Stop if Optimum Found
            if optimum is not None and best_so_far.fitness >= optimum:
                break
        
        # Print Best Result from Trial No. X (Final Parent Fitness)
        print(f"Run {run +1} Best For {problem.meta_data.name}: f = {best_so_far.fitness}")
        
        # Reset Problem for Next Run
        problem.reset()
    
    # Return Best Fitness & Bitstring found across all trials
    return best_so_far.fitness, best_so_far.bitstring


# Main Function to Run the Tests
def main():
    """
    Set up PBO problems, attach logger, and run Random Search.
    """
    # Create Logger for the Random Search Algorithm
    log_ga = logger.Analyzer(root="data/exercise-3",
                             folder_name="Run-GA",
                             algorithm_name="GeneticAlgorithm",
                             algorithm_info="Genetic Algorithm with Crossover and Mutation")
    
    # Declare Problems to be Tested (Om, LeadingOnes, LABS, etc.)
    problems = [
        get_problem(fid=1, dimension=100, instance=1, problem_class=ProblemClass.PBO),   # F1: OneMax
        get_problem(fid=2, dimension=100, instance=1, problem_class=ProblemClass.PBO),   # F2: LeadingOnes
        get_problem(fid=3, dimension=100, instance=1, problem_class=ProblemClass.PBO),   # F3: Linear Function
        get_problem(fid=18, dimension=100, instance=1, problem_class=ProblemClass.PBO),  # F18: LABS
        get_problem(fid=23, dimension=100, instance=1, problem_class=ProblemClass.PBO),  # F23: N-Queens
        get_problem(fid=24, dimension=100, instance=1, problem_class=ProblemClass.PBO),  # F24: Concatenated Trap
        get_problem(fid=25, dimension=100, instance=1, problem_class=ProblemClass.PBO)   # F25: NK Landscapes
    ]

    # Run the GA on all Problems
    for problem in problems:
        problem.attach_logger(log_ga)
        genetic_algorithm(problem, budget=100_000, trials=10, pop_size=20)
    
    # Ensure Logger Flushes Remaining Data
    del log_ga


# Call & Run Tests
if __name__ == "__main__":
    main()