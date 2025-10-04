"""
Exercise 5 – ACO algorithm
-----------------------------------------------------------------------------

This script implements an Ant Colony Optimisation (ACO) algorithm for 
Pseudo-Boolean Optimisation (PBO) problems using the IOHexperimenter framework. 
The algorithm is run with 20 ants (minimum requirement is 10), and a simple
local search is applied to improve constructed solutions. Multiple trials are 
executed on selected PBO benchmark problems using the same iteration and budget 
constraints as in Exercise 2.

Author:
    Emily Carey

Date:
    September 2025
"""

# Import Libraries
import numpy as np
from ioh import get_problem, ProblemClass, logger


# Implement Ant Colony Optimisation Algorithm
def ACO(func, budget=100_000, n_ants=20, evaporation=0.9, trials=10):
    # Input Problem Instance  
    n = func.meta_data.n_variables

    # From Exercise 2 Check for Optimum - Special Case for F18 with n=32
    if func.meta_data.problem_id == 18 and func.meta_data.n_variables == 32:
        optimum = 8
    else:
        optimum = func.optimum.y

    # Display the Target Optimum for the Given Function
    print(f"Target Optimum For {func.meta_data.name}: {optimum}")

    # At the Start No Solutions Yet
    best_sol = None
    best_fitness = -np.inf
    
    # Run Algorithm Mutiple times same as Exercise 2
    for run in range(trials):
        # Initialise Pheromone Values 
        pheromone = np.full(n, 0.5)
        eval = 0
        
        # Lowest Possible Start
        run_best_fitness = -np.inf
        run_best_sol = None

        # While Termination Conditions Not Met
        # Same Stopping Rule as Exercise 2 so Stop when the Budget is Used Up
        while eval < budget:
            solutions, fitnesses = [], []

            # Each Ant Constructs a Solution 
            for _ in range(n_ants):
                # Construct Solution (T) where the Solution is Built Probabilistically Based on Pheromone Values
                solution = np.random.rand(n) < pheromone
                solution = solution.astype(int)

                # Evaluate the Constructed Solution (Same Style as Exercise 2)
                fitness = func(solution)
                eval += 1

                # Local Search (1-bit Flip Improvement)
                i = np.random.randint(n)
                neighbour = solution.copy()
                neighbour[i] = 1 - neighbour[i]

                # Evaluate Fitness & Update Counter
                neigh_fit = func(neighbour)
                eval += 1

                # Accept Neighbour if it is at Least as Good as Current Solution
                if neigh_fit >= fitness:
                    solution, fitness = neighbour, neigh_fit

                # Append Results
                solutions.append(solution)
                fitnesses.append(fitness)

                # Track Best Solution So Far like Exercise 2
                if fitness > run_best_fitness:
                    run_best_fitness = fitness
                    run_best_sol = solution

                # Stop if Optimum is Found
                if fitness >= optimum:
                    break

            # Apply Pheromone Update
            best_idx = np.argmax(fitnesses)
            best_solution = solutions[best_idx]

            # Calculate the New Pheromone
            pheromone = pheromone * evaporation + best_solution * (1 - evaporation)

            # Keep Probabilities Bounded 
            pheromone = np.clip(pheromone, 0.05, 0.95)

            # Stop Early if Optimum Found
            if run_best_fitness >= optimum or eval >= budget:
                break
        
        # Ouput the Best Solution Found
        print(f"Run {run+1} Best For {func.meta_data.name}: f = {run_best_fitness}")

        # Reset Problem for Next Run
        func.reset()

        # Update Global Best Solution if Current Run Found Better
        if run_best_fitness > best_fitness:
            best_fitness = run_best_fitness
            best_sol = run_best_sol

    # Return the Best Result (Final Parent Fitness)
    return best_fitness, best_sol


# Main Function to Run the Tests
def main():
    """
    Set up PBO problems, attach logger, and run Random Search.
    """
    # Create Logger for the ACO Algorithm
    logger_aco = logger.Analyzer(root="data/exercise-5",
                                 folder_name="Running-ACO",
                                 algorithm_name="ACO",
                                 algorithm_info="Ant Colony Optimisation")

    # Declare Problems to be Tested (Om, LeadingOnes, LABS, etc.)
    problems = [
        get_problem(fid=1, dimension=100, instance=1, problem_class=ProblemClass.PBO),   # F1: OneMax
        get_problem(fid=2, dimension=100, instance=1, problem_class=ProblemClass.PBO),   # F2: LeadingOnes
        get_problem(fid=3, dimension=100, instance=1, problem_class=ProblemClass.PBO),   # F3: A Linear Function with Harmonic Weights
        get_problem(fid=18, dimension=100, instance=1, problem_class=ProblemClass.PBO),  # F18: LABS (Low Autocorrelation Binary Sequence)
        get_problem(fid=23, dimension=100, instance=1, problem_class=ProblemClass.PBO),  # F23: N-Queens Problem
        get_problem(fid=24, dimension=100, instance=1, problem_class=ProblemClass.PBO),  # F24: Concatenated Trap
        get_problem(fid=25, dimension=100, instance=1, problem_class=ProblemClass.PBO)   # F25: NKL (NK Landscapes)
    ]

    # Run ACO on all problems
    for problem in problems:
        problem.attach_logger(logger_aco)
        ACO(problem, budget=100_000, n_ants=20, trials=10)

    # Ensure Logger Flushes Remaining Data
    del logger_aco


# Call & Run Tests
if __name__ == "__main__":
    main()