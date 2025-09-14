"""
Exercise 5 – ACO algorithm
-----------------------------------------------------------------------------

This script implements the dsign of an ACO algoritm using at least 10 ants
in this case 20. Local search is utilised to improve the constructed solutions.
The same function and iteration udget setting is used from Exercise 2

Author:
    Emily Carey

Date:
    September 2025
"""


import numpy as np
from ioh import get_problem, ProblemClass, logger

def ACO(func, budget=100_000, n_ants=20, evaporation=0.9, trials=10):

    # Input problem instance  
    n = func.meta_data.n_variables

    # From exercise 2 check for optimum - Special case for F18 with n=32
    if func.meta_data.problem_id == 18 and func.meta_data.n_variables == 32:
        optimum = 8
    else:
        optimum = func.optimum.y

    # Display the Target Optimum for the Given Function
    print(f"Target Optimum For {func.meta_data.name}: {optimum}")

    # At the start no solutions yet
    best_sol = None
    best_fitness = -np.inf
    
    # Run alg mutiple times same as exercise 2
    for run in range(trials):
        # Initialise pheromone values 
        pheromone = np.full(n, 0.5)
        eval = 0
        run_best_fitness = -np.inf
        run_best_sol = None


        # While termination conditions not met
        # Same stopping rule as Exercise 2 so stop when the budget is used up
        while eval < budget:
            solutions, fitnesses = [], []

            # Each ant constructs a solution for j = 1, . . . , na do (pseduo code)
            for _ in range(n_ants):

                # Construct solution (T) where the solution is built probabilistically based on pheromone values
                solution = np.random.rand(n) < pheromone
                solution = solution.astype(int)
                # Evaluate the constructed solution (same style as Exercise 2)
                fitness = func(solution)
                eval += 1

                # Local search (1-bit flip improvement)
                i = np.random.randint(n)
                neighbour = solution.copy()
                neighbour[i] = 1 - neighbour[i]
                neigh_fit = func(neighbour)
                eval += 1
                if neigh_fit >= fitness:
                    solution, fitness = neighbour, neigh_fit

                # Append
                solutions.append(solution)
                fitnesses.append(fitness)

                # Track best solution so far like Exercise 2
                if fitness > run_best_fitness:
                    run_best_fitness = fitness
                    run_best_sol = solution

                #Stop if optimum is found
                if fitness >= optimum:
                    break

            # Apply pheromone update
            best_idx = np.argmax(fitnesses)
            best_solution = solutions[best_idx]

            pheromone = pheromone * evaporation + best_solution * (1 - evaporation)

            # Keep probabilities bounded 
            pheromone = np.clip(pheromone, 0.05, 0.95)

            if run_best_fitness >= optimum or eval >= budget:
                break
        
        #Ouput the best sol foundd
        print(f"Run {run+1} Best For {func.meta_data.name}: f = {run_best_fitness}")

        # Reset problem for next run
        func.reset()

        if run_best_fitness > best_fitness:
            best_fitness = run_best_fitness
            best_sol = run_best_sol

    return best_fitness, best_sol


# Example: Run like in Exercise 2 with same benchmarks
def main():
    logger_aco = logger.Analyzer(root="data/exercise-5",
                                 folder_name="Running-ACO",
                                 algorithm_name="ACO",
                                 algorithm_info="Ant Colony Optimisation")

    problems = [
        get_problem(fid=1, dimension=100, instance=1, problem_class=ProblemClass.PBO),   # F1: OneMax
        get_problem(fid=2, dimension=100, instance=1, problem_class=ProblemClass.PBO),   # F2: LeadingOnes
        get_problem(fid=3, dimension=100, instance=1, problem_class=ProblemClass.PBO),   # F3: A Linear Function with Harmonic Weights
        get_problem(fid=18, dimension=100, instance=1, problem_class=ProblemClass.PBO),  # F18: LABS (Low Autocorrelation Binary Sequence)
        get_problem(fid=23, dimension=100, instance=1, problem_class=ProblemClass.PBO),  # F23: N-Queens Problem
        get_problem(fid=24, dimension=100, instance=1, problem_class=ProblemClass.PBO),  # F24: Concatenated Trap
        get_problem(fid=25, dimension=100, instance=1, problem_class=ProblemClass.PBO)   # F25: NKL (NK Landscapes)
    ]

    # Run on all problems
    for problem in problems:
        problem.attach_logger(logger_aco)
        ACO(problem, budget=100_000, n_ants=20, trials=10)

    # Ensure Logger Flushes Remaining Data
    del logger_aco

# Call & Run Tests
if __name__ == "__main__":
    main()
