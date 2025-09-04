"""
Exercise 2 – Randomised Local Search (RLS) and (1+1) Evolutionary Algorithm
-----------------------------------------------------------------------------

This script implements RLS and the (1+1) Evolutionary Algorithm for Pseudo-Boolean 
Optimisation (PBO) problems using the IOHexperimenter framework. It runs multiple 
trials on selected PBO benchmark problems (F1, F2, F3, F18, F23, F24, F25) with 
n = 100 and a maximum of 100,000 evaluations per trial. Results are logged for 
later analysis with IOHanalyzer to generate fixed-budget performance plots and 
summary statistics.

Author:
    Max Busato (a1851532)

Date:
    September 2025
"""

# Import Libraries
import numpy as np
from ioh import get_problem, ProblemClass, logger
from Exercise_1 import random_search


# Implement (1+1) Evolutionary Algorithm
def one_plus_one_EA(func, budget=None, trials=10):
    """
    Run (1+1) Evolutionary Algorithm on a given PBO problem.

    Args:
        func (ioh.problem): The Pseudo-Boolean Optimisation (PBO) problem 
                            instance.
        budget (int, optional): Maximum number of fitness evaluations.
        trials (int): Number of independent runs.

    Returns:
        tuple: Returns the (best_fitness, best_solution) of the final trial.
    """
    # Default Budget = 50 * n^2 if Not Provided
    if budget is None:
        budget = int(func.meta_data.n_variables ** 2 * 50)

    # Special Case: Known Optimum for Function F18 with n=32
    if func.meta_data.problem_id == 18 and func.meta_data.n_variables == 32:
        optimum = 8
    else:
        optimum = func.optimum.y

    # Display the Target Optimum for the Given Function
    print(f"Target Optimum For {func.meta_data.name}: {optimum}")

    # Run the Algorithm n times Independently
    for run in range(trials):
        # Set Size of n
        n = func.meta_data.n_variables

        # Generate Random Binary Solution
        parent_solution = np.random.randint(2, size=n)
    
        # Evaluate the Parent Solution
        parent_fitness = func(parent_solution)
                
        # Count Evaluation of Parent
        evaluations = 1  

        # Flip Each Bit with Probability 1/n Forever
        while evaluations < budget:
            # Take the Best Solution so far
            candidate = parent_solution.copy()

            # Decide Whether to Flip Each Bit
            for i in range(n):
                if np.random.random() < 1.0/n:
                    candidate[i] = 1 - candidate[i]

            # Evaluate Fitness & Update Counter
            candidate_fitness = func(candidate)
            evaluations += 1

            # Candiate Replaces Parent if Better or Equal
            if candidate_fitness >= parent_fitness:
                parent_solution = candidate
                parent_fitness = candidate_fitness
            
            # Stop Early if Optimum Found
            if parent_fitness >= optimum:
                break

        # Print Best Result from Trial No. X (Final Parent Fitness)
        print(f"Run {run+1} Best For {func.meta_data.name}: f = {parent_fitness}")

        # Reset Problem for Next Run
        func.reset()

    # Return the Best Result (Final Parent Fitness)
    return parent_fitness, parent_solution


# Implement Randomised Local Search (RLS)
def RLS(func, budget=None, trials=10):
    """
    Run Randomised Local Search on a given PBO problem.

    Args:
        func (ioh.problem): The Pseudo-Boolean Optimisation (PBO) problem 
                            instance.
        budget (int, optional): Maximum number of fitness evaluations.
        trials (int): Number of independent runs.

    Returns:
        tuple: Returns the (best_fitness, best_solution) of the final trial.
    """
    # Default Budget = 50 * n^2 if Not Provided
    if budget is None:
        budget = int(func.meta_data.n_variables ** 2 * 50)

    # Special Case: Known Optimum for Function F18 with n=32
    if func.meta_data.problem_id == 18 and func.meta_data.n_variables == 32:
        optimum = 8
    else:
        optimum = func.optimum.y

    # Display the Target Optimum for the Given Function
    print(f"Target Optimum For {func.meta_data.name}: {optimum}")

    # Run the Algorithm n times Independently
    for run in range(trials):
        # Set Size of n
        n = func.meta_data.n_variables

        # Generate Random Binary Solution
        parent_solution = np.random.randint(2, size=n)
    
        # Evaluate the Parent Solution
        parent_fitness = func(parent_solution)
                
        # Count Evaluation of Parent
        evaluations = 1  

        # Randomly Flip 1 Bit of Parent Forever
        while evaluations < budget:
            # Take the Best Solution So Far
            candidate = parent_solution.copy()

            # Decide which Bit to Flip
            i = np.random.randint(n)
            candidate[i] = 1 - candidate[i]

            # Evaluate Fitness & Update Counter
            candidate_fitness = func(candidate)
            evaluations += 1

            # Candiate Replaces Parent if Better or Equal
            if candidate_fitness >= parent_fitness:
                parent_solution = candidate
                parent_fitness = candidate_fitness
            
            # Stop Early if Optimum Found
            if parent_fitness >= optimum:
                break

        # Print Best Result from Trial No. X (Final Parent Fitness)
        print(f"Run {run+1} Best For {func.meta_data.name}: f = {parent_fitness}")

        # Reset Problem for Next Run
        func.reset()

    # Return the Best Result (Final Parent Fitness)
    return parent_fitness, parent_solution


# Main Function to Run the Tests
def main():
    """
    Set up PBO problems, attach logger, and run Random Search.
    """
    # Create Loggers Once for Each Algorithm
    logger_rs = logger.Analyzer(root="data/exercise-2", 
                                folder_name="Run-RandomSearch",
                                algorithm_name="RandomSearch",
                                algorithm_info="Random Search")
    logger_rls = logger.Analyzer(root="data/exercise-2", 
                                folder_name="Run-RLS",
                                algorithm_name="RLS",
                                algorithm_info="Randomized Local Search")
    logger_1p1 = logger.Analyzer(root="data/exercise-2", 
                                folder_name="Run-OnePlusOne",
                                algorithm_name="OnePlusOneEA",
                                algorithm_info="(1+1) Evolutionary Algorithm")

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

    # Run Random Search on all Problems
    for problem in problems:
        problem.attach_logger(logger_rs)
        random_search(problem, budget=100_000, trials=10)

    # Run RLS on all Problems
    for problem in problems:
        problem.attach_logger(logger_rls)
        RLS(problem, budget=100_000, trials=10)

    # Run (1+1) EA on all Problems
    for problem in problems:
        problem.attach_logger(logger_1p1)
        one_plus_one_EA(problem, budget=100_000, trials=10)

    # Ensure Logger Flushes Remaining Data
    del logger_rs
    del logger_rls
    del logger_1p1


# Call & Run Tests
if __name__ == "__main__":
    main()