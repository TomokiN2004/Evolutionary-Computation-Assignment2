"""
Exercise 1 – IOH Basic Functionality
--------------------------------------

This script implements a baseline Random Search algorithm using the 
IOHexperimenter framework. It runs multiple independent trials on selected PBO 
benchmark problems (OneMax, LeadingOnes, LABS) and logs the results for later 
analysis with IOHanalyzer.

Author:
    Kamila Azamova (a1864343)

Date:
    September 2025
"""

# Import Libraries
from ioh import get_problem, ProblemClass, logger
import numpy as np
import sys


# Implement Random Search Algorithm
def random_search(func, budget=None, trials = 10):
    """
    Run a Random Search on a given PBO problem.

    Args:
        func (ioh.problem): The Pseudo-Boolean Optimisation (PBO) problem 
                            instance from IOHexperimenter.
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
        best_fitness = -sys.float_info.max  # Lowest Possible Start
        best_solution = None

        # Random Search Loop
        for _ in range(budget):
            # Generate a Random Binary Solution
            candidate = np.random.randint(2, size=func.meta_data.n_variables)

            # Evaluate the Candidate Solution on the Given Problem 
            fitness = func(candidate)

            # Update Best-So-Far
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = candidate

            # Stop Early if Optimum Found
            if best_fitness >= optimum:
                break

        # Print Best Result from Trial No. X 
        print(f"Run {run+1} Best For {func.meta_data.name}: f = {best_fitness}")

        # Reset Problem for Next Run
        func.reset()

    # Return the Best Result
    return best_fitness, best_solution


# Main Function to Run the Tests
def main():
    """
    Set up PBO problems, attach logger, and run Random Search.
    """
    # Create Logger for the Random Search Algorithm
    log = logger.Analyzer(root="data/exercise-1",
                             folder_name="Run-RandomSearch",
                             algorithm_name="RandomSearch",
                             algorithm_info="Random Search")

    # Declare Problems to be Tested (Om, LeadingOnes, LABS, etc.)
    problems = [
        get_problem(fid=1, dimension=50, instance=1, problem_class=ProblemClass.PBO),   # F1: OneMax
        get_problem(fid=2, dimension=50, instance=1, problem_class=ProblemClass.PBO),   # F2: LeadingOnes
        get_problem(fid=18, dimension=50, instance=1, problem_class=ProblemClass.PBO),  # F18: LABS (Low Autocorrelation Binary Sequence)
    ]

    # Run Random Search on all Problems
    for problem in problems:
        problem.attach_logger(log)
        random_search(problem, trials=10)

    # Ensure Logger Flushes Remaining Data
    del log


# Call & Run Tests
if __name__ == "__main__":
    main()