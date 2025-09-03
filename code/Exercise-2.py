"""
Please Fill Me In
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
        func (ioh.problem): The Pseudo-Boolean Optimisation (PBO) problem instance from IOHexperimenter.
        budget (int, optional): Maximum number of fitness evaluations.
        trials (int): Number of independent runs.

    Returns:
        tuple: A pair (best_fitness (float), best_solution (np.ndarray)) where:
    """
    # Default Budget = 50 * n^2 if Not Provided
    if budget is None:
        budget = int(func.meta_data.n_variables ** 2 * 50)

    # Set Optimum for Function
    optimum = func.optimum.y

    # Display the Target Optimum for the Given Function
    print(f"Target Optimum For {func.meta_data.name}: {optimum}")

    # Run the Algorithm 10 times Independently
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


# Implement (1+1) Evolutionary Algorithm
def one_plus_one_EA(func, budget=None, trials=10):
    """
    Run (1+1) Evolutionary Algorithm on a given PBO problem.

    Args:
        func (ioh.problem): The Pseudo-Boolean Optimisation (PBO) problem instance.
        budget (int, optional): Maximum number of fitness evaluations.
        trials (int): Number of independent runs.

    Returns:
        tuple: Currently returns None. Will return (best_fitness, best_solution) when implemented.
    """
    return None


# Implement Randomised Local Search (RLS)
def RLS(func, budget=None, trials=10):
    """
    Run Randomised Local Search on a given PBO problem.

    Args:
        func (ioh.problem): The Pseudo-Boolean Optimisation (PBO) problem instance.
        budget (int, optional): Maximum number of fitness evaluations.
        trials (int): Number of independent runs.

    Returns:
        tuple: Currently returns None. Will return (best_fitness, best_solution) when implemented.
    """
    return None


# Main Function to Run the Tests
def main():
    """
    Set up PBO problems, attach logger, and run Random Search.
    """
    # Create Loggers Once for Each Algorithm
    logger_rs = logger.Analyzer(root="data/exercise-2", folder_name="Run-RandomSearch",
                                algorithm_name="RandomSearch",
                                algorithm_info="Random Search")
    logger_rls = logger.Analyzer(root="data/exercise-2", folder_name="Run-RLS",
                                algorithm_name="RLS",
                                algorithm_info="Randomized Local Search")
    logger_1p1 = logger.Analyzer(root="data/exercise-2", folder_name="Run-OnePlusOne",
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