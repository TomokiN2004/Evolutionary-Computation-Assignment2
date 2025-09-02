"""
Please Fill Me In
"""

# Import Libraries
from ioh import get_problem, ProblemClass, logger
import numpy as np
import sys


# Implement Random Search Algorithm
def random_search(func, budget=None):
    """
    Run a Random Search on a given PBO problem.

    Args:
        func (ioh.problem): The Pseudo-Boolean Optimisation (PBO) problem instance from IOHexperimenter.
        budget (int, optional): Maximum number of fitness evaluations. Defaults to 50 * n_variables^2.

    Returns:
        tuple: A pair (best_fitness (float), best_solution (np.ndarray)) where:
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
    print(f"Target Optimum: {optimum}")

    # Run the Algorithm 10 times Independently
    for run in range(10):
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

        # Reset Problem for Next Run
        func.reset()

    # Return the Best Result
    return best_fitness, best_solution


# Main Function to Run the Tests
def main():
    """
    Set up PBO problems, attach logger, and run Random Search.
    """
    # Declare Problems to be Tested (Om, LeadingOnes, LABS)
    problems = [
        get_problem(fid=1, dimension=50, instance=1, problem_class=ProblemClass.PBO),   # OneMax
        get_problem(fid=2, dimension=50, instance=1, problem_class=ProblemClass.PBO),   # LeadingOnes
        get_problem(fid=18, dimension=50, instance=1, problem_class=ProblemClass.PBO)   # LABS
    ]

    # Create Logger Compatible with IOHanalyzer
    logger_instance = logger.Analyzer(
        root="data/exercise-1",
        folder_name="run",
        algorithm_name="random_search",
        algorithm_info="Baseline random search for IOHexperimenter"
    )

    # Run Random Search on Each Problem
    for problem in problems:
        problem.attach_logger(logger_instance)
        best_f, best_x = random_search(problem)
        print(f"Best Result for {problem.meta_data.name}: f = {best_f}")

    # Ensure Logger Flushes Remaining Data
    del logger_instance


# Call & Run Tests
if __name__ == "__main__":
    main()