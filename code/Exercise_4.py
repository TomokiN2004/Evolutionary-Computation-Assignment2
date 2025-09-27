"""
Exercise 4 – MMAS and MMAS_star algorithms implementation
--------------------------------------



Author:
    Tomoki Nonogaki (a1898137)

Date:
    September 2025
"""

from ioh import get_problem, ProblemClass, logger
import numpy as np

# Constructs solution 
def construct_solution(tau, alpha, beta):
    n = len(tau)
    solution = np.zeros(n, dtype = int)
    for i in range(n):
        eta = 1.0
        # Calculating t(v,w)
        prob_1 = tau[i] ** alpha * eta ** beta
        # Calculating Σ{(v,u),u∈Nv} 
        prob_0 = (1-tau[i]) ** alpha * eta ** beta
        # Calculating success probability based on prob 1 and prob 2
        success_prob = prob_1 / (prob_1 + prob_0)
        
        if np.random.rand() < success_prob:
            solution[i] = 1
        else:
            solution[i] = 0
    return solution



def MMAS_algorithm(PBO_instance, budget = 100000, ants_num = 20, evaporation_rates = [1, 0.1, 0.01], alpha = 1.0, beta = 2.0, trials = 10):
    """
    Runs Max Min Ant System on a given PBO problem
    
    Args:
        PBO_instance (ioh.problem): The PBO problem instance.
        budget (int): Max number of fitness evaluations.
        ants_num (int): Number of ants per iteration.
        evaporation_rate (list): List of evaporation rates to test [1, 1/sqrt(n), 1/n] where n = 100.
        alpha (float): Pheromone importance weight.
        beta (float): Heuristic importance weight.
        trials (int): Number of independnet runs.
    """

    # Acquire length of bits
    n = PBO_instance.meta_data.n_variables

    # Special cases 
    optimum = PBO_instance.optimum.y if PBO_instance.meta_data.problem_id != 18 or n != 32 else 8
    
    # Experiment each evaporation rate
    for evaporation_rate in evaporation_rates:
        # Run the Algorithm n times Independently
        for run in range(trials):
            # Defining tau max and tau min
            tau_max = 1-(1/n)
            tau_min = 1/n
            
            # Initialize all pheromone with 1/2
            tau = np.full(n, 1/2)

            eval_count = 0
            best_fitness = -np.inf
            best_solution = None

            # While loop in the range of budget (100000 default)
            while eval_count < budget:
                best_fitness_per_iter = -np.inf
                best_sol_per_iter = None

                # Let ant constructs solution
                for _ in range (ants_num):
                    # Get a solution
                    solution = construct_solution(tau, alpha, beta)
                    # Evaluate the solution
                    fitness = PBO_instance(solution)
                    # Increase the counter
                    eval_count += 1

                    # Update best fitness and best solution if new solution is better than or equal to best so far
                    if fitness >= best_fitness_per_iter:
                        best_fitness_per_iter = fitness
                        best_sol_per_iter = solution
                    
                    # Store best fitness and solutions per run
                    if fitness >= best_fitness:
                        best_fitness = fitness
                        best_solution = solution

                # Pheromone update 
                for i in range(n):
                    if best_sol_per_iter[i] == 1:
                        tau[i] = min((1 - evaporation_rate) * tau[i] + evaporation_rate, tau_max)
                    else:
                        tau[i] = max((1 - evaporation_rate) * tau[i], tau_min)
                
                if best_fitness >= optimum:
                        break
                        
            print(f"Name: {PBO_instance.meta_data.name}, MMAS run:{run+1}, Evaporation rate:{evaporation_rate},  best:{best_fitness}")
            PBO_instance.reset()



def MMAS_star_algorithm(PBO_instance, budget = 100000, ants_num = 20, evaporation_rates = [1, 0.1, 0.01], alpha = 1.0, beta = 2.0, trials = 10):
    """
    Runs Max Min Ant System star on a given PBO problem
    
    Args:
        PBO_instance (ioh.problem): The PBO problem instance.
        budget (int): Max number of fitness evaluations.
        ants_num (int): Number of ants per iteration.
        evaporation_rate (list): List of evaporation rates to test [1, 1/sqrt(n), 1/n] where n = 100.
        alpha (float): Pheromone importance weight.
        beta (float): Heuristic importance weight.
        trials (int): Number of independnet runs.
    """
     # Acquire length of bits
    n = PBO_instance.meta_data.n_variables

    # Special cases 
    optimum = PBO_instance.optimum.y if PBO_instance.meta_data.problem_id != 18 or n != 32 else 8

    # Experiment each evaporation rate
    for evaporation_rate in evaporation_rates:
        # Run the Algorithm n times Independently
        for run in range(trials):
            # Defining tau max and tau min
            tau_max = 1-(1/n)
            tau_min = 1/n
            
            # Initialize all pheromone with 1/2
            tau = np.full(n, 1/2)

            eval_count = 0
            global_best_fitness = -np.inf
            global_best_solution = None

            # While loop in the range of budget (100000 default)
            while eval_count < budget:
                # Let ant constructs solution
                for _ in range (ants_num):
                    # Get a solution
                    solution = construct_solution(tau, alpha, beta)
                    # Evaluate the solution
                    fitness = PBO_instance(solution)
                    # Increase the counter
                    eval_count += 1

                    # Update best fitness and best solution if new solution is strictly better than best so far 
                    if fitness > global_best_fitness:
                        global_best_fitness = fitness 
                        global_best_solution = solution
                

                # Pheromone update 
                for i in range(n):
                    if global_best_solution[i] == 1:
                        tau[i] = min((1 - evaporation_rate) * tau[i] + evaporation_rate, tau_max)
                    else:
                        tau[i] = max((1 - evaporation_rate) * tau[i], tau_min)
                
                if global_best_fitness >= optimum:
                    break
                
            print(f"Name: {PBO_instance.meta_data.name}, MMAS star run:{run+1}, Evaporation rate:{evaporation_rate},  best:{global_best_fitness}")
            PBO_instance.reset()
                
# Main Function to Run the Tests
def main():
    """
    Set up PBO problems, attach logger, and run MMAS and MMAS_star
    """
    # Create Loggers Once for Each Algorithm
    logger_MMAS = logger.Analyzer(root="data/exercise-4", 
                                folder_name="Run-MMAS",
                                algorithm_name="MMAS",
                                algorithm_info="MMAS")
    logger_MMAS_star = logger.Analyzer(root="data/exercise-4", 
                                folder_name="Run-MMAS_star",
                                algorithm_name="MMAS_star",
                                algorithm_info="MMAS_star")
 

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
        problem.attach_logger(logger_MMAS)
        MMAS_algorithm(problem)

    for problem in problems:
        problem.attach_logger(logger_MMAS_star)
        MMAS_star_algorithm(problem)

    # Ensure Logger Flushes Remaining Data
    del logger_MMAS
    del logger_MMAS_star

# Call & Run Tests
if __name__ == "__main__":
    main()