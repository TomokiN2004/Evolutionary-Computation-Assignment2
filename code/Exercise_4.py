"""
Exercise 4 – MMAS and MMAS_star Algorithm Implementation
--------------------------------------------------------

This script implements the Max-Min Ant System (MMAS) and MMAS* algorithms,
variations of Ant Colony Optimization (ACO), for Pseudo-Boolean Optimisation 
(PBO). The implementation strictly follows the construction and update procedures
described in the article "Analysis of different MMAS ACO algorithms..." (Figure 1, 
2, and 3). The algorithms are executed on PBO benchmark functions (F1, F2, F3, F18,
F23, F24, F25) with n = 100 and a max budget of 100,000 evaluations. Results are 
logged across varying pheromone evaporation rates, ρ ∈ {1, 1/√n, 1/n}, to enable 
performance comparison with the RLS and (1+1) EA algorithms implemented in Exercise 2.

Author:
    Tomoki Nonogaki (a1898137)

Date:
    September 2025
"""

# Import Libraries
from ioh import get_problem, ProblemClass, logger
import numpy as np


# Constructs A Solution 
def construct_solution(tau, alpha, beta):
    """
    Constructs a binary solution vector {0, 1}^n probabilistically based on 
    pheromone trail (tau) and heuristic information, following the ACO principle.

    Args:
        tau (np.ndarray): Pheromone trail vector, where tau[i] is the pheromone 
                          value associated with setting bit i to 1.
        alpha (float): The exponent (weight) applied to the pheromone value.
        beta (float): The exponent (weight) applied to the heuristic information (eta).

    Returns:
        np.ndarray: A constructed binary solution vector of length n.
    """
    # The Dimension of the Problem (Number of Bits/Variables)
    n = len(tau)
    solution = np.zeros(n, dtype = int)

    # Iterate Through Each Bit/Variable to Decide its Value
    for i in range(n):
        # Set eta to 1 (No External Heuristic is Used)
        eta = 1.0

        # Calculating t(v,w)
        prob_1 = tau[i] ** alpha * eta ** beta

        # Calculating Σ{(v,u),u∈Nv} 
        prob_0 = (1-tau[i]) ** alpha * eta ** beta

        # Calculating Success Probability Based on prob 1 & prob 2
        success_prob = prob_1 / (prob_1 + prob_0)
        
        # Probabilistic Decision: 
        if np.random.rand() < success_prob:
            # Set Bit to 1
            solution[i] = 1
        else:
            # Set Bit to 0
            solution[i] = 0
    
    # Return the Solution
    return solution


# Implement MMAS Algorithm
def MMAS_algorithm(PBO_instance, budget = 100_000, ants_num = 20, evaporation_rates = [1, 0.1, 0.01], alpha = 1.0, beta = 2.0, trials = 10):
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
    # Acquire Length of Bits
    n = PBO_instance.meta_data.n_variables

    # Special Cases 
    optimum = PBO_instance.optimum.y if PBO_instance.meta_data.problem_id != 18 or n != 32 else 8
    
    # Experiment Each Evaporation Rate
    for evaporation_rate in evaporation_rates:
        # Run the Algorithm n times Independently
        for run in range(trials):
            # Defining tau max & tau min
            tau_max = 1-(1/n)
            tau_min = 1/n
            
            # Initialise All Pheromone with 1/2
            tau = np.full(n, 1/2)
            eval_count = 0

            # Initialise Global Best Fitness Found in this Run
            best_fitness = -np.inf
            best_solution = None

            # While Loop in the Range of Budget (100000 default)
            while eval_count < budget:
                # Reset Best Fitness Found Within the Current Iteration (Colony Best)
                best_fitness_per_iter = -np.inf
                best_sol_per_iter = None

                # Let Ant Constructs Solution
                for _ in range (ants_num):
                    # Get a Solution
                    solution = construct_solution(tau, alpha, beta)

                    # Evaluate the Solution
                    fitness = PBO_instance(solution)

                    # Increase the Counter
                    eval_count += 1

                    # Update Best Fitness & Best Solution if New Solution is Better Than or Equal to Best So Far
                    if fitness >= best_fitness_per_iter:
                        best_fitness_per_iter = fitness
                        best_sol_per_iter = solution
                    
                    # Store Best Fitness & Solutions Per Run
                    if fitness >= best_fitness:
                        best_fitness = fitness
                        best_solution = solution

                # Pheromone Update Phase: Apply Evaporation & Deposition
                for i in range(n):
                    # Evaporation is Applied Implicitly: (1 - rho) * tau[i]
                    if best_sol_per_iter[i] == 1:
                        # Deposition for bit=1, then enforce tau_max limit
                        tau[i] = min((1 - evaporation_rate) * tau[i] + evaporation_rate, tau_max)
                    else:
                        # For bit=0, Only Evaporation is Applied, then Enforce tau_min Limit
                        tau[i] = max((1 - evaporation_rate) * tau[i], tau_min)
                
                # Check for Termination: Stop if the Known Optimum is Reached
                if best_fitness >= optimum:
                        break
            
            # Print Best Result from Trial No. X (Final Parent Fitness)
            print(f"Name: {PBO_instance.meta_data.name}, MMAS run:{run+1}, Evaporation rate:{evaporation_rate},  best:{best_fitness}")
            PBO_instance.reset()


# Implement MMAS* Algorithm
def MMAS_star_algorithm(PBO_instance, budget = 100_000, ants_num = 20, evaporation_rates = [1, 0.1, 0.01], alpha = 1.0, beta = 2.0, trials = 10):
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
    # Acquire Length of Bits
    n = PBO_instance.meta_data.n_variables

    # Special Cases 
    optimum = PBO_instance.optimum.y if PBO_instance.meta_data.problem_id != 18 or n != 32 else 8

    # Experiment Each Evaporation Rate
    for evaporation_rate in evaporation_rates:
        # Run the Algorithm n times Independently
        for run in range(trials):
            # Defining tau max and tau min
            tau_max = 1-(1/n)
            tau_min = 1/n
            
            # Initialise all Pheromone with 1/2
            tau = np.full(n, 1/2)
            eval_count = 0

            # Initialise Global Best Fitness Found in this Run
            global_best_fitness = -np.inf
            global_best_solution = None

            # While Loop in the Range of Budget (100000 default)
            while eval_count < budget:
                # Let Ant Constructs Solution
                for _ in range (ants_num):
                    # Get a Solution
                    solution = construct_solution(tau, alpha, beta)

                    # Evaluate the Solution
                    fitness = PBO_instance(solution)

                    # Increase the Counter
                    eval_count += 1

                    # Update Best Fitness & Best Solution if New Solution is Strictly Better than Best So Far 
                    if fitness > global_best_fitness:
                        global_best_fitness = fitness 
                        global_best_solution = solution
                
                # Pheromone Update 
                for i in range(n):
                    # Evaporation is Applied Implicitly: (1 - rho) * tau[i]
                    if global_best_solution[i] == 1:
                        # Deposition for bit=1, then enforce tau_max limit
                        tau[i] = min((1 - evaporation_rate) * tau[i] + evaporation_rate, tau_max)
                    else:
                        # For bit=0, Only Evaporation is Applied, then Enforce tau_min Limit
                        tau[i] = max((1 - evaporation_rate) * tau[i], tau_min)
                
                # Check for Termination: Stop if the Known Optimum is Reached
                if global_best_fitness >= optimum:
                    break
            
            # Print Best Result from Trial No. X (Final Parent Fitness)
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

    # Run MMAS on all Problems
    for problem in problems:
        problem.attach_logger(logger_MMAS)
        MMAS_algorithm(problem)

    # Run MMAS* on all Problems
    for problem in problems:
        problem.attach_logger(logger_MMAS_star)
        MMAS_star_algorithm(problem)

    # Ensure Logger Flushes Remaining Data
    del logger_MMAS
    del logger_MMAS_star

# Call & Run Tests
if __name__ == "__main__":
    main()