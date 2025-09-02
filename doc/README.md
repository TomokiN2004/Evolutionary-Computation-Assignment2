# **Evolutionary Computation Assignment 2: Pseudo-Boolean Optimisation**

## **Project Overview**

This project implements and analyses a range of iterative search and evolutionary algorithms for **Pseudo-Boolean Optimisation (PBO)** problems, using the [IOHexperimenter](https://iohprofiler.github.io/) benchmarking framework.  

The aim is to gain practical experience with algorithm design, comparison, and analysis by applying different approaches to the PBO benchmark suite. The assignment explores both classical and modern optimisation techniques, highlighting their strengths and weaknesses across a variety of problem types.  

Specifically, the following are covered:

- **Basic exploration** of IOHprofiler functionality with Random Search.  
- **Implementation and analysis** of Randomised Local Search (RLS) and the (1+1) Evolutionary Algorithm.  
- **Design and testing** of a Genetic Algorithm (GA) with crossover, mutation, and population-based search.  
- **Implementation of Ant Colony Optimisation (ACO) methods**, including Max-Min Ant System (MMAS) and MMAS*, based on published research.  
- **Development of a custom ACO algorithm** and performance comparison against existing ACO variants.  

All experiments are run consistently on selected benchmark problems from the IOHprofiler PBO suite. Results are analysed using [IOHanalyzer](https://iohanalyzer.liacs.nl/), with fixed-budget plots and statistical summaries to compare algorithmic performance.

---

## **Project Structure**

```plaintext
final/
├── code/ # All Python Source Code
│ ├── Exercise-1.py
│ ├── (Other Exercises .py ...)
│
├── data/ # Raw Results from Experiments
│ ├── exercise-1/ # Raw Results for Exercise 1
| ├── (Other Exercises/ ...)
│ ├── exercise-1-results.zip # Final Results Used for Producing Analysis Figures
| ├── (Other Exercises .zip ...)
│
└── doc/ # Documentation & Analysis
  ├── exercise-1/
  | ├── figures.pdf # Plots Generated via IOHanalyzer
  | └── analysis.txt # Written Observations & Discussion
  ├── (Other Exercises/ ...)
  ├── README.md # ← This File
  └── team_contributions.txt # Roles & Contributions of Group Members
```

---

## **How to Run the Code**

### **1. Requirements**

- Python 3.10 or later  
- Non-standard libraries (install via pip):  

```bash
pip install ioh numpy
```

### **2. Running the Exercises**

- Each exercise has a built-in driver script:

```bash
python3.10 code/Exercise-x.py
```

- Results are saved to the `data/exercise-x` directory.

### **3. Results Files**

- **Raw results:** stored in `data/exercise-x-results.zip` for each exercise.
- **Figures:** exported as .pdf files in `doc/exercise-x`.
- **Analysis:** included as part of the documentation for each exercise.

---

## **Authors**

- Tomoki Nonogaki
- Nethmi Ranathunga
- Mohit Mittal
- Kamila Azamova
- Emily Carey
- Maxwell Busato

---
