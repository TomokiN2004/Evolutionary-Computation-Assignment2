# **Evolutionary Computation Assignment 1: TSP Solver**

## **Project Overview**

This project implements a modular, object-oriented Python library to solve the Traveling Salesman Problem *(TSP)* using both local search and evolutionary algorithms, as described in the assignment specification.

Each major component *(problem representation, operators, population models, selection methods, etc.)* is implemented in a flexible way so that algorithms can be easily extended and compared. All tests are reproducible and all results are recorded for analysis.

---

## **Project Structure**

```
final/
├── code/                     # All source code
│   ├── TSP_class.py          # Core TSP representation and utilities
│   └── (other modules...)
├── data/                     # Testing data folder
│   └── Testing TSPlib Files
├── results/                  # Output from algorithm runs
│   ├── local_search.txt
│   ├── local_search_analysis.txt
│   ├── your_EA.txt
│   ├── inverover.txt
│   └── your_EA_inverover_comparison.txt
└── doc/
    ├── readme.md             # ← This File
    ├── team_contributions.txt
    └── algorithm_design.txt
```

---

## **How to Run the Code**

#### **1. Requirements**

- Python 3.10 or later
- Non-standard libraries (install via pip):

> ```pip install numpy scipy```

*(Standard library modules used include: sys, time, random, statistics, datetime, multiprocessing, pathlib. These do not require installation.)*

#### **2. Run TSP Tests *(Exercises 1–5 & 7 Style Utilities)***

- Use `Test_Functions.py` as the test harness.
- From the project root:

> ```python3.10 code/Test_Functions.py```

- In `main()` inside `Test_Functions.py`, comment/uncomment the function calls to run
  specific, isolated tests. An example is below for testing Exercise 3:

```
    if __name__ == "__main__":
        # test_tsp_class()
        # test_permutations()
        # test_local_search()
        test_populations()
```

- Results are written to the `results/` directory.

#### **3. Run Exercise 6 Experiments (EAs)**

- Use `EA_experiments.py` in the same way as `Test_Functions.py`:
  - Toggle individual experiment functions in its `main()` to run particular EA setups.
- From the project root:
    > ```python3.10 code/EA_experiments.py```
- Results are written to the `results/` directory.

#### **4. Data Files:**

- Place input `.tsp` files in the `data/` directory.
- Only TSPLIB instances with `EDGE_WEIGHT_TYPE: EUC_2D` are supported.

---

## **Authors**

- Tomoki Nonogaki
- Nethmi Ranathunga
- Mohit Mittal
- Kamila Azamova
- Emily Carey
- Maxwell Busato

---

Export as pdf.
