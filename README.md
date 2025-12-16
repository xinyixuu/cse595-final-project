# CSE595 Final Project  
**Analyzing Random-Label In-Context Learning via MLP-Head Classification**

## Overview
This repository contains the code and experimental results for the CSE595 final project.  
The project investigates the behavior of **random-label in-context learning (ICL)** and compares **token-based ICL** with **representation-based classification using an MLP head**, with a particular focus on understanding whether performance gains under random-label settings reflect genuine label learning or prompt/format priors.

Experiments are conducted on the **Deontology** and **Justice** datasets using small-scale language models.
## Project Structure
├── final_project_code_guangchen_li/   
    ├── data_io.py                     # Data loading and preprocessing│   
    ├── model.py                       # Model and MLP head definitions 
    ├── trainer_justice.py             # Training and evaluation pipeline
    ├── evaluate.py                    # Evaluation utilities
    ├── draw.py                        # Visualization scripts
    └── test.py                        # Entry point for running experiments
## Running the Code
1. Install required dependencies (PyTorch, NumPy, Matplotlib, etc.).
2. Run test.py
