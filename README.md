# hmm5731@psu.edu
linear_regression.py
 Linear Regression Project Implementation.
 
    This file implements polynomial regression using both Maximum Likelihood (ML) and 
    Maximum A Posteriori (MAP) approaches. It generates synthetic noisy sinusoidal data, 
    fits polynomial models of various degrees, and visualizes the results with confidence intervals.
    - generateNoisyData()
    - plot_with_shadded_bar()
    - linear_regression()

models.py
 Class ML() Description: 
    Polynomial Regression Model implementing Linear Least Squares.

    A machine learning model that fits polynomial features up to a specified degree
    using the normal equation method. Transforms 1D input data into polynomial features
    and performs regression by computing optimal weights through matrix operations.
        - __init__(self, degree=3)
        - _create_polynomial_features(self, x)
        - fit(self, x, y)
        - predict(self, x)

    Class MAP() Description: 
    Maximum A Posteriori (MAP) Polynomial Regression Model.
    A Bayesian regression model that fits polynomial features using MAP estimation
    with Gaussian prior and likelihood. Incorporates regularization through prior
    precision (alpha) and likelihood precision (beta) hyperparameters.
    - MAP
        - __init__(self, alpha=0.005, beta=11.1, degree=3)
        - _create_polynomial_features(self, x)
        - fit(self, x, y)
        - predict(self, x)

You are given starter code for CSE 583/EE 552 PRML Project 1  which contains the following: 

- Part 1, in the linear_regression.py file you will have the functions:
  - `generateNoisyData`: a function to generate noisy sample points and save data.
  - `linearRegression`: the main function you will need to complete. It contains a sample script to load data, plot points and curves.
  - `plot_GT_data`: A visualization function to draw curve with shaded error bar.
  - In the `models.py` file, you will implement your ML and MAP classes from scrath. If you choose to implement a classifier, you will also do such in this file.

## Packages
This project will use three primary packages: numpy, sklearn, and matplotlib. You will likely need to install such packages via pip if you don't already have them installed.

We recommend creating a virtual environment using Anaconda for this project.

## Package Imports
You may not add any additional packages to the code. You must also write the pertinant parts of the code (such as the projection, ML, and MAP) from scratch and may only make use of numpy.

Read the project description on CANVAS for more details on the project.