'''
Start code for Project 1-Part 1: Linear Regression. 
CSE583/EE552 PRML
LA: Mukhil Muruganantham, 2025
LA: Vedant Sawant, 2025

Your Details: (The below details should be included in every python 
file that you add code to.)
{
    Name: Hayden Moore
    PSU Email ID: hmm5731
    Description:
    Linear Regression Project Implementation.
    This file implements polynomial regression using both Maximum Likelihood (ML) and 
    Maximum A Posteriori (MAP) approaches. It generates synthetic noisy sinusoidal data, 
    fits polynomial models of various degrees, and visualizes the results with confidence intervals.
    - generateNoisyData()
    - plot_with_shadded_bar()
    - linear_regression()
}
'''


import math
import os

import matplotlib.pyplot as plt
import numpy as np

# import my models code
from models import ML, MAP

def generateNoisyData(num_points=50):
    """
    Generates noisy sample points and saves the data. The function will save the data as a npz file.
    Args:
        num_points: number of sample points to generate.
    """
    x = np.linspace(1, 4*math.pi, num_points)
    y = np.sin(x*0.5)

    # Define the noise model
    nmu = 0
    sigma = 0.3
    noise = nmu + sigma * np.random.randn(num_points)
    t = y + noise

    # Save the data
    np.savez('data.npz', x=x, y=y, t=t, sigma=sigma)

# Feel free to change aspects of this function to suit your needs.
# Such as the title, labels, colors, etc.
# Added custom file_name parameter to output each test and label it correctly {ML, MAP}
# And Added noisy data 't' prediciton values 'pred'
def plot_with_shadded_bar(x=None, y=None, t=None, pred=None, sigma=None, file_name="output.png"):
    """
    ML and MAP Plots
    Args:
        x: x values
        y: y values (ground truth)
        t: t values (target values/noisy data)
        pred: pred values (prediction values from ML or MAP)
        file_name: file_name value (name to label and save accordingly)
        sigma: standard deviation
    """
    if not os.path.exists('results'):
        os.makedirs('results')
        
    fig, ax = plt.subplots()

    # Plot the Ground Truth X, Y
    ax.plot(x, y, 'r', label='Ground Truth')
    # Plot the Model predictions X, Pred
    ax.plot(x, pred, 'b', label=file_name)
    # Plot the prediction area with standard deviation
    ax.fill_between(x, pred-sigma, pred+sigma, color='b', alpha=0.2)
    # Plot the noisy data points
    #ax.scatter(x, t, color='b', marker='o', facecolors='none', edgecolors='blue', label='Noisy Data')

    # Set labels and create figure
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    # label based off of 'file_name'
    ax.set_title(f"{file_name} Results")
    plt.legend()
    # save file based off of 'file_name'
    plt.savefig(f'results/{file_name}.png')
    plt.close(fig)

def linear_regression():
    """
    Linear Regression Main Function for Maximum Likelihood (ML) and Maximum A Posteriori (MAP)
    Steps:
        1. Load in hw1-data.npz
        2. Generate Class Instance (ML or MAP) w/ Polynomial Degree
        3. Fit the model to the noisy data
        4. Collect the predictions on x from the fitted model
        5. Plot the results
    """
    # Load in data
    #data_name = "hw1-data-50.npz"
    data_name = "hw1-data-1000.npz"
    np.load(data_name)
    x = np.load(data_name)['x']
    y = np.load(data_name)['y']
    t = np.load(data_name)['t']
    sigma = np.load(data_name)['sigma']

    # Generate Class Instance (ML or MAP) w/ Polynomial Degree
    ML_Model = ML(degree=9)
    # Fit the model to the noisy data
    ML_Model.fit(x,t)
    # Collect the predictions on x from the fitted model
    pred = ML_Model.predict(x)
    # Plot the results
    plot_with_shadded_bar(x=x, y=y, t=t, pred=pred, sigma=sigma, file_name="ml_model")

    # Generate Class Instance (ML or MAP) w/ Polynomial Degree
    MAP_Model = MAP(degree=9)
    # Fit the model to the noisy data
    MAP_Model.fit(x,t)
    # Collect the predictions on x from the fitted model
    pred = MAP_Model.predict(x)
    # Plot the results
    plot_with_shadded_bar(x=x, y=y, t=t, pred=pred, sigma=sigma, file_name="map_model")

    
def main():
    generateNoisyData(1000)
    linear_regression()


if __name__ == '__main__':
    main()