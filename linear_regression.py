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
    Description: (A short description of what each of the functions you're written does).
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
    np.savez('data_test.npz', x=x, y=y, t=t, sigma=sigma)

# Feel free to change aspects of this function to suit your needs.
# Such as the title, labels, colors, etc.
def plot_with_shadded_bar(x=None, y=None, t=None, pred=None, sigma=None):
    """
    Plots the GT data for visualization.
    Args:
        x: x values
        y: y values
        sigma: standard deviation
    """
    if not os.path.exists('results'):
        os.makedirs('results')

    # Example plotting with the GT data, you can use this as a reference. You will later 
    # use this function to plot the results of your model.
    # np.load('data.npz')
    # x = np.load('data.npz')['x']
    # y = np.load('data.npz')['y']
    # t = np.load('data.npz')['t']
    # sigma = np.load('data.npz')['sigma']

    # Plot the ground truth curve of the generated data.
    fig, ax = plt.subplots()

    # Plot curve with red shaded region spans on std.
    ax.plot(x, y, 'r', label='Ground Truth Example')
    ax.plot(x, pred, 'b', label='Predicted')
    ax.fill_between(x, y-sigma, y+sigma, color='r', alpha=0.2)

    # Plot the noisy data points.
    ax.scatter(x, t, label='Noisy Data')

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_title('Data Point Plot Example')

    plt.savefig('results/gt_data.png')
    plt.close(fig)


# TODO: Use the existing functions to create/load data as needed. You will now call your linear regression model
# to fit the data and plot the results.
def linear_regression():
    # Load the data
    np.load('data.npz')
    x = np.load('data.npz')['x']
    y = np.load('data.npz')['y']
    t = np.load('data.npz')['t']
    sigma = np.load('data.npz')['sigma']

    # ML_Model = ML(degree=3)
    # ML_Model.fit(x,t)
    # pred = ML_Model.predict(x)
    # plot_with_shadded_bar(x=x, y=y, t=t, pred=pred, sigma=sigma)

    MAP_Model = MAP(degree=3)
    MAP_Model.fit(x,t)
    pred = MAP_Model.predict(x)
    plot_with_shadded_bar(x=x, y=y, t=t, pred=pred, sigma=sigma)

    
def main():
    generateNoisyData(50)
    # plot_with_shadded_bar()
    linear_regression()
    print("COMPLETED")



if __name__ == '__main__':
    main()