'''
Start code for Project 1-Part 1 and optional 2. 
CSE583/EE552 PRML
LA: Mukhil Muruganantham, 2025
LA: Vedant Sawant, 2025
Your Details: (The below details should be included in every python 
file that you add code to.)
{
    Name: Hayden Moore
    PSU Email ID: hmm5731
    Description: (A short description of what each of the functions you're written does).
    - ML
        - __init__(self, degree=3)
        - _create_polynomial_features(self, x)
        - fit(self, x, y)
        - predict(self, x)
    - MAP
        - __init__(self, alpha=0.005, beta=11.1, degree=3)
        - _create_polynomial_features(self, x)
        - fit(self, x, y)
        - predict(self, x)
}
'''


import numpy as np

class ML:
    def __init__(self, degree=3):
        """
        Args:
            degree (int): Degree of polynomial features
        """
        self.degree = degree
        self.weights = None
    
    def _create_polynomial_features(self, x):
        """Create polynomial features up to specified degree"""
        X = np.zeros((len(x), self.degree + 1))
        for i in range(self.degree + 1):
            X[:, i] = x ** i
        return X
    
    def fit(self, x, y):
        """
        Args:
            x (np.array): Input features
            y (np.array): Target values
        """
        X = self._create_polynomial_features(x)
        
        # w = (X^T X)^-1 X^T y
        self.weights = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))
    
    def predict(self, x):
        """
        Args:
            x (np.array): Input features
        Returns:
            np.array: Predicted values
        """
        if self.weights is None:
            raise ValueError("Model must be fitted before making predictions")
            
        # Create polynomial features and compute predictions
        X = self._create_polynomial_features(x)
        return np.dot(X, self.weights)

class MAP:
    def __init__(self, alpha=0.005, beta=11.1, degree=3):
        """
        Args:
            alpha (float): Precision of prior (regularization strength)
            beta (float): Precision of likelihood
            degree (int): Degree of polynomial features
        """
        self.alpha = alpha
        self.beta = beta
        self.degree = degree
        self.weights = None
    
    def _create_polynomial_features(self, x):
        """Create polynomial features up to specified degree"""
        X = np.zeros((len(x), self.degree + 1))
        for i in range(self.degree + 1):
            X[:, i] = x ** i
        return X
    
    def fit(self, x, y):
        """
        Args:
            x (np.array): Input features
            y (np.array): Target values
        """
        X = self._create_polynomial_features(x)
        
        # w = (βX^T X + αI)^-1 βX^T y
        N = len(x)
        I = np.eye(self.degree + 1)
        A_inv = np.linalg.inv(np.dot(self.beta * X.T, X) + self.alpha * I)
        self.weights = np.dot(A_inv, np.dot(self.beta * X.T, y))
    
    def predict(self, x):
        """
        Args:
            x (np.array): Input features
        Returns:
            np.array: Predicted values
        """
       
        # Create polynomial features and compute predictions
        X = self._create_polynomial_features(x)
        return np.dot(X, self.weights)