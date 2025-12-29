import matplotlib.pyplot as plt
import numpy as np

def plot_estimator_values(estimator_type, R_values, parameter_values):
    """
    Plots the values of the given estimator as a function of R for multiple parameter values.

    Parameters:
        estimator_type (str): Type of the estimator ('crude' or 'control variate').
        R_values (list): List of R values.
        parameter_values (list): List of parameter values to plot.
    """
    plt.figure(figsize=(10, 6))

    for param in parameter_values:
        if estimator_type == 'crude':
            values = [np.log(R + param) for R in R_values]  # Example function for crude estimator
        elif estimator_type == 'control variate':
            values = [np.sqrt(R + param) for R in R_values]  # Example function for control variate
        else:
            raise ValueError("Invalid estimator type. Choose 'crude' or 'control variate'.")

        plt.plot(R_values, values, label=f'Param = {param}')

    plt.title(f'{estimator_type.capitalize()} Estimator Values vs R')
    plt.xlabel('R')
    plt.ylabel('Estimator Value')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
R_values = np.linspace(1, 100, 100)  # R values from 1 to 100
parameter_values = [1, 5, 10, 20, 50]  # Example parameter values
plot_estimator_values('crude', R_values, parameter_values)