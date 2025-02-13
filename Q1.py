import numpy as np
from scipy.integrate import nquad
from scipy.linalg import det, inv

def gaussian_integral_verification(A, w):
    """
    This function numerically verifies the given expression by calculating the integral on the LHS
    and comparing it with the closed-form expression on the RHS.

    Parameters:
    A (numpy.ndarray): A symmetric positive-definite matrix of shape (N, N).
    w (numpy.ndarray): A vector of shape (N,).

    Returns:
    lhs_value (float): The value of the integral on the LHS.
    rhs_value (float): The value of the closed-form expression on the RHS.
    """
    N = len(w)  # Dimension of the vectors and matrix

    # Define the integrand function for the LHS integral
    def integrand(*args):
        v = np.array(args)  # Convert the tuple of arguments into a numpy array
        # Compute the exponent: -0.5 * v^T A v + v^T w
        exponent = -0.5 * np.dot(v, np.dot(A, v)) + np.dot(v, w)
        return np.exp(exponent)

    # Calculate the LHS integral using nquad
    lhs_value, _ = nquad(integrand, [(-np.inf, np.inf)] * N)

    # Calculate the RHS closed-form expression
    A_inv = inv(A)  # Inverse of matrix A
    det_A_inv = det(A_inv)  # Determinant of A inverse
    exponent_rhs = 0.5 * np.dot(w, np.dot(A_inv, w))
    rhs_value = np.sqrt(det_A_inv * (2 * np.pi) ** N) * np.exp(exponent_rhs)
    
    return lhs_value, rhs_value

################################
# part B
#################################

# Define a symmetric positive-definite matrix A
A1 = np.array([[4, 2, 1],
              [2, 5, 3],
             [1, 3, 6]])

A2 = np.array([[4, 2, 1],
              [2, 1, 3],
             [1, 3, 6]])
    
# Define vector w
w1 = np.array([1, 2, 3])

# Verify the expression
lhs_value_A, rhs_value_A = gaussian_integral_verification(A1, w1)

# this won't run -- A' is not positive-definite
#lhs_value_Ap, rhs_value_Ap = gaussian_integral_verification(A2, w1)

# Print the results
print(f"LHS Integral Value of A: {lhs_value_A}")
print(f"RHS Closed-Form Value of A: {rhs_value_A}")
print(f"Differences for A: {abs(lhs_value_A - rhs_value_A)}")

## A' matrix doesn't work in this code because it is not positive definite and does not converge
#print(f"LHS Integral Value of A': {lhs_value_Ap}")
#print(f"RHS Closed-Form Value of A': {rhs_value_Ap}")
#print(f"Differences for A': {abs(lhs_value_Ap - rhs_value_Ap)}")

print(f"LHS Integral Value of A': not positive-definite")
print(f"RHS Closed-Form Value of A': not positive-definite")
print(f"Differences for A': not positive-definite")

# A' is not positive definite, the integral cant be computed
# verfied that A is approx 4.2758 by integral expression or
# the closed form expression
# Code runs:
# LHS Integral Value of A: 4.275823659021463
# RHS Closed-Form Value of A: 4.2758236590115155
# Differences for A: 9.947598300641403e-12
# LHS Integral Value of A': not positive-definite
# RHS Closed-Form Value of A': not positive-definite
# Differences for A': not positive-definite

#################################
# Part C
####################################

import numpy as np
from scipy.linalg import inv

def calculate_moments(A, w, print_moments=None):
    """
    Calculate the first, second, and higher-order moments of a multivariate normal distribution
    defined by matrix A and vector w. Optionally print specific moments.

    Parameters:
        A (np.ndarray): The precision matrix (inverse of covariance matrix).
        w (np.ndarray): The mean-related vector.
        print_moments (list of str): List of moments to print (e.g., ['v1', 'v1v2', 'v1^2v2']).

    Returns:
        dict: A dictionary containing the calculated moments.
    """
    # Step 1: Compute the inverse of matrix A
    A_inv = inv(A)  # A_inv is the covariance matrix

    # Step 2: Compute the mean vector (μ = A^{-1} w)
    mean = np.dot(A_inv, w)

    # Step 3: Initialize a dictionary to store the moments
    moments = {}

    # Step 4: Calculate the first moments (⟨v_i⟩ = μ_i)
    for i in range(len(mean)):
        moments[f'v{i+1}'] = mean[i]  # Store ⟨v_i⟩ in the dictionary

    # Step 5: Calculate the second moments (⟨v_i v_j⟩ = A_inv[i, j] + μ_i μ_j)
    for i in range(len(mean)):
        for j in range(i, len(mean)):  # Only compute upper triangle to avoid redundancy
            if i == j:
                moments[f'v{i+1}^2'] = A_inv[i, j] + mean[i] ** 2  # ⟨v_i^2⟩
            else:
                moments[f'v{i+1}v{j+1}'] = A_inv[i, j] + mean[i] * mean[j]  # ⟨v_i v_j⟩

    # Step 6: Calculate higher-order moments (⟨v_i^2 v_j⟩, ⟨v_i v_j^2⟩, etc.)
    # ⟨v_i^2 v_j⟩ = A_inv[i, i] * mean[j] + 2 * A_inv[i, j] * mean[i] + mean[i]^2 * mean[j]
    # ⟨v_i^2 v_j^2⟩ = A_inv[i, i] * A_inv[j, j] + 2 * A_inv[i, j]^2 + A_inv[i, i] * mean[j]^2 + A_inv[j, j] * mean[i]^2 + mean[i]^2 * mean[j]^2
    moments['v1^2v2'] = A_inv[0, 0] * mean[1] + 2 * A_inv[0, 1] * mean[0] + mean[0] ** 2 * mean[1]
    moments['v2v3^2'] = A_inv[1, 1] * mean[2] + 2 * A_inv[1, 2] * mean[1] + mean[1] ** 2 * mean[2]
    moments['v1^2v2^2'] = (A_inv[0, 0] * A_inv[1, 1] + 2 * A_inv[0, 1] ** 2 +
                           A_inv[0, 0] * mean[1] ** 2 + A_inv[1, 1] * mean[0] ** 2 +
                           mean[0] ** 2 * mean[1] ** 2)
    moments['v2^2v3^2'] = (A_inv[1, 1] * A_inv[2, 2] + 2 * A_inv[1, 2] ** 2 +
                           A_inv[1, 1] * mean[2] ** 2 + A_inv[2, 2] * mean[1] ** 2 +
                           mean[1] ** 2 * mean[2] ** 2)

    # Step 7: Print specific moments if requested
    if print_moments:
        print("Requested Moments:")
        for moment in print_moments:
            if moment in moments:
                print(f"{moment}: {moments[moment]}")
            else:
                print(f"{moment}: Not found in calculated moments.")

    return moments

# Step 8: Define the input matrix A and vector w
A = np.array([[4, 2, 1],
              [2, 5, 3],
              [1, 3, 6]])
w = np.array([1, 2, 3])

# Step 9: Specify which moments to print
requested_moments = ['v1', 'v2', 'v3', 'v1v2', 'v2v3', 'v1v3', 'v1^2v2', 'v2v3^2', 'v1^2v2^2', 'v2^2v3^2']

# Step 10: Calculate the moments and print the requested ones
moments = calculate_moments(A, w, print_moments=requested_moments)