import numpy as np
from sympy import symbols, Matrix, diff, integrate


def perpendicular_distance_to_diagonal(matrix):
    """Calculate perpendicular distances from each row of the matrix to the diagonal vector (For Geometric Approach).

       Method:
       1. Creat the diagonal vector based on length of a single row. 
       2. View each row as a vector start at the origin, and calculate dot product of the row and the diagonal vector.
       3. Use equation: h = |v|sin(t) = sqrt(|d|^2|v|^2 - (d dot v)^2)/|d| to calculate perpendicular distances.
    """

    diagonal = np.ones(len(matrix.T)).reshape(1, -1)  # ex: [[1, 1, 1, 1]] for length of a single row = 4
    
    # Calculate distances element wise.
    # Shape: d(1, P), Matrix(N, P), Output(N, 1)
    dot_product = np.tensordot(diagonal, matrix, axes = ([1,1]))  # Dot products. 
    norm_product = np.linalg.norm(diagonal) * np.linalg.norm(matrix, axis = 1)  # |d||v|.
    distance = np.sqrt(norm_product**2 - dot_product**2)/np.linalg.norm(diagonal)  # Distances based on equation.

    return distance

def perm(u_vector):
    """Implementation of perm function described in Mangold (2015). (For Extremal Approach)"""

    item_list = []

    for i in range(len(u_vector)):
        for j in range(i + 1, len(u_vector)):
            item_list.append(u_vector[i]*u_vector[j])
            item_list.append(u_vector[i]*(1 - u_vector[j]))
            item_list.append((1 - u_vector[i])*u_vector[j])
            item_list.append((1 - u_vector[i])*(1 - u_vector[j]))
    
    return tuple(item_list)

def element_wise_product_sum(X, Y):
    """Implementation of <x, y> function described in Mangold (2015). (For Extremal Approach)"""

    product_sum = 0

    for i in range(len(X)):
        product_sum += X[i]*Y[i]
    
    return product_sum

def pi(X, minus_one = False):
    """Function for calculating u1*u2*u3... or (1 - u1)*(1 - u2)*(1 - u3).... (For Extremal Approach)"""

    product = 1

    if (minus_one):
        for i in range(len(X)):
            product *= (1 - X[i])
    else:
        for i in range(len(X)):
            product *= X[i]
        
    return product

def get_density_func(cdf, var):
    """Function for calculating the density function for a multivariate cumulative function. (For Extremal Approach)"""

    pdf = cdf

    for i in range(len(var)):
        pdf = diff(pdf, var[i])
    
    return pdf

def get_partial_derivative(fun, var):
    """Function for getting partial derivatives of a function. (For Extremal Approach)"""

    item_list = []

    for i in range(len(var)):
        item_list.append(diff(fun, var[i]))
    
    return tuple(item_list)

def intergrate_all_var(fun, var, start, end):
    """Implementation of integration in equation 11 described in Mangold (2015). (For Extremal Approach)"""

    intergration = fun

    for i in range(len(var)):
        intergration = integrate(intergration, (var[i], start, end))
        
    return intergration

def generate_formula(p):
    """Generate symbolic formulas described in Mangold (2015), and return formulas needed for implementation of extremal approach."""

    q = 2**p
    
    # Generate strings: "u1, u2, u3, ... up", "theta1, theta2, theta3, ... thetaq"
    u_list = [str(i) for i in range(1, p+1)]
    u_string = "u" + " u".join(u_list)
    theta_list = [str(i) for i in range(1, q+1)]
    theta_string = "theta" + " theta".join(theta_list)

    # Generate symbol tuples: (u1, u2, u3, ... up), (theta1, theta2, theta3, ... thetaq)
    u_vector = symbols(u_string)
    theta_vector = symbols(theta_string)
    
    # Generate functions
    perm_u = perm(u_vector)
    C_theta_u = pi(u_vector)*(1 + element_wise_product_sum(theta_vector, perm_u)*pi(u_vector, minus_one = True))  # Equation 2 described in Mangold (2015)
    c_theta_u = get_density_func(C_theta_u, u_vector)  # Density function of equation 2 described in Mangold (2015)
    c_dot_theta_u = get_partial_derivative(c_theta_u, theta_vector)  # Partial derivatives of the density function with respect to each theta
    
    return c_dot_theta_u, u_vector

def generate_cov_inv(formula, var):
    """Function for generating inverse of covariance matrix similar to the one in Example 3.2 described in Mangold (2015). (For Extremal Approach)"""

    cov = np.zeros((len(formula), len(formula)))
    
    for i in range(len(formula)):
        for j in range(len(formula)):
            cov[i][j] = intergrate_all_var(formula[i]*formula[j], var, 0, 1)
            
    return np.linalg.inv(cov)