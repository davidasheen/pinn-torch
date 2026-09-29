import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import quad

def calculate_stretched_coordinate(X_physical, eps_0=0.4, eps_inf=1.135, k=25.0):
    """
    Approximates the stretched dimensionless axial coordinate X* 
    using an exponential blending function for the stretching variable.
    
    Parameters:
    X_physical : True dimensionless axial distance (x * nu) / (U_mean * h^2)
    eps_inf    : Fully developed stretching limit (1.5 for channel max velocity ratio)
    k          : Empirical decay constant tuning the transition rate
    """
    
    # Define the empirical stretching function epsilon(X)
    def stretching_function(X):
        return eps_inf - (eps_inf - eps_0) * np.exp(-k * X)
    
    # Define the integrand: 1 / epsilon(X)
    def integrand(X):
        return 1.0 / stretching_function(X)
    
    # If X_physical is an array, integrate for each point
    if isinstance(X_physical, (list, np.ndarray)):
        X_star = np.zeros_like(X_physical, dtype=float)
        for i, X in enumerate(X_physical):
            result, _ = quad(integrand, 0, X)
            X_star[i] = result
        return X_star
    
    # If X_physical is a single scalar value
    else:
        result, _ = quad(integrand, 0, X_physical)
        return result


def _transcendental_eq(x):
    """The eigenvalue equation for constant flow rate in a channel."""
    return np.tan(x) - x

def get_channel_eigenvalues(num_terms):
    """
    Finds the first 'num_terms' positive roots of tan(x) = x.
    Roots are bracketed asymptotically close to (n + 0.5) * pi.
    """
    roots = np.zeros(num_terms)
    for n in range(1, num_terms + 1):
        # The n-th root lies strictly between n*pi and (n + 0.5)*pi
        bracket = [n * np.pi + 1e-5, (n + 0.5) * np.pi - 1e-5]
        res = root_scalar(_transcendental_eq, bracket=bracket, method='brentq')
        roots[n-1] = res.root
    return roots

def calculate_channel_pressure_drop(rho, u_mean, t_hat, num_terms=50):
    """
    Calculates the cumulative pressure drop (p0 - p) for a transient 
    accelerating channel flow (or the spatial entrance region).
    
    Parameters:
    rho       : Density of the fluid (kg/m^3)
    u_mean    : Constant mean velocity of the flow (m/s)
    t_hat     : Dimensionless time (or dimensionless stretched coordinate X*)
    num_terms : Number of terms for the infinite series approximation
    """
    # Calculate the trigonometric eigenvalues
    gamma_n = get_channel_eigenvalues(num_terms)
    
    # Evaluate the integrated exponential decay summation
    integrated_sum = np.sum(
        (1 / gamma_n[:,None]**2) * (1 - np.exp(-(gamma_n[:,None]**2) * t_hat[None,:])),
        axis=0,
    )
    
    # Calculate the cumulative pressure drop (p0 - p)
    # The first term is the steady Poiseuille drop; the second is the transient penalty
    p0_minus_p = (rho * u_mean**2) * (3 * t_hat + 2 * integrated_sum)
    
    return p0_minus_p

def calculate_channel_pressure_gradient(h, u_mean, t_hat, mu, num_terms=50):
    """
    Calculates the pressure gradient (dp/dx) for a transient 
    accelerating channel flow (or the spatial entrance region).
    
    Parameters:
    h         : Channel half-height (m)
    u_mean    : Constant mean velocity of the flow (m/s)
    t_hat     : Dimensionless time (or dimensionless stretched coordinate X*)
    mu        : Dynamic viscosity of the fluid (Pa s)
    num_terms : Number of terms for the infinite series approximation
    """
    # Calculate the trigonometric eigenvalues
    gamma_n = get_channel_eigenvalues(num_terms)
    
    # Evaluate the exponential decay summation
    exp_sum = np.sum(np.exp(-(gamma_n[:,None]**2) * t_hat[None,:]),axis=0)
    
    # Calculate the pressure gradient 
    dp_dx = (-mu * u_mean / (h**2)) * (3 + 2 * exp_sum)
    
    return dp_dx

def calculate_channel_velocity(y, h, u_mean, t_hat, num_terms=50):
    """
    Calculates the transient velocity profile for a parallel-plate channel
    subjected to a sudden constant flow rate.
    
    Parameters:
    y         : Transverse position coordinate (m), from -h to h
    h         : Channel half-height (m)
    u_mean    : Constant mean velocity of the flow (m/s)
    t_hat     : Dimensionless time (nu * t / h^2)
    num_terms : Number of terms for the Fourier series approximation
    """
    # Obtain the eigenvalues (gamma_n)
    gamma_n = get_channel_eigenvalues(num_terms)
    
    # Dimensionless transverse coordinate
    y_star = y / h
    
    # Calculate the steady Poiseuille state (parabolic profile)
    u_steady = 1.5 * u_mean * (1 - y_star**2)
    
    # Calculate the transient infinite series term
    # The initial condition dictates the specific Fourier coefficients. 
    # For a plug-flow start, the coefficients C_n are derived via orthogonality.
    summation = 0
    for gamma in gamma_n:
        # C_n for an impulsive constant flow rate start
        c_n = 2 / (gamma**2)
        
        # Spatial mode shape: (cos(gamma * y/h) / cos(gamma)) - 1
        mode_shape = (np.cos(gamma * y_star) / np.cos(gamma)) - 1
        
        # Exponential temporal decay
        time_decay = np.exp(-(gamma**2) * t_hat)
        
        summation += c_n * mode_shape * time_decay
        
    # Combine steady state with the decaying transient response
    u_y_t = u_steady + u_mean * summation
    
    return u_y_t