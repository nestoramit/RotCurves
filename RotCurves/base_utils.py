import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from concurrent.futures import ThreadPoolExecutor


def integrate_quad_list(func, a, b):
    """
    Perform numerical integration with given limits using `scipy.integrate.quad`.

    Parameters:
    func : callable
        The function to integrate.
    a : float or list of floats
        The lower limit(s) of integration.
    b : float or list of floats
        The upper limit(s) of integration.

    Returns:
    np.array : The integration results for each limit pair.
    """

    # Check that a and b are either both single values or both lists, not mixed
    if isinstance(a, (list, np.ndarray)) and isinstance(b, (list, np.ndarray)):
        if len(a) != len(b):
            raise ValueError("If both 'a' and 'b' are lists, they must have the same length.")

    # If a or b is a single value, convert to list for consistency
    if not isinstance(a, (list, np.ndarray)):
        a = [a] * len(b) if isinstance(b, (list, np.ndarray)) else [a]
    if not isinstance(b, (list, np.ndarray)):
        b = [b] * len(a) if isinstance(a, (list, np.ndarray)) else [b]

    # Define a helper function for parallel integration
    def integrate_limits(a_val, b_val):
        if a_val == b_val:
            return 0.0
        result, _ = quad(func, a_val, b_val)
        return result

    # Use ThreadPoolExecutor to parallelize if a and b are lists
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(integrate_limits, a, b))

    return np.array(results)

def solve_numerical_using_brentq(func, p0, p1_oom=4., N=100, verbose=False, last=False):
    """
    Use Brent's method to find the root (zero value) of the function `func` starting from an initial guess `p0`.

    This function attempts to find the root by expanding around the initial guess `p0` and searching for sign changes.
    If no sign change is found in the neighborhood of `p0`, the function will attempt to recursively search.

    Parameters:
    - func: callable
        Function to find the root for.
    - p0: float
        Initial guess for the root.
    - p1_oom: float, optional
        Exponent for the range to search for the root. Default is 4.0.
    - N: int, optional
        Number of points to search in each direction. Default is 1000.
    - verbose: bool, optional
        If True, prints the progress. Default is False.
    - last: bool, optional
        If True, only the final search will be performed. Default is False.

    Returns:
    - float: The root found by Brent's method.
    """

    # Initial preparations
    sign_at_p0 = np.sign(func(p0))
    higher_values = np.logspace(0., p1_oom, num=N) * p0
    lower_values = np.logspace(0., -p1_oom, num=N) * p0

    # Check for sign change in the higher range
    higher_values_sign = np.sign(func(higher_values))
    lower_values_sign = np.sign(func(lower_values))

    # Identify where the sign changes and select p1
    p1 = None
    go_higher = False

    # Check if the sign changes at higher values than p0
    if any(higher_values_sign == -sign_at_p0):
        p1 = next(x for x in higher_values if np.sign(func(x)) == -sign_at_p0)
        go_higher = True
    # Check if the sign changes at lower values than p0
    elif any(lower_values_sign == -sign_at_p0):
        p1 = next(x for x in lower_values if np.sign(func(x)) == -sign_at_p0)
    else:
        if not last:
            pvalue = solve_numerical_using_brentq(func=func, p0=-p0, p1_oom=p1_oom, last=True)
            return pvalue
        else:
            # If no sign change found, plot the function for inspection
            X1 = np.logspace(-p1_oom, p1_oom, num=N) * p0
            X2 = np.logspace(-p1_oom, p1_oom, num=N) * -p0
            Y1 = func(X1)
            Y2 = func(X2)

            plt.figure()
            plt.plot(X1 / p0, Y1, label="func(X1)")
            plt.plot(X2 / p0, Y2, label="func(X2)")
            plt.xlabel('values / p0')
            plt.ylabel('func')
            plt.legend()
            plt.show()
            raise Exception("Brentq solver couldn't find two points with different signs...")

    # Verbose output
    if verbose:
        print(f'Went {"higher" if go_higher else "lower"}')
        print(f'Bounds: {p0}, {p1} ({np.sign(func(p0))}, {np.sign(func(p1))})')

    # Use Brent's method to find the root within the bounds [p0, p1]
    return brentq(func, p0, p1)

def create_r_space(edge, resolution):
    plus = np.arange(start=resolution, stop=edge + resolution, step=resolution)
    minus = -plus[::-1]
    R = np.array([*minus, 0, *plus])

    return R

def create_r_space_oversampled(edge, resolution, oversample):
        """
        Create a radial space array that ensures perfect divisibility by oversample factor.
        
        This method modifies the standard create_r_space behavior to ensure that
        the resulting array length is always divisible by the oversample factor,
        eliminating the need for interpolation during rebinning.
        
        Parameters
        ----------
        edge : float
            Maximum radius for the array [kpc].
        resolution : float
            Spatial resolution (pixel size) [kpc].
        oversample : int
            Oversampling factor.
            
        Returns
        -------
        ndarray
            Radial array with length divisible by oversample factor.
            
        Notes
        -----
        For even oversample factors:
            - Expands the lower edge by oversample/2 points
            - Shifts the entire array by oversample*dx/2 to maintain symmetry
            
        For odd oversample factors:
            - Expands both edges by floor(oversample/2) points
            - Maintains symmetry around the original array
        """
        
        # Create the original array to understand the target structure
        original_array = create_r_space(edge=edge, resolution=resolution)
        original_length = len(original_array)
        
        if oversample == 1:
            # No oversampling, use standard function
            return original_array
                
        if oversample % 2 == 0:
            # Even oversample: expand lower edge and shift
            lower_points = oversample // 2
            lower_edge = - (edge + resolution * lower_points) 
            
            upper_points = oversample // 3
            upper_edge = edge + resolution * upper_points
            R = np.arange(start=lower_edge, stop=upper_edge+resolution, step=resolution)

            shift = resolution / 2
            R = R + shift
            return R
                        
        else:
            # Odd oversample: expand both edges symmetrically
            lower_points = upper_points = oversample // 2
            lower_edge = - (edge + resolution * lower_points) 
            upper_edge = edge + resolution * upper_points
            R = np.arange(start=lower_edge, stop=upper_edge+resolution, step=resolution)

            return R
        
        # Verify the result is divisible by oversample
        if len(R) % oversample != 0:
            logger.warning(f"Created array length {len(R)} is not divisible by oversample {oversample}")
        
        return R

def safe_sqrt(squared, sign_array=None):
    sgn = np.sign(sign_array) if sign_array is not None else 1.
    return np.sqrt(np.maximum(0, squared)) * sgn

def safe_round_to_zero(array, atol=1e-10):
    if np.isscalar(array):
        return 0. if np.isclose(array, 0., atol=atol) else array
    else:
        array[np.isclose(array, 0., atol=atol)] = 0.
        return array