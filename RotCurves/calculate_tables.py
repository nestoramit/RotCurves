import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.special import k0
from scipy.interpolate import CubicSpline
from scipy.special import gammaincinv
import random
import time
import os
import itertools
import parmap

from RotCurves.const import (
    ROOT_DIR,
)

TABLES_PATH = os.path.join(
    ROOT_DIR,
    "lookup_tables"
)

# FWHM to sigma gaussian relation
FWHM2sig = 2 * np.sqrt(2 * np.log(2))

def x_range(N):
    X = np.logspace(-3, -1, num=int(N * 0.05), endpoint=False)
    X = np.append(X, np.logspace(-1, np.log10(0.5), num=int(N * 0.15), endpoint=False))
    X = np.append(X, np.logspace(np.log10(0.5), 0, num=int(N * 0.3), endpoint=False))
    X = np.append(X, np.logspace(0, np.log10(5), num=int(N * 0.3), endpoint=False))
    X = np.append(X, np.logspace(np.log10(5), np.log10(50.), num=int(N * 0.20), endpoint=False))

    return X

def single_noordermeer_calculation(idx, q0, n, xrange):
    r"""x is defined as x \equiv r / r_eff"""
    x = xrange[idx]
    e = np.sqrt(1 - q0 * q0)
    b = gammaincinv(2 * n, 0.5)
    const = 2 * b ** (n + 1) / (np.pi * n ** 2)

    if n == 1:
        V_n = quad(
            lambda m: k0(b * m) * m * m / np.sqrt(x * x - (m * e) * (m * e)), 0, x
        )[0]
    else:
        inner_integral = lambda m: quad(
            lambda u: n * np.exp(-b*u) / np.sqrt(u**(2*n) - m**2),
            a=m**(1/n),
            b=np.inf,
        )[0]
        V_n = quad(
            lambda m: inner_integral(m) * m**2 / np.sqrt(x**2 - (m*e)**2),
            a=0,
            b=x,
        )[0]

    return V_n * const

def create_noordermeer_lookuptable(
        q_range,
        n_range,
        N=200,
        tol=1e-4,
        printtime=False,
        overwrite=False,
        running_in_cluster=False
):
    if not running_in_cluster:
        output_dir = os.path.join(TABLES_PATH,
                                  "Noordermeer_lookup_tables")
    else:
        output_dir = os.path.join(TABLES_PATH,
                                  "/mnt/sdceph/users/ycohen/Nestor/inputs/Noordermeer_lookup_tables")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    n_runs = len(n_range) * len(q_range)
    print(f"total Noordermeer lookup tables: {n_runs}")
    print(f"    n indexes: {len(n_range)}")
    print(f"    q0 values: {len(q_range)}")

    params_list = itertools.product(q_range, n_range)

    for params in params_list:
        starttime = time.time_ns()
        q, n = params
        q = np.round(q, 2)
        n = np.round(n, 2)

        # Create test array to compare the lookup table to
        # x_test = np.asarray(random.sample(range(1, 5000), 50)) / 1000
        x_test = np.logspace(-2, 1.5, num=100)
        indexes = range(len(x_test))
        test_results = parmap.map(single_noordermeer_calculation, indexes, q, n, x_test)
        test_results = np.asarray(test_results)

        if not running_in_cluster:
            output_name = os.path.join(output_dir, 'noor_n%2.2f_q%2.2f' % (n, q))
        else:
            output_name = '/'.join([output_dir, 'noor_n%s_q%s.csv' % (n, q)])

        do_calc = True
        if (os.path.exists(output_name + '.npy')) or (os.path.exists(output_name + '.csv')):
            print('n=%s, q=%s Already exists :)' % (n, q))
            do_calc = False
            if overwrite:
                print('Overwriting existing file n=%s, q=%s...' % (n, q))
                do_calc = True

        if do_calc:
            print("calculating for n=%s, q=%s..." % (n, q))

            for i in range(10):
                X = x_range(N)
                indexes = range(len(X))
                results = parmap.map(single_noordermeer_calculation, indexes, q, n, X)
                results = np.asarray(results)

                # compare agains test values if < tol for all
                inerp_vel = CubicSpline(x=X, y=results)
                ok = all((np.abs(inerp_vel(x_test) - test_results)) < tol)

                if ok:
                    print('ok!')
                    break
                else:
                    print(f'N={N:3.0f} not ok...')
                    N *= 1.1
                    print(f"    Increasing to {N:.0f}")

            data = np.append(X.reshape((len(X), 1)), results.reshape(len(results), 1), axis=1)

            # cols = ['x r/reff', 'V2']
            # df = pd.DataFrame(data=data, columns=cols, dtype=float)
            # df.to_csv(output_name + '.csv', index=False, header=cols)

            np.save(output_name, data)

            if printtime:
                print('Done. n=%s, q=%s (%s sec)' % (n, q, np.round((time.time_ns() - starttime) * 1e-9, 0)))

def single_GaussianRing_integral_v2(idx, h, x_range):
    x = x_range[idx]

    A = 4 * np.log(2)
    C = 4 * A / np.pi

    density_function_dimless = lambda x: np.exp(- A * h ** 2. * np.power(x - 1., 2.))
    density_prime_function_dimless = lambda x: - 2 * A * h ** 2. * (x - 1) * density_function_dimless(x)
    Iprime_function = lambda a: quad(
        lambda x: a * density_prime_function_dimless(np.sqrt(x ** 2 + a ** 2)) / np.sqrt(x ** 2 + a ** 2), 0, np.inf)[0]
    V2_function = lambda x: - quad(lambda a: a * Iprime_function(a) / np.sqrt(x ** 2 - a ** 2), 0, x)[0]
    V2 = V2_function(x) * C

    return V2

def single_GaussianRing_integral_menc(idx, h, x_range):
    x = x_range[idx]
    A = 4 * np.log(2)

    density_function_dimless = lambda x: np.exp(- A * h ** 2. * np.power(x - 1., 2.))
    totmass = quad(lambda a: a * density_function_dimless(a), 0, np.inf)[0]
    menc_function = lambda x: quad(lambda a: a * density_function_dimless(a), 0, x)[0] / totmass
    menc = menc_function(x)

    return menc

def single_GaussianRing_integral_phi(idx, h, x_range):
    x = x_range[idx]
    A = 4 * np.log(2)

    density_function_dimless = lambda x: np.exp(- A * h ** 2. * np.power(x - 1., 2.))
    density_prime_function_dimless = lambda x: - 2 * A * h ** 2. * (x - 1) * density_function_dimless(x)
    Iprime_function = lambda a: quad(
        lambda x: a * density_prime_function_dimless(np.sqrt(x ** 2 + a ** 2)) / np.sqrt(x ** 2 + a ** 2), 0, np.inf)[0]
    phi_function = lambda x: quad(
        lambda a: np.arcsin(np.minimum(2 * a / ((a + x) + np.abs(a - x)), 1.)) * Iprime_function(a),
        0, np.inf)[0]
    phi = phi_function(x)

    return phi

def single_GaussianRing_integral(idx, h, x_range):
    V2 = single_GaussianRing_integral_v2(idx, h, x_range)
    menc = single_GaussianRing_integral_menc(idx, h, x_range)
    return V2, menc

def create_GaussianRing_lookuptable(
        h_range,
        N=200,
        tol=0.01,
        running_in_cluster=False,
        printtime=True,
        overwrite=False
):
    if not running_in_cluster:
        output_dir = os.path.join(TABLES_PATH,
                                  "GaussianRing_lookup_tables")
    else:
        output_dir = "/mnt/sdceph/users/ycohen/Nestor/inputs/GaussianRing_lookup_tables"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    n_runs = len(h_range)
    print("creating Gaussian Ring lookup table... Total:", n_runs)

    x_test = np.asarray(random.sample(sorted(np.logspace(-1, 1, num=100)), 30))
    test_indexes = range(len(x_test))

    j = 1
    for h in h_range:
        starttime = time.time_ns()
        h = np.round(h, 2)

        if not running_in_cluster:
            output_name = os.path.join(output_dir, 'gaussring_h%2.2f' % h)
        else:
            output_name = '/'.join([output_dir, 'gaussring_h%2.2f' % h])

        do_calc = True
        if os.path.exists(output_name + '.npy'):
            print('h=%s Already exists :)' % h)
            do_calc = False
            if overwrite:
                print('Overwriting existing file h=%s...' % h)
                do_calc = True

        if do_calc:
            print("calculating for h=%s..." % h)

            # Create test array to compare the lookup table to
            test_results = parmap.map(single_GaussianRing_integral, test_indexes, h, x_test)
            test_results = np.asarray(test_results)

            for i in range(10):
                # Create array with length N and calc values
                X = x_range(N)
                indexes = range(len(X))
                results = parmap.map(single_GaussianRing_integral, indexes, h, X)
                results = np.asarray(results)

                # compare agains test values if < tol for all
                interp_vel = CubicSpline(x=X, y=results[:, 0])
                ok = all((np.abs(interp_vel(x_test) - test_results[:, 0])) < tol)

                if ok:
                    print('ok!')
                    break
                else:
                    print('N=%3.0f not ok...' % N)
                    N *= 1.1

            data = np.append(X.reshape((len(X), 1)), results, axis=1)
            np.save(output_name, data)

            if printtime:
                print('Done. h=%s (%s sec)' % (h, np.round((time.time_ns() - starttime) * 1e-9, 0)))

        print(f"Finished {np.round(j / n_runs * 100, 1)}%")
        j += 1


if __name__ == '__main__':
    create_noordermeer_lookuptable(
        q_range=[0.2],
        n_range=[4.0],
        N=1000,
        overwrite=True,
        printtime=True,
    )

    # create_GaussianRing_lookuptable(h_range=np.arange(0.05, 0.5, 0.05))
    # create_GaussianRing_lookuptable(h_range=np.arange(0.5, 2.0, 0.05))
    # create_GaussianRing_lookuptable(h_range=np.arange(2.0, 5.0, 0.05))
    # create_GaussianRing_lookuptable(h_range=np.arange(5.0, 10.0, 0.10))
    # create_GaussianRing_lookuptable(h_range=np.arange(10.0, 20.0, 0.5))
