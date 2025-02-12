from utils import *

def solve_noordermeer_integral(q0, n, x, print_time=False):
    starttime = time.time()

    q = q0 * np.sqrt(np.power(np.sin(np.deg2rad(75)), 2) + np.power(np.cos(np.deg2rad(75)) / q0, 2))
    e = np.sqrt(1 - q0 * q0)

    b_sersic = {"1": 1.678, "1.0": 1.678, "2": 3.672, "2.0": 3.672, "3": 5.670, "3.0": 5.670, "4": 7.669, "4.0": 7.669}
    if n == 1.0 or n == 2.0 or n == 3.0 or n == 4.0:
        b = b_sersic[str(n)]
    else:
        b = scp_functions.gammaincinv(2 * n, 0.5)

    if n == 1:
        V_n = scp_integrate.quad(lambda m: scp_functions.k0(b * m) * m * m / np.sqrt(x * x - (m * e) * (m * e)), 0, x)[
            0]
    else:
        V_n = scp_integrate.quad(
            lambda m: scp_integrate.quad(
                lambda t: np.exp(- b * np.power(t, (1 / n))) * np.power(t, ((1 / n) - 1)) / np.sqrt(t * t - m * m), m,
                np.inf)[0]
                      * m * m / np.sqrt(x * x - (m * e) * (m * e)), 0, x)[0]

    if print_time:
        print()
        print("Noordermeer integral runtime: %s seconds" % round((time.time() - starttime), 2))

    return V_n


def single_noordermeer_calculation(idx, q, n, xrange):
    x = xrange[idx]
    return solve_noordermeer_integral(q, n, x)


def create_noordermeer_lookuptable(q_range, n_range, N=200, tol=0.001, printtime=False, overwrite=False,
                                   running_in_cluster=False):
    if not running_in_cluster:
        output_dir = os.path.join(lookuptables_dir, "Noordermeer_lookup_tables")
    else:
        output_dir = r"/mnt/sdceph/users/ycohen/Nestor/inputs/Noordermeer_lookup_tables"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    run_length = len(n_range) * len(q_range)
    print("total Noordermeer lookup tables:", run_length)

    params_list = itertools.product(q_range, n_range)

    for params in params_list:
        starttime = time.time_ns()
        q, n = params
        q = np.round(q, 2)
        n = np.round(n, 2)

        # Create test array to compare the lookup table to
        x_test = np.asarray(random.sample(range(1, 5000), 70)) / 1000
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
                inerp_vel = scp_interp.CubicSpline(x=X, y=results)
                ok = all((np.abs(inerp_vel(x_test) - test_results)) < tol)

                if ok:
                    print('ok!')
                    break
                else:
                    print('N=%3.0f not ok...' % N)
                    N *= 1.1

            data = np.append(X.reshape((len(X), 1)), results.reshape(len(results), 1), axis=1)
            cols = ['x r/reff', 'V2']
            df = pd.DataFrame(data=data, columns=cols, dtype=float)

            df.to_csv(output_name + '.csv', index=False, header=cols)
            np.save(output_name, data)

            if printtime:
                print('Done. n=%s, q=%s (%s sec)' % (n, q, np.round((time.time_ns() - starttime) * 1e-9, 0)))


def single_GaussianRing_integral(idx, invh, x_range):
    x = x_range[idx]

    A = 4 * np.log(2)
    density_function_dimless = lambda x: np.exp(- A * invh ** 2. * np.power(x - 1., 2.))
    density_prime_function_dimless = lambda x: - 2 * A * invh ** 2. * (x - 1) * density_function_dimless(x)
    Iprime_function = lambda a: scp_integrate.quad(
        lambda x: a * density_prime_function_dimless(np.sqrt(x ** 2 + a ** 2)) / np.sqrt(x ** 2 + a ** 2), 0, np.inf)[0]
    # potential_function = lambda x: integrate.quad(lambda a: np.arcsin(np.minimum(2 * a / ((a + x) + np.abs(a - x)), 1.)) * Iprime_function(a), 0, np.inf)[0]
    V2_function = lambda x: - scp_integrate.quad(lambda a: a * Iprime_function(a) / np.sqrt(x ** 2 - a ** 2), 0, x)[0]

    totmass = scp_integrate.quad(lambda a: a * density_function_dimless(a), 0, np.inf)[0]
    menc_function = lambda x: scp_integrate.quad(lambda a: a * density_function_dimless(a), 0, x)[0] / totmass

    return V2_function(x), menc_function(x)


def create_GaussianRing_lookuptable(invh_range, N=200, tol=0.01,
                                    running_in_cluster=False, printtime=False, overwrite=False):
    if not running_in_cluster:
        output_dir = os.path.join(lookuptables_dir, "GaussianRing_lookup_tables")
    else:
        output_dir = r"/mnt/sdceph/users/ycohen/Nestor/inputs/GaussianRing_lookup_tables"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    run_length = len(invh_range)
    print("creating Gaussian Ring lookup table... Total:", run_length)

    x_test = np.asarray(random.sample(sorted(np.logspace(-1, 1, num=100)), 30))
    test_indexes = range(len(x_test))

    j = 0
    for invh in invh_range:
        starttime = time.time_ns()
        invh = np.round(invh, 2)

        if not running_in_cluster:
            output_name = os.path.join(output_dir, 'Gauss_invh_%2.2f' % invh)
        else:
            output_name = '/'.join([output_dir, 'Gauss_invh_%2.2f' % invh])

        do_calc = True
        if (os.path.exists(output_name + '.npy')) or (os.path.exists(output_name + '.csv')):
            print('invh=%s Already exists :)' % invh)
            do_calc = False
            if overwrite:
                print('Overwriting existing file invh=%s...' % invh)
                do_calc = True

        if do_calc:
            print("calculating for invh=%s..." % invh)

            # Create test array to compare the lookup table to
            test_results = parmap.map(single_GaussianRing_integral, test_indexes, invh, x_test)
            test_results = np.asarray(test_results)

            for i in range(10):
                # Create array with length N and calc values
                X = x_range(N)
                indexes = range(len(X))
                results = parmap.map(single_GaussianRing_integral, indexes, invh, X)
                results = np.asarray(results)

                # compare agains test values if < tol for all
                interp_vel = scp_interp.CubicSpline(x=X, y=results[:, 0])
                ok = all((np.abs(interp_vel(x_test) - test_results[:, 0])) < tol)

                if ok:
                    print('ok!')
                    break
                else:
                    print('N=%3.0f not ok...' % N)
                    N *= 1.1

            data = np.append(X.reshape((len(X), 1)), results, axis=1)
            cols = ['x r/rpeak', 'V2', 'menc']
            df = pd.DataFrame(data=data, columns=cols, dtype=float)

            if printtime:
                print('Done. invh=%s (%s sec)' % (invh, np.round((time.time_ns() - starttime) * 1e-9, 0)))

            df.to_csv(output_name + '.csv', index=False, header=cols)
            np.save(output_name, data)

        print(f"Finished {np.round(j / run_length * 100, 1)}%")
        j += 1


def x_range(N):
    X = np.logspace(-3, -1, num=int(N * 0.05), endpoint=False)
    X = np.append(X, np.logspace(-1, np.log10(0.5), num=int(N * 0.15), endpoint=False))
    X = np.append(X, np.logspace(np.log10(0.5), 0, num=int(N * 0.3), endpoint=False))
    X = np.append(X, np.logspace(0, np.log10(3), num=int(N * 0.3), endpoint=False))
    X = np.append(X, np.logspace(np.log10(3), np.log10(50.), num=int(N * 0.20), endpoint=False))

    return X


if __name__ == '__main__':
    # invh_range = invh_list
    # Rpeak_range = np.arange(1., 10., step=0.1)
    # Rpeak_range = np.append(Rpeak_range, np.arange(10., 20.5, 0.5))
    # create_GaussianRing_BT_lookuptable(invh_range, Rpeak_range, printtime=True, overwrite=True)

    invh_range = np.arange(0.05, 0.5, 0.05)
    invh_range = np.append(invh_range, np.arange(0.5, 1.5, 0.01))
    invh_range = np.append(invh_range, np.arange(1.5, 5, 0.05))
    invh_range = np.append(invh_range, np.arange(5, 10, 0.1))
    create_GaussianRing_lookuptable(invh_range, N=200, tol=0.01, printtime=True)

