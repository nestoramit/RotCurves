from scipy.special import gammaincinv


if __name__ == '__main__':
    reff = 2.
    n = 1.
    b = lambda n: gammaincinv(2*n, 0.5)
    for n in [1., 2., 3., 4.]:
        print(b(n))
