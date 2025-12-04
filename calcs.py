import numpy as np
import os
from scipy.special import gammaincinv

from tests.test_baryons import *
from tests.test_rotation_curve import *

if __name__ == '__main__':
    test_oversample_rebinning_consistency_odd()
