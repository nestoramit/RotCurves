import numpy as np
import os
from RotCurves.const import LOOKUP_TABLES_PATH


if __name__ == '__main__':
    n = 4.
    q0 = 1.
    old = np.load(os.path.join(LOOKUP_TABLES_PATH, "Noordermeer_lookup_tables", f"noor_n{n:.2f}_q{q0:.2f}.npy"))
    new = np.load(os.path.join(LOOKUP_TABLES_PATH, "Noordermeer_lookup_tables", f"noor_n{n:.2f}_q{q0:.2f}.npy"))

    chk = old.transpose()[1] - new.transpose()[1]