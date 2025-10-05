# docs/conf.py
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve()                 # .../RotCurves/docs/source/conf.py
PKG_PARENT = HERE.parents[3]                    # .../rotationcurves  (parent of RotCurves/)
sys.path.insert(0, str(PKG_PARENT))             # put repo root first on sys.path

print(f"[conf.py] added to sys.path: {PKG_PARENT}")  # shows up in build log

project = "RotCurves"
author = "RotCurves"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
]

# NumPy-style docstrings
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# Autosummary/autodoc defaults
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"

# If your code imports heavy deps, mock them to prevent build failures
autodoc_mock_imports = [
    "time",
    "numpy",
    "scipy", "scipy.integrate",
    "astropy", "astropy.cosmology",
    "seaborn",
    "warnings",
    "logging"
]
html_theme = "sphinx_rtd_theme"
