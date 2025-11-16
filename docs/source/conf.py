# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# docs/source/conf.py
import os, sys
sys.path.insert(0, os.path.abspath("../.."))

project = 'rotcurves'
copyright = '2025, Amit Nestor Shachar'
author = 'Amit Nestor Shachar'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",          # Google/NumPy docstrings
    "sphinx_autodoc_typehints",
    "myst_parser",                   # Markdown support
]

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "navigation_with_keys": True,
}
html_static_path = ['_static']
