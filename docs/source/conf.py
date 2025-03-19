# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LipiNet'
copyright = '2025, Macabe Daley et al.'
author = 'Macabe Daley et al.'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_nb",
]

templates_path = ['_templates']
exclude_patterns = [
    "notebooks/.wip/*"
]

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

nb_execution_timeout = 90  # Set timeout to 90 seconds
nb_execution_mode = "cache"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

html_logo = '_static/.lipinet_logo_v1_0051.png'
