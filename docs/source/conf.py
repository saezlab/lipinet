# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'LipiNet'
copyright = '2025, Macabe Daley et al.'
author = 'Macabe Daley et al.'
release = '1.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",                      # MyST + notebook support
    # "myst_parser",
    "sphinx.ext.autodoc",           # pull in docstrings
    "sphinx.ext.napoleon",          # NumPy/Google style
    "sphinx.ext.autosummary",       # generate API tables
    "sphinx_copybutton",            # “copy” buttons on code blocks
    "sphinx_autodoc_typehints",     # show Python 3 type hints
    "sphinx_tabs.tabs",             # tabbed content
    "sphinx_design",                # cards, grids, dropdowns
    "sphinx.ext.intersphinx",       # links to external docs
    "sphinxcontrib.bibtex",         # if you have a references.bib
    "sphinx.ext.mathjax",           # render math
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",          # social-media previews
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "groupwise"

# MyST settings
myst_enable_extensions = [
    "amsmath", "colon_fence", "deflist", "dollarmath",
    "html_image", "html_admonition",
]
myst_heading_anchors = 6

# Bibtex
bibtex_bibfiles = ["references.bib"]

# Intersphinx targets
intersphinx_mapping = {
    "python":   ("https://docs.python.org/3", None),
    "numpy":    ("https://numpy.org/doc/stable/",     None),
    "pandas":   ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scanpy":   ("https://scanpy.readthedocs.io/en/stable/", None),
}

templates_path = ['_templates']
exclude_patterns = [
    "notebooks/.wip/*"
]

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))

# Detect RTD build
on_rtd = os.environ.get("READTHEDOCS") == "True"

# If we’re on RTD, never re‑execute; just render committed outputs.
# Otherwise (e.g. local dev) use cache mode so missing outputs still run.
if on_rtd:
    nb_execution_mode = "off"
else:
    nb_execution_mode = "cache"
nb_execution_timeout = 90  # Set timeout to 90 seconds

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ['_static']
html_css_files = ['image-fixes.css']

html_logo = '_static/.lipinet_logo_v1_0051.png'
html_favicon = '_static/.lipinet_logo_v1_0051.png'

html_theme_options = {
   "repository_url":        "https://github.com/saezlab/lipinet",
   "use_repository_button": True,
   "use_edit_page_button":  True,
   "path_to_docs":          "docs/source",
   "home_page_in_toc":      False,
   "show_navbar_depth":     2,
   # if you want the “Launch in Binder/Colab” buttons:
   "launch_buttons": {
     "colab_url":     "https://colab.research.google.com",
   },
 }