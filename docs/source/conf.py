# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

"""Sphinx configuration for LipiNet docs (aligned with OnionNet)."""

# -- Project information -----------------------------------------------------

project = "LipiNet"
author = "Macabe Daley et al."
copyright = "2025, Macabe Daley et al."
release = "1.4.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",  # Markdown support
    "nbsphinx",  # Notebook support
    "sphinx.ext.autodoc",  # pull in docstrings
    "sphinx.ext.napoleon",  # NumPy/Google style
    "sphinx.ext.autosummary",  # generate API tables
    "sphinx_copybutton",  # copy buttons on code blocks
    "sphinx_tabs.tabs",  # tabbed content
    "sphinx_design",  # cards, grids, dropdowns
    "sphinx.ext.intersphinx",  # links to external docs
    "sphinxcontrib.bibtex",  # references.bib
    "sphinx.ext.mathjax",  # render math
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",  # social-media previews
]

# Autosummary and docstring settings
autosummary_generate = True
autodoc_typehints = "none"  # avoid duplicating types; let docstrings speak
napoleon_google_docstring = True
napoleon_numpy_docstring = True
autodoc_member_order = "groupwise"

# MyST-Parser settings (for Markdown)
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_heading_anchors = 6

# BibTeX
bibtex_bibfiles = ["references.bib"]

# Intersphinx targets
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "graph-tool": ("https://graph-tool.skewed.de/static/docs/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["notebooks/.wip/*"]

# Allow Markdown and RST sources, and set root document
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
try:
    root_doc = "index"
except NameError:
    master_doc = "index"

# Make sure package can be imported by Sphinx
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
# Preserve custom CSS if present
html_css_files = ["image-fixes.css"]

html_logo = "_static/.lipinet_logo_v1_0051.png"
html_favicon = "_static/.lipinet_logo_v1_0051.png"

html_theme_options = {
    "repository_url": "https://github.com/saezlab/lipinet",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs/source",
    "home_page_in_toc": False,
    "show_navbar_depth": 2,
}
