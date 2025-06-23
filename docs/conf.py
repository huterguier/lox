# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'Lox'
copyright = '2025, huterguier'
author = 'huterguier'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_logo = '_static/lox.png'
html_favicon = '_static/favicon.png'

html_theme = 'sphinx_book_theme'
html_static_path = ['_static']
html_css_files = [
    'style.css',
]

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',       # for Google/Numpy docstrings
    'sphinx_autodoc_typehints',  # for type hints
    'myst_nb',
]

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}
