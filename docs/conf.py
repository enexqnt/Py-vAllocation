# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Py-vAllocation'
copyright = '2025, enexqnt'
author = 'enexqnt'

version = '0.1'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
]

# Map common type aliases so cross references resolve properly.
napoleon_type_aliases = {
    'ndarray': 'numpy.ndarray',
    'np.ndarray': 'numpy.ndarray',
    'npt.NDArray': 'numpy.ndarray',
    'np.floating': 'numpy.floating',
    'Series': 'pandas.Series',
    'DataFrame': 'pandas.DataFrame',
    'pd.Series': 'pandas.Series',
    'pd.DataFrame': 'pandas.DataFrame',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True
