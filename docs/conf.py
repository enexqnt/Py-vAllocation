import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Py-vAllocation'
copyright = '2025, enexqnt'
author = 'enexqnt'

version = '0.3.0'
release = '0.3.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.todo',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'sphinxcontrib.bibtex',
    'nbsphinx',
]

bibtex_bibfiles = ['references.bib']
bibtex_reference_style = 'author_year' # or 'label' or 'numeric'
bibtex_cite_key_template = '{key}'

# Mock heavy dependencies so autodoc works even if they are not installed.
autodoc_mock_imports = [
    'numpy',
    'pandas',
    'scipy',
    'cvxopt',
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

exclude_patterns = [
    '_build', 'Thumbs.db', '.DS_Store',
    'tutorials/notebooks/*.ipynb',
]
source_suffix = ['.rst', '.ipynb']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    '__init__': True,
    'exclude-members': 'weights,marginal_risk,psi,gamma,rho,objective,target_return,max_return',
}

linkcheck_ignore = [
    r'https://doi\.org/10\.1093/.*',
    r'https://doi\.org/10\.2469/faj\.v48\.n5\.28',
    r'https://doi\.org/10\.2139/ssrn\.792628',
]
