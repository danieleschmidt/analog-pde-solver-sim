"""
Sphinx configuration for Analog PDE Solver documentation.
"""

import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'Analog PDE Solver'
copyright = '2025, Analog PDE Team'
author = 'Analog PDE Team'
release = '0.1.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'myst_parser',
    'sphinx_rtd_theme',
    'sphinx.ext.coverage',
    'sphinx.ext.doctest',
]

# Source file parsers
source_suffix = {
    '.rst': None,
    '.md': 'myst_parser',
}

# The master toctree document
master_doc = 'index'

# Internationalization
language = 'en'

# List of patterns to ignore when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML output configuration
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

html_context = {
    "display_github": True,
    "github_user": "yourusername",
    "github_repo": "analog-pde-solver-sim",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_static_path = ['_static']
html_css_files = ['custom.css']

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# Math configuration
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
        'macros': {
            'laplacian': r'\nabla^2',
            'del': r'\nabla',
            'pd': r'\partial',
            'conductance': r'G',
            'voltage': r'V',
            'current': r'I'
        }
    }
}

# Coverage configuration
coverage_show_missing_items = True