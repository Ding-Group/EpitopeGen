# Configuration file for the Sphinx documentation builder.

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))  # Important for finding your package

project = 'EpiGen'
copyright = '2024, Minuk Ma'
author = 'Minuk Ma'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',      # For automatically including docstrings
    'sphinx.ext.napoleon',     # For Google/NumPy style docstrings
    'sphinx.ext.viewcode',     # For viewing source code
    'sphinx.ext.githubpages',  # For GitHub Pages
    'myst_parser'              # For markdown support
]

html_theme = 'sphinx_rtd_theme'  # Read the Docs theme
html_static_path = ['_static']
