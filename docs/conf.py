# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from docutils import nodes

# Add the project root directory to PYTHONPATH:
sys.path.insert(0, os.path.abspath("../"))

project = "jaxFlowSim"
copyright = "2025, Diego Renner"
author = "Diego Renner"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = "alabaster"
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
latex_engine = "xelatex"  # pdflatex is default; alternatives: 'xelatex', 'lualatex'
latex_elements = {
    "classoptions": ",oneside",
}

def remove_gifs_in_latex(app, doctree, docname):
    if app.builder.format == 'latex':
        for node in doctree.traverse(nodes.image):
            if node['uri'].endswith('.gif'):
                node.parent.remove(node)

def setup(app):
    app.connect('doctree-resolved', remove_gifs_in_latex)
