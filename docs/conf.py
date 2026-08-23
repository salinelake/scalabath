"""Sphinx configuration for the scalabath documentation."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

project = "scalabath"
author = "Pinchen Xie and Zhen Huang"
copyright = "2026, Pinchen Xie"  # noqa: A001

try:
    release = package_version("scalabath")
except PackageNotFoundError:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

needs_sphinx = "8.2"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autodoc_class_signature = "mixed"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

html_theme = "sphinx_rtd_theme"
html_title = f"scalabath {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "style_external_links": True,
}
html_context = {
    "display_github": True,
    "github_user": "salinelake",
    "github_repo": "scalabath",
    "github_version": "main",
    "conf_py_path": "/docs/",
}
