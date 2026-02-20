import os
import sys
from collections import Counter
sys.path.insert(0, os.path.abspath('../src'))

import openMWR
from pybtex.plugin import register_plugin
from pybtex.style.formatting.unsrt import Style as UnsrtStyle
from pybtex.style.labels import BaseLabelStyle


class AuthorYearLabelStyle(BaseLabelStyle):
    """Labels like 'Rosenkranz, 2017' for bibliography list entries."""

    @staticmethod
    def _person_label(person):
        last = " ".join(person.prelast_names + person.last_names).strip()
        if last:
            return last
        return str(person).split(",")[0].strip()

    def _base_label(self, entry):
        people = entry.persons.get("author") or entry.persons.get("editor") or []
        year = entry.fields.get("year", "n.d.")

        if not people:
            key = entry.fields.get("key", entry.key)
            return f"{key}, {year}"
        if len(people) == 1:
            return f"{self._person_label(people[0])}, {year}"
        if len(people) == 2:
            return f"{self._person_label(people[0])} and {self._person_label(people[1])}, {year}"
        return f"{self._person_label(people[0])} et al., {year}"

    def format_labels(self, sorted_entries):
        labels = [self._base_label(entry) for entry in sorted_entries]
        counts = Counter(labels)
        seen = Counter()
        for label in labels:
            if counts[label] == 1:
                yield label
            else:
                suffix = chr(ord("a") + seen[label])
                seen[label] += 1
                yield f"{label}{suffix}"


class AuthorYearFormattingStyle(UnsrtStyle):
    default_sorting_style = "author_year_title"
    default_label_style = "openmwr_author_year"


register_plugin("pybtex.style.labels", "openmwr_author_year", AuthorYearLabelStyle)
register_plugin("pybtex.style.formatting", "openmwr_author_year", AuthorYearFormattingStyle)


project = 'openMWR'
html_title = "openMWR" #openMWR Documentation
copyright = f'2026, openMWR developers' #{datetime.datetime.now().year}
author = 'openMWR developers'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',                # For Jupyter notebooks
    'sphinx.ext.autodoc',      # Reads docstrings from your code
    "sphinx.ext.autosummary",
    "sphinxcontrib.bibtex",
    'sphinx.ext.napoleon',     # Supports Google & NumPy docstring style
    'sphinx.ext.viewcode',     # Shows source code links
    #'myst_parser',             # For Markdown pages
    "sphinx_copybutton",
]

templates_path = ['../_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['../_static']
html_css_files = ["../_static/custom.css"]

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@4/tex-mml-chtml.js"


nb_execution_mode = "off"

html_theme_options = {
    "header_links_before_dropdown": 8,
    "footer_start": ["sphinx-version"],
    "footer_center": ["copyright"],
    "footer_end": ["theme-version"],
    #"navbar_align": "left",
    #"search_as_you_type": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/D.Schleebruegge/openMWR",
            "icon": "fab fa-github",   # Font Awesome brand icon
            "type": "fontawesome",
        },
    ],
}

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# AutoDoc configuration
autosummary_generate = True
autodoc_typehints = "none" 
#autodoc_typehints = "description"
autosummary_generate_overwrite = True  # Overwrites old files when needed
autoclass_content = "both"  # Class docstring + __init__ docstring
#autosummary_imported_members = True  # Documents imported symbols as well
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"
bibtex_default_style = "openmwr_author_year"
bibtex_bibliography_header = ".. rubric:: References"


autodoc_default_options = {
    'members': True,             # Documents class methods, etc.
    'undoc-members': True,       # Also shows undocumented members
    'show-inheritance': True,    # Shows class inheritance
    'member-order': 'bysource',
    #'inherited-members': True,   # Shows inherited methods
}


#autosummary_output_dir = os.path.join("api")


# The master toctree document.
master_doc = "index"

autodoc_inherit_docstrings = False
