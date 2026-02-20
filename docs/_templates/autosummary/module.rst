{{ fullname | escape | underline }}

.. automodule:: {{ fullname }}

   {% if modules %}
   .. rubric:: Modules

   .. autosummary::
   {% for item in modules %}
      {{ item }}
   {% endfor %}
   {% endif %}

   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
      {{ item }}
   {% endfor %}
   {% endif %}

   {% if functions %}
   .. rubric:: Functions

   .. autosummary::
   {% for item in functions %}
      {{ item }}
   {% endfor %}
   {% endif %}

   {% if classes %}
   .. rubric:: Classes

   .. autosummary::
   {% for item in classes %}
      {{ item }}
   {% endfor %}
   {% endif %}

   {% if exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
   {% for item in exceptions %}
      {{ item }}
   {% endfor %}
   {% endif %}

.. bibliography::
   :filter: docname in docnames
