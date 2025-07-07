{{ fullname }}
{{ "=" * fullname|length }}

.. autoclass:: {{ fullname }}
   :show-inheritance:
   :members:
{% if fullname == "lox.logdict" %}   :special-members: __add__, __or__ {% endif %}
   :undoc-members:

   {% block methods %}
   .. automethod:: __init__

   {% if methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      .. automethod:: {{ fullname }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
