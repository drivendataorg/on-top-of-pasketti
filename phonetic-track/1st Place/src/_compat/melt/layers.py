"""``melt.layers`` placeholder.

The upstream code only references ``melt.layers`` in two doc-strings inside
``models/base.py`` (``LinearAttentionPooling`` / ``NonLinearAttentionPooling``)
and never actually instantiates anything from it. Importing the empty module
keeps the syntactic ``from melt.layers import ...`` style intact for
downstream forks that may use it.
"""
