Tree I/O
========

This example maps to ``examples/tree_io.py`` and demonstrates loading a sample
tree file through the optional ``phylognn.io`` boundary.

Inputs
------

- Repository sample data under ``examples_data/simulated_trees/``.
- Optional DendroPy-backed tree I/O dependencies.

Actions
-------

Run the script from the repository root:

.. code-block:: bash

   python examples/tree_io.py

The script checks whether ``dendropy`` is importable before running the
file-loading section. When available, it reads one sample tree and attaches a
small feature set.

Expected outputs
----------------

With DendroPy installed, the script prints a ``Tree I/O summary`` with the
loaded file, tip count, and selected root features. Without DendroPy, it prints
an actionable optional-dependency message and exits with code 0.

Failure modes
-------------

Missing sample files or non-DendroPy parsing errors are raised normally so the
underlying issue remains visible.

Optional settings
-----------------

Install optional tree I/O dependencies with ``python -m pip install -e ".[beast]"``.

Source
------

.. literalinclude:: ../../../examples/tree_io.py
   :language: python
