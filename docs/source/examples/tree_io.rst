Tree I/O
========

Script: ``examples/tree_io.py``.

Inputs
------

- Repository sample data under ``examples_data/simulated_trees/``.
- Optional DendroPy-backed tree I/O dependencies.

Run command
-----------

Run the script from the repository root:

.. code-block:: bash

   python examples/tree_io.py

The script checks whether ``dendropy`` is importable before running the
file-loading section. When available, it reads one sample tree and attaches a
small feature set.

Expected output
---------------

With DendroPy installed, the script prints a ``Tree I/O summary`` with the
loaded file, tip count, and selected root features. Without DendroPy, it prints
an actionable optional-dependency message and exits with code 0.

Stable stdout markers include:

.. code-block:: text

   Tree I/O summary
   Loaded tree file:
   Optional dependency missing: dendropy

Files written
-------------

None.

Optional dependencies
---------------------

Install optional tree I/O dependencies with
``python -m pip install -e ".[beast]"``.

Failure modes
-------------

Missing sample files or non-DendroPy parsing errors are raised normally so the
underlying issue remains visible.

Source
------

.. literalinclude:: ../../../examples/tree_io.py
   :language: python
