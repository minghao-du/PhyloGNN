Models Reference
================

Import path
-----------

.. code-block:: python

   from phylognn.models import (
       BaseGATNet,
       BasePhyloGNN,
       GATBiLSTMNet,
       PGLSRegressionHead,
       TemporalBiLSTMEncoder,
   )

Base classes
------------

.. py:class:: BasePhyloGNN

   Abstract base class for phylogenetic graph models.

   .. py:method:: validate_data(data, require_batch=False)

      Validate `x`, `edge_index`, and optionally `batch`.

   .. py:method:: get_num_parameters(trainable_only=True)

      Count model parameters.

   .. py:method:: freeze_encoder()

      Freeze modules returned by `get_encoder_modules()`.

   .. py:method:: unfreeze_all()

      Make all parameters trainable.

.. py:class:: BaseGATNet(input_dim, preprocess_dim, gat_hidden_dim=64, gat_heads=4, num_gat_layers=3, dropout_prob=0.2, use_preprocessing=True, encoder_type="res_gat")

   Reusable GAT encoder base with optional feature preprocessing.

End-user model
--------------

.. py:class:: GATBiLSTMNet(input_dim, output_dim, preprocess_dim=32, gat_hidden_dim=64, gat_heads=4, num_gat_layers=3, dropout_prob=0.2, use_preprocessing=True, encoder_type="res_gat", temporal_mode="lstm", num_time_bins=None, temporal_hidden_dim=128, temporal_fc_hidden_dims=None, num_lstm_layers=2, temporal_aggregation="mean", graph_pool="sum", head_hidden_dim=64, output_positive=False)

   End-user model combining `GAT` node encoding with graph pooling, `FC`
   temporal encoding, or `BiLSTM` temporal encoding.

   Inputs follow the graph field contract documented in
   :doc:`../user_guide/graph_conversion`. The output tensor has shape
   `[batch_size, output_dim]`.

PGLS regression head
--------------------

.. py:class:: PGLSRegressionHead(input_dim, output_dim)

   Trainable linear projection from ordered leaf representations ``[N, D]``
   to trait predictions ``[N, T]``. ``input_dim`` and ``output_dim`` must be
   positive. Inputs must be finite ``float32`` or ``float64`` tensors whose
   dtype and device match the head parameters; the output preserves both and
   no implicit conversion is performed.

   .. py:method:: forward(representations)

      Validate a nonempty two-dimensional tensor with width ``input_dim`` and
      return predictions with ``output_dim`` traits without changing leaf row
      order. Covariance data is supplied separately to
      ``phylognn.training.PGLSLoss``.

Temporal encoder
----------------

.. py:class:: TemporalBiLSTMEncoder(input_dim, hidden_dim, num_layers=1, dropout_prob=0.0, aggregation="mean")

   Bidirectional LSTM temporal encoder exported as a stable public component.

Excluded internals
------------------

Low-level layers such as `GATBlock`, `ResidualGATStack`, `PositionalEncoding`,
and `MLPHead` are not package-level user APIs.

Related guide
-------------

See :doc:`../user_guide/training` and :doc:`../user_guide/graph_conversion`.
