Quick Start
==========

Basic usage example:

.. code-block:: python

   from epitopegen import EpitopeGenPredictor

   # Initialize predictor
   predictor = EpitopeGenPredictor()

   # Predict epitopes for TCR sequences
   tcrs = ["CASIPEGGRETQYF", "CAVRATGTASKLTF"]
   results = predictor.predict(tcrs)
