Quick Start
==========

Basic usage example:

.. code-block:: python

   from epitopegen import EpiGenPredictor

   # Initialize predictor
   predictor = EpiGenPredictor()

   # Predict epitopes for TCR sequences
   tcrs = ["CASIPEGGRETQYF", "CAVRATGTASKLTF"]
   results = predictor.predict(tcrs)
