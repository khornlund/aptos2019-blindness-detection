===================================================================================================
`Aptos 2019 Blindness Detection <https://www.kaggle.com/c/aptos2019-blindness-detection/overview>`_
===================================================================================================

.. contents:: Table of Contents
   :depth: 2

Report
======

I came in `89th place (silver) <https://www.kaggle.com/c/aptos2019-blindness-detection/leaderboard>`_.

Below I'll detail some of the things I tried throughout the competition.



Class Balancing
---------------

I wrote a custom ``BatchSampler`` to use with PyTorch, in order to over/under sample the data
according to a parameter ``alpha``.

By choosing a value for ``alpha`` between 0 and 1, you can linearly change the sample distribution
between the true distribution (``alpha = 0``), and a uniform distribution (``alpha = 1``).

.. image:: ./resources/sample-distributions-2019-data.png

Note the extreme imbalance for the 2015 data.

.. image:: ./resources/sample-distributions-2015-data.png

Typically for training on the 2015 data I used an ``alpha`` value of 0.8, and for fine-tuning on
the 2019 data I used alpha values in the range 0.2 to 0.8.

User Guide
==========

Requirements
------------
* Python >= 3.6
* PyTorch >= 1.1
* Tensorboard >= 1.4

Folder Structure
----------------

::

  aptos2019-blindness-detection/
  │
  ├── aptos/
  │    │
  │    ├── cli.py - command line interface
  │    ├── main.py - main script to start train/test
  │    │
  │    ├── base/ - abstract base classes
  │    │   ├── base_model.py - abstract base class for models
  │    │   └── base_trainer.py - abstract base class for trainers
  │    │
  │    ├── data_loader/ - anything about data loading goes here
  │    │   └── data_loaders.py
  │    │
  │    ├── model/ - models, losses, and metrics
  │    │   ├── loss.py
  │    │   ├── metric.py
  │    │   └── model.py
  │    │
  │    ├── trainer/ - trainers
  │    │   └── trainer.py
  │    │
  │    └── utils/
  │        ├── logger.py - class for train logging
  │        ├── visualization.py - class for Tensorboard visualization support
  │        └── saving.py - manages pathing for saving models + logs
  │
  ├── logging.yml - logging configuration
  │
  ├── data/ - directory for storing input data
  │
  ├── experiments/ - directory for storing configuration files
  │
  ├── saved/ - directory for checkpoints and logs
  │
  └── tests/ - tests folder


Usage
-----

.. code-block:: bash

  $ conda env create --file environment.yml
  $ conda activate aptos

See ``notebooks/preprocess.ipynb`` to preprocess the data for training.

To start training, run:

.. code-block:: bash

  $ aptos train -c experiments/config.yml


Tensorboard Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~
This template supports `<https://pytorch.org/docs/stable/tensorboard.html>`_ visualization.

1. Run training

    Set `tensorboard` option in config file true.

2. Open tensorboard server

    Type `tensorboard --logdir saved/runs/` at the project root, then server will open at
    `http://localhost:6006` (if clicking the link doesn't work, paste this into your browser)


