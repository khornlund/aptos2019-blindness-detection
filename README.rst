=====
aptos
=====
`<https://www.kaggle.com/c/aptos2019-blindness-detection/overview>`_

.. contents:: Table of Contents
   :depth: 2

Requirements
============
* Python >= 3.6
* PyTorch >= 1.1
* Tensorboard >= 1.4

Folder Structure
================

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
=====

.. code-block:: bash

  $ conda env create --file environment.yml
  $ conda activate aptos

You can run the tests using:

.. code-block:: bash

  $ pytest tests

See ``notebooks/preprocess.ipynb`` to preprocess the data for training.

To start training, run:

.. code-block:: bash

  $ aptos train -c experiments/config.yml


Tensorboard Visualization
--------------------------
This template supports `<https://pytorch.org/docs/stable/tensorboard.html>`_ visualization.

1. Run training

    Set `tensorboard` option in config file true.

2. Open tensorboard server

    Type `tensorboard --logdir saved/runs/` at the project root, then server will open at
    `http://localhost:6006` (if clicking the link doesn't work, paste this into your browser)


TODO
====

1. Additional data sources eg. `<http://www.adcis.net/en/third-party/messidor/>`_
2. Alternative data processing techniques
3. Alternative augmentation techniques
4. Is robust loss better the mse? Better than classification?
5. Model arch: can we improve EfficientNet?


Acknowledgments
===============
This template is inspired by

  1. `<https://github.com/victoresque/pytorch-template>`_
  2. `<https://github.com/daemonslayer/cookiecutter-pytorch>`_
