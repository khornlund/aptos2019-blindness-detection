===================================================================================================
`Aptos 2019 Blindness Detection <https://www.kaggle.com/c/aptos2019-blindness-detection/overview>`_
===================================================================================================

.. contents:: Table of Contents
   :depth: 3

Report
======

I received a silver medal for coming in `89th place (top 3%) <https://www.kaggle.com/c/aptos2019-blindness-detection/leaderboard>`_.

Overview
--------

1. `EfficientNet <https://github.com/lukemelas/EfficientNet-PyTorch>`_ (pretrained on ImageNet)
2. Regression
3. Variety of preprocessing/augmentation techniques
4. Train on 2015 train + test data
5. Fine-tune on 2019 data
6. Ensemble 5 models

Network + Training Loop
-----------------------

Architecture
~~~~~~~~~~~~
I used the ``EfficientNet-b2`` primarily. I experimented with the ``b3`` but didn't notice a
significant improvement, and larger models were limited by using my GPU at home.

I tried using `Apex <https://github.com/NVIDIA/apex>`_ to train using half-precision but had
problems saving/loading models, and problems installing ``Apex`` in the Kaggle kernels.

I was impressed people were able to use ``b5`` in the kernel with large batch size by using
FastAI's half precision functionality. In future I'll investigate this more.

Loss
~~~~
I experimented with using
`Robust Loss <https://github.com/jonbarron/robust_loss_pytorch>`_, which was introduced at CVPR
recently. I found this didn't perform as well as ``MSE``.

Optimizer
~~~~~~~~~
I experimented with ``Adam``, ``RMSProp``, and ``SGD`` early on and found ``Adam`` to perform best.

Later on I switched to using
``Ranger <https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer>`_, which is
``RAdam`` + ``LookAhead``.

Data
----
I pretrained on the train + test data from 2015, before fine-tuning on the 2019 training set.

Preprocessing & Augmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
I used a variety of preprocessing an augmentation techniques, which I'll outline in more detail
below.

Sampling Strategy
~~~~~~~~~~~~~~~~~
The data for this competition had quite imbalanced classes, so I wrote a custom ``PyTorch``
``BatchSampler`` to help with this problem.

See ``aptos.data_loader.sampler`` for implementation details.

Class Balancing
***************
Based on the choice of an ``alpha`` parameter in ``[0, 1]`` the sampler would adjust the sample
distribution to be between true distribution (``alpha = 0``), and a uniform distribution
(``alpha = 1``).

Overrepresented classes would be undersampled, and underrepresented classes oversampled.

.. image:: ./resources/sample-distributions-2019-data.png

Note the extreme imbalance for the 2015 data.

.. image:: ./resources/sample-distributions-2015-data.png

Typically for training on the 2015 data I used an ``alpha`` value of 0.8, and for fine-tuning on
the 2019 data I used alpha values in the range 0.2 to 0.8.

Standardised Batches
********************
Each sample generated would contain exactly the specified proportion of classes.

Here are a few sample batches of labels from a sampler with ``alpha = 0.5`` and ``batch_size = 32``

.. code::

    Batch: 0
    Classes: [1, 0, 0, 0, 2, 4, 0, 2, 0, 0, 3, 2, 1, 0, 2, 0, 0, 3, 0, 0, 4, 4, 0, 2, 1, 3, 3, 1, 2, 0, 0, 4]
    Counts: {0: 14, 1: 4, 2: 6, 3: 4, 4: 4}

    Batch: 1
    Classes: [4, 1, 1, 2, 0, 0, 0, 4, 2, 4, 0, 3, 1, 3, 0, 0, 3, 2, 0, 2, 4, 2, 0, 0, 2, 3, 0, 1, 0, 0, 0, 0]
    Counts: {0: 14, 1: 4, 2: 6, 3: 4, 4: 4}

    Batch: 2
    Classes: [0, 4, 0, 0, 0, 3, 3, 2, 0, 4, 2, 3, 0, 3, 2, 0, 0, 1, 2, 2, 0, 1, 0, 0, 4, 0, 2, 1, 1, 4, 0, 0]
    Counts: {0: 14, 1: 4, 2: 6, 3: 4, 4: 4}

Note that the class counts are the same for each batch. I found this helped training converge
faster, and my models generalised better. It was also a way to create diversity of models trained
with the same architecture - much like how people use varying image sizes.


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


