# Deep Learning for Science School Tutorial:<br>Deep Learning At Scale

This repository contains the material for the DL4Sci tutorial:
*Deep Learning at Scale*.

It contains specifications for datasets, a couple of CNN models, and
all the training code to enable training the models in a distributed fashion
using Horovod.

As part of the tutorial, you will train a ResNet model to classify images
from the CIFAR10 dataset on multiple nodes with synchronous data parallelism.

**Contents**
* [Links](https://github.com/NERSC/dl4sci-scaling-tutorial#links)
* [Installation](https://github.com/NERSC/dl4sci-scaling-tutorial#installation)
* [Navigating the repository](https://github.com/NERSC/dl4sci-scaling-tutorial#navigating-the-repository)
* [Hands-on walk-through](https://github.com/NERSC/dl4sci-scaling-tutorial#hands-on-multi-node-training-example)
* [Code references](https://github.com/NERSC/dl4sci-scaling-tutorial#code-references)

## Links

NERSC JupyterHub: https://jupyter-dl.nersc.gov

Slides: https://drive.google.com/drive/folders/10NqOLaqPTZ0nobE7JNSaGwrXQi_ACD35?usp=sharing

## Installation

1. Start a terminal on Cori, either via ssh or from the Jupyter interface.
    * **IMPORTANT: if using jupyter, you need to use a SHARED CPU. Click the CPU button instead of the GPU button to run this example!**
2. Clone the repository using git:\
   `git clone https://github.com/NERSC/dl4sci-scaling-tutorial.git`

That's it! The rest of the software (Keras, TensorFlow) is pre-installed on Cori
and loaded via the scripts used below.

## Navigating the repository

**`train_horovod.py`** - the main training script which can be steered with YAML
configuration files.

**`data/`** - folder containing the specifications of the datasets. Each dataset
has a corresponding name which is mapped to the specification in `data/__init__.py`

**`models/`** - folder containing the Keras model definitions. Again, each model
has a name which is interpreted in `models/__init__.py`.

**`configs/`** - folder containing the configuration files. Each
configuration specifies a dataset, a model, and all relevant configuration
options (with some exceptions like the number of nodes, which is specified
instead to SLURM via the command line).

**`scripts/`** - contains an environment setup script and some SLURM scripts
for easily submitting the example jobs to the Cori batch system.

**`utils/`** - contains additional useful code for the training script, e.g.
custom callbacks, device configuration, and optimizers logic.

## Hands-on multi-node training example

We will use a customized ResNet model in this example to classify CIFAR10
images and demonstrate distributed training with Horovod.

1. Check out the ResNet model code in [models/resnet.py](models/resnet.py).
   Note that the model code is broken into multiple functions for easy reuse.
   We provide here two versions of ResNet models: a standard ResNet50 (with 50
   layers) and a smaller ResNet consisting of 26 layers.
    * Identify the identy block and conv block functions. *How many convolutional
      layers do each of these have*?
    * Identify the functions that build the ResNet50 and the ResNetSmall. Given how
      many layers are in each block, *see if you can confirm how many layers (conv
      and dense) are in the models*. **Hint:** we don't normally count the
      convolution applied to the shortcuts.

2. Inspect the optimizer setup in [utils/optimizers.py](utils/optimizers.py).
    * Note how we scale the learning rate (`lr`) according to the number of
      processes (ranks).
    * Note how we construct our optimizer and then wrap it in the Horovod
      DistributedOptimizer.

3. Inspect the main [train_horovod.py](train_horovod.py) training script.
    * Identify the `init_workers` function where we initialize Horovod.
      Note where this is invoked in the main() function (right away).
    * Identify where we setup our distributed training callbacks.
    * *Which callback ensures we have consistent model weights at the start of training?*
    * Identify the callbacks responsible for the learning rate schedule (warmup and decay).

4. Finally, look at the configuration file:
   [configs/cifar10_resnet.yaml](configs/cifar10_resnet.yaml)
    * YAML allows to express configurations in rich, human-readable, hierarchical structure.
    * Identify where you would edit to modify the optimizer, learning-rate, batch-size, etc.

That's mostly it for the code. Note that in general when training distributed
you might want to use more complicated data handling, e.g. to ensure different
workers are always processing different samples of your data within a training
epoch. In this case we aren't worrying about that and are, for simplicity,
relying on the independent random shuffling of the data by each worker as well
as the random data augmentation.

5. To gain an appreciation for the speedup of training on
   multiple nodes, first train the ResNet model on a single node.
   Adjust the configuration in [configs/cifar10_resnet.yaml](configs/cifar10_resnet.yaml)
   to train for just 1 epoch and then submit the job to the Cori batch system with
   SLURM sbatch and our provided SLURM batch script:\
   `sbatch -N 1 scripts/cifar_resnet.sh`
    * **Important:** the first time you run a CIFAR10 example, it will
    automatically download the dataset. If you have more than one job attempting
    this download simultaneously it will likely fail.

6. Now we are ready to train our ResNet model on multiple nodes using Horovod
   and MPI! If you changed the config to 1 epoch above, be sure to change it back
   to 32 epochs for this step. To launch the ResNet training on 8 nodes, do:\
   `sbatch -N 8 scripts/cifar_resnet.sh`

7. Check on the status of your job by running `sqs`.
   Once the job starts running, you should see the output start to appear in the
   slurm log file `logs/cifar-cnn-*.out`. You'll see some printouts from every
   worker. Others are only printed from rank 0.

8. When the job is finished, check the log to identify how well your model learned
   to solve the CIFAR10 classification task. For every epoch you should see the
   loss and accuracy reported for both the training set and the validation set.
   Take note of the best validation accuracy achieved.

Now that you've finished the main tutorial material, try to play with the code
and/or configuration to see the effect on the training results. You can try changing
things like
* Change the optimizer (search for Keras optimizers on google).
* Change the nominal learning rate, number of warmup epochs, decay schedule
* Change the learning rate scaling (e.g. try "sqrt" scaling instead of linear)

Most of these things can be changed entirely within the configuration.
See [configs/imagenet_resnet.yaml](configs/imagenet_resnet.yaml) for examples.

## Code references

Keras ResNet50 official model:
https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py

Horovod ResNet + ImageNet example:
https://github.com/uber/horovod/blob/master/examples/keras_imagenet_resnet50.py

CIFAR10 CNN and ResNet examples:
https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
https://github.com/keras-team/keras/blob/master/examples/cifar10_resnet.py
