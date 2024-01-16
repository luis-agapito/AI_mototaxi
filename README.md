# Convolutional Neural Networks: Recognition of mototaxis

The objective is to show the use of AI for detecting the presence/absence of mototaxis in complex environments. All necessary code is provided but the raw training data is not. Interested users are encouraged to build their own datasets.

<p align="middle">
  <img src="figures/mototaxi1.jpg" width="175" />
  <img src="figures/pic2.jpg" width="175" /> 
  <img src="figures/pic3.png" width="175" />
  <img src="figures/pic4.jpg" width="175" />
</p>

The remarkable success of neural networks is mostly due to two factors that have converged only recently, within the last decade: 1) the availability of large datasets and 2) access to large computational resources. Currently, developing and training new AI architectures from scratch would require a fair share of hardware (GPU) resources; nonetheless, here we show a workflow that can still be run and tested on the CPU of a regular laptop for illustrating the variety of concepts involved.

## Pytorch implementation:
1. Preprocessing the raw images.
    - For convenience all image crops are made larger than 224x224 pixels for MobileNet and square (same height and width) to avoid padding.
    - Some cameras add rotation flags which will rotate the images when visualizing them in Python.
    - Helper utilities for the task of manually cropping camera images. See `notebooks/manual_cropping.py`.
2. [DataLoaders: Pipeline for feeding the data.](http://nbviewer.org/github/luis-agapito/AI_mototaxi/blob/main/notebooks/data_loaders.ipynb?flush_cache=True)
    - Partitioning the data into three datasets: train, validation (aka, development), and test.
    - Building the Dataloader.
    - Inspecting the Dataloader: Verify the images comprising minibatches for each epoch.
3. Customizing MobileNet [^1].
    - [Evaluating performance of the default model.](http://nbviewer.org/github.com/luis-agapito/AI_mototaxi/blob/main/notebooks/default_model.ipynb?flush_cache=True)
    - [Fine-tuning the model. Two cases: single and multiple layers.](http://nbviewer.org/github.com/luis-agapito/AI_mototaxi/blob/main/notebooks/custom_model_training.ipynb?flush_cache=True)
4. [Evaluating the performance of the fine-tuned model.](http://nbviewer.org/github.com/luis-agapito/AI_mototaxi/blob/main/notebooks/custom_model_verification.ipynb?flush_cache=True)

We find that **model2** achieved ~95% of accuracy. For example, it correctly predicted mototaxi/no_mototaxi for all the 16 random pictures shown below.


<img src="figures/model2.jpg">


## [Tensorflow-Keras implementation](http://nbviewer.org/github/luis-agapito/AI_mototaxi/blob/main/notebooks/tensorflow_mototaxi.ipynb?flush_cache=True) :
1. Custom model, case1.
    - Splitting into train, validation, and test datasets.
    - Augmenting the dataset.
    - Training using `train_dataset`.

2. Custom model, case2.
    - Training using `train_dataset`.
    - Validation using `test_dataset`.



## References
[^1]: [paper](https://arxiv.org/abs/1801.04381), [implementation](https://pytorch.org/hub/pytorch_vision_mobilenet_v2/)
