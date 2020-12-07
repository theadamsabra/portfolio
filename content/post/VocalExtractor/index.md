---
title: Vocal Extractor
subtitle: Using Neural Networks to Extract Vocals from Songs
---
# Introduction

The challenge of audio source separation has always been a challenging one for audio engineers. Audio source separation is often referred to as the cocktail party problem, where one is attending a cocktail party and honing in on one conversation among the dozens around them. This problem, for many years, was considered to be nearly impossible. However, with the optimization of Convolutional Neural Networks (CNNs) (via. AlexNet,) the problem's solution approached with the new techniques that arose from neural networks. In this post, I aim to break down how to use a variant of the CNN on [DSD100](https://sigsep.github.io/datasets/dsd100.html) to extract the sounds provided by the dataset.

## About the Data Set

DSD100 is a Deep Learning dataset that allows for researchers in the audio space to work on audio source separation. In it, contains 100 songs. Each song contains its final mixed version and its four stems: bass, vocals, drums, and other sounds. The mixed version acts as the input, $x$, whereas one of the four stems - in our case, the vocals - acts as our output, $y$.

# Processing the Data

To process the data for relevant results, digital signal processing techniques must be used to extract meaningful features for the neural network to understand and recognize. These techniques are predominately Fourier-based spectrograms.

# Building the U-Net Convolutional Network

Thanks to the research of [Jansson et al.]()
