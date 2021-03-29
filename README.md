# CycleGAN Reimplementation in Tensorflow

This is a reimplementation of the cycle-consistent generative adversarial network (CycleGAN) by Zhu et al., 2017 (see References).
The project has been adopted as part of a university course assignment and is completed with by a paper summarising objectives and findings in ablation studies.
	
## Table of contents
* [Technical details](#technical-details)
* [Setup](#setup)
* [CycleGAN according to Zhu et al.](#cycle-gan)

## Technologies
Our project was created with:
* Tensorflow version: 2.4.1
	
## Setup
To run this project, open XYZ notebook in colab. The git repository is cloned in the notebook. 

## CycleGAN according to Zhu et al.


The repository includes implementations of a CycleGAN heavily based on the architectural choices proposed by Zhu et al. (see generator.py and discriminator.py).
Additionally, further model architectures are included in the folder models. First, as Zhu et al. cite Johnson et al., 2016 as reference for their generator architecture, we have re-implemented the network proposed by them as well. Besides, this includes implementation of the networks with instance instead of batch normalization.
The folder utilities comprises functions with image operations to print a pre-defined subset of test images, which offers a basis for comparison for further ablation studies. 

