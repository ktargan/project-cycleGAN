# CycleGAN Reimplementation in Tensorflow

This is a reimplementation of the cycle-consistent generative adversarial network (CycleGAN) by Zhu et al., 2017 (see References).
The project has been adopted as part of a university course assignment and is completed by a paper summarising objectives and results of ablations studies.
	
## Table of contents
* [Technical details](#technical-details)
* [CycleGAN according to Zhu et al.](#cycle-gan)
* [Setup](#setup)

## Technologies
Our project was created with:
* Tensorflow version: 2.4.1
	
## CycleGAN according to Zhu et al.

In their paper, Zhu \textit{et al.} propose a network that learns image-to-image translation without paired training examples. Therefore, their approach is valuable for settings where aligned input and output images are not readily available. To form a complete CycleGAN, Zhu \textit{et al.} rely on two Generative Adversarial Networks (GANs) and thus, exploit the advantages of adversarial losses in several ways. In the absence of supervision on the basis of mapped ground truth images, the model is trained to learn a distribution close to the original distribution underlying the output domain (through the discriminator classification). Thus, any single image of the input domain has a large corresponding space of mapping possibilities and the model does not just learn to create a single target image. To ensure that the model performs well nonetheless, Zhu \textit{et al.} enforce the cycle-consistency of translation through an additional loss objective. As a result, the model is more resistant to mode collapse and is able to translate between both domains. 

This repository includes implementations of a CycleGAN heavily based on the architectural choices proposed by Zhu et al. (see generator.py and discriminator.py).
Additionally, further model architectures are included in the folder models. First, as Zhu et al. cite Johnson et al., 2016 as reference for their generator architecture, we have re-implemented the network proposed by them as well. Besides, this includes implementation of the networks with instance instead of batch normalization.
The folder utilities comprises functions with image operations to print a pre-defined subset of test images, which offers a basis for comparison for further ablation studies. 

## Setup
The notebook "" offers the code for training the CycleGAN with an implementation that is close to the original architecture by Zhu \textit{et al.} on the horse2zebra dataset. It may be most convenient to open the notebook in google colab (and use the offered GPU).
As a first step in the notebook, the git repository is cloned for the current runtime (to have it permanently mount the google drive and clone the repository at the desired path). Next all relevant modules will be imported and the training is started.  
