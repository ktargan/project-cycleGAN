# CycleGAN Reimplementation in Tensorflow

This is a reimplementation of the cycle-consistent generative adversarial network (CycleGAN) by Zhu et al., 2017 (see References).
The project has been adopted as part of a university course assignment and is completed by a paper summarising objectives and results of ablations studies.
	
## Table of contents
* [Technical details](#technical-details)
* [CycleGAN based on Zhu et al.](#cycle-gan)
* [Setup](#setup)
* [Results](#results)
* [References](#references)

## Technical details
Our project was created with:
* Tensorflow version: 2.4.1
	
## CycleGAN based on Zhu et al.

In their paper, Zhu et al. propose a network that learns image-to-image translation without paired training examples. Therefore, their approach is valuable for settings where aligned input and output images are not readily available. To form a complete CycleGAN, Zhu et al. rely on two Generative Adversarial Networks (GANs) and thus, exploit the advantages of adversarial losses in several ways. In the absence of supervision on the basis of mapped ground truth images, the model is trained to learn a distribution close to the original distribution underlying the output domain (through the discriminator classification). Thus, any single image of the input domain has a large corresponding space of mapping possibilities and the model does not just learn to create a single target image. To ensure that the model performs well nonetheless, Zhu et al. enforce the cycle-consistency of translation through an additional loss objective. As a result, the model is more resistant to mode collapse and is able to translate between both domains. 


![grafik](https://user-images.githubusercontent.com/64196273/113331736-98662680-9320-11eb-8c1a-39d006f0f60e.png)

Illustration by Zhu et al., 2017: Visualizes both the structure of the GAN architecture as well as the cycle-consistency component. The cycle consistency loss is split into two components as illustrated in b) and c). An image of domain X is translated to domain Y by generator G. As a second step the generated image is passed to generator F to recreate an image of the original domain X. On this basis one side of the cycle consistency loss can be computed.


This repository includes implementations of a CycleGAN heavily based on the architectural choices proposed by Zhu et al. (see generator.py and discriminator.py).
Additionally, further model architectures are included in the folder models. First, as Zhu et al. cite Johnson et al., 2016 as reference for their generator architecture, we have re-implemented the network proposed by them as well. Besides, this folder includes implementation of the networks with instance instead of batch normalization.
The folder utilities comprises functions with image operations to print a pre-defined subset of test images, which offers a basis for comparison for further ablation studies. 

## Setup
The notebook "training" offers the code for training the CycleGAN with an implementation that is close to the original architecture by Zhu et al. on the horse2zebra dataset (or alternatively a custom dataset). It may be most convenient to open the notebook in google colab (and use the offered GPU) or another cloud computing service.
As a first step in the notebook, the git repository is cloned for the current runtime (to have it permanently mount the google drive and clone the repository at the desired path). Next all relevant modules will be imported and the training is started.

## Results 
Some tables providing the results of training the model on different datasets are provided in the folder 'results'. The file 'fantasy_worlds' showcases images that were generated by the respective model after 30 epochs. In the 'classic' file, you will find images that were generated after training for 100 epochs on the horse2zebra dataset.

## References

Jun-Yan Zhu and Taesung Park and Phillip Isola and Alexei A. Efros (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. CoRR, abs/1703.10593.

Johnson, J., Alahi, A., & Fei-Fei, L. (2016). Perceptual losses for real-time style transfer and super-resolution. In European Conference on Computer Vision.
