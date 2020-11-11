# Ultrasound Dataset Augmentation Using Pix2Pix
### Final Project for Duke University BME590D: Deep Learning Applications in Healthcare
### Sarah Putney

This project implements the conditional GAN pix2pix outlined in the paper [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) using tensorflow and keras, and applies it towards performing augmentation of the [OASBUD](https://zenodo.org/record/545928#.X6wf12hKg2w) (Open Access Series of Breast Ultrasound Data) dataset.

Try out the pre-trained models/tasks in [this](https://drive.google.com/file/d/1VWbzFvD7gPKUssgn_74vcLiHbPqlvvsz/view?usp=sharing) Google Colab notebook, and train models using the code from this repository in [this](https://drive.google.com/file/d/17il7c6-KVO3CM5MGkBQc9E5pIfYX2esn/view?usp=sharing) one.

The architectures used in the project are in line with that of the paper: the UNet architecture and a 70x70 PatchGAN. Images generated by the implemented GAN on the Edges2Shoes and Cityscape datasets (below) approximate the quality published in the original paper.

![edges2shoes generated images](https://github.com/sputney13/pix2pix_ultrasound/blob/main/img/edge2shoe.PNG)
![cityscape generated images](https://github.com/sputney13/pix2pix_ultrasound/blob/main/img/cityscapes.PNG)

B Mode images generated using the masks from the OASBUD dataset appeared to being noise-smoothed versions of the true image.

![b mode generated image](https://github.com/sputney13/pix2pix_ultrasound/blob/main/img/oasbud_gen.PNG)

Augmenting the dataset for classification (trained on EfficientNetB0) improved accuracy by approximately 7% to 55%.

Augmenting the dataset for segmentation (trained on UNet) improved accuracy by approximately 7% to 94%. Examples of segmentation performance with and without augmentation are included below:

| With Augmentation        | Without Augmentation           |
|:------------------------:|:------------------------------:|
| ![aug seg][aug 1]        | ![no aug seg][no aug 1]        |
| ![aug seg][aug 2]        | ![no aug seg][no aug 2]        |
| ![aug seg][aug 3]        | ![no aug seg][no aug 3]        |

[aug 1]: https://github.com/sputney13/pix2pix_ultrasound/blob/main/img/aug_1.PNG
[aug 2]: https://github.com/sputney13/pix2pix_ultrasound/blob/main/img/aug_2.PNG
[aug 3]: https://github.com/sputney13/pix2pix_ultrasound/blob/main/img/aug_3.PNG
[no aug 1]: https://github.com/sputney13/pix2pix_ultrasound/blob/main/img/no_aug_1.PNG
[no aug 2]: https://github.com/sputney13/pix2pix_ultrasound/blob/main/img/no_aug_2.PNG
[no aug 3]: https://github.com/sputney13/pix2pix_ultrasound/blob/main/img/no_aug_3.PNG
