# ImageStyleTransfer

## Introduction
This repository implements style transfer methods by deep learning. You may be confused seeing repository's name and contents, it is natural because this repository mixes neural style transfer (ex. AdaIN) and image-to-image translation (ex. CycleGAN). I'm sorry. Anyway, from this point, I don't mention to neural style transfer. In image-to-image translation, I mainly implement the method of unpaired image-to-image translation. If I think of image-to-image translation on characters' images, it is interesting for me to convert hair color or hair style of character. However, if you do this in the system of paired image-to-image translation such as pix2pix, you'll find it difficult to collect paired data because there are few images of the same character in various hair colors. Therefore, I think it is better to implement unpaired image-to-image translation.

## Image-to-Image Translation Experiment

## Methods

- [x] CycleGAN
- [x] StarGAN
- [x] AdaIN
- [x] InstaGAN
- [x] StyleAttentionNet
- [x] U-GAT-IT
- [x] RelGAN
- [ ] CartoonGAN
- [ ] DRIT
- [ ] MUNIT
- [ ] FUNIT
- [ ] SoftAdaIN

## Quick Results
### CycleGAN
[Paper](https://arxiv.org/pdf/1703.10593.pdf "here")  
Size:128×128  
![CycleGAN](./CycleGAN/result.jpg)

### StarGAN
[Paper](https://arxiv.org/abs/1711.09020 "here")  
Size:128×128  
![StarGAN](./StarGAN/result_2.png)

### AdaIN
[Paper](https://arxiv.org/pdf/1703.06868.pdf)

![AdaIN](https://github.com/SerialLain3170/Style-Transfer/blob/master/AdaIN/images/anime.png)

### InstaGAN
[Paper](https://arxiv.org/pdf/1812.10889.pdf)

![InstaGAN](https://github.com/SerialLain3170/ImageStyleTransfer/blob/master/InstaGAN/result.png)

### U-GAT-IT
[Paper](https://arxiv.org/pdf/1907.10830.pdf)

![U-GAT-IT](https://github.com/SerialLain3170/ImageStyleTransfer/blob/master/UGATIT/Result.jpg)

### RelGAN
[Paper](https://arxiv.org/pdf/1908.07269.pdf)

![RelGAN](https://github.com/SerialLain3170/ImageStyleTransfer/blob/master/RelGAN/RelGAN_result.jpg)
