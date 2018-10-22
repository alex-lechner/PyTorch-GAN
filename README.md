# PyTorch-GAN

[//]: #(References)
[cifar10]: https://www.cs.toronto.edu/~kriz/cifar.html
[gan-architecture]: ./imgs/gan-architecture.png
[gans-beginners-guide]: https://skymind.ai/wiki/generative-adversarial-network-gan
[gan-animation]: ./imgs/gan-animation.gif
[pytorch-install]: https://pytorch.org/

---

A Generative Adversarial Network is a technique to create artificial images with a Neural Network. The algorithm was invented by Ian Goodfellow and can be used for e.g. image enhancement, (artistic) style transfer, etc.

The GAN technique trains two Neural Networks simultaneously: A Generator network and a Discriminator network.

The Generator network is a Deconvolutional Neural Network which takes randomly generated noise as an input and returns a generated image as an output.
The Discriminator network is Convolutional Neural Network which is trained on fake images from the Generator network and real images from a given dataset ([CIFAR10][cifar10] was used in this project). The goal of the Discriminator network is to distinguish between real and fake images. The goal of the Generator network is to trick the Discriminator network by generating realistic fake images.

![gan-architecture][gan-architecture]

*Source: [A Beginner's Guide to Generative Adversarial Networks (GANs)][gans-beginners-guide]*


## Installation

For this project, only the PyTorch library is needed. To install PyTorch [visit the website and choose your specifications][pytorch-install].

## Training

To start training open navigate to the root folder of this project in your Command Line/Terminal and execute:

```sh
python dcgan.py
```

After each epoch, the current state of the Generator model and Discriminator model is saved. If you want to resume training at a given epoch you have to execute the following line:

```sh
python dcgan.py --resume_generator path/to/your-generator-model.pth --resume_discriminator path/to/your-discriminator-model.pth
```

To change the batch size simply change ```BATCH_SIZE = 64``` in `dcgan.py` on line 18 to the size you want.

To change the number of epochs simply change ```EPOCH_SIZE = 25``` in `dcgan.py` on line 20 to the number you want.

## Result

After training the GAN for 25 epochs (which took nearly 1 day on my local machine) the Generator network is almost able to generate objects. Even though the objects are still a bit abstract you can already guess what the Generator network attempts to generate in some images.

![gan-animation][gan-animation]

*GAN results after each epoch*