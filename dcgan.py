import os
import json
import argparse
import logging
import logging.config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from neural_net import Generator, Discriminator

BATCH_SIZE = 64
IMAGE_SIZE = 64  # img size will be 64x64
EPOCH_SIZE = 25


def load_logging_config():
    """
    Loads the config file for logging.
    """
    if not os.path.exists('log'):
        os.makedirs('log')
    try:
        with open('logging_config.json', 'r', encoding='utf-8') as config:
            logging.config.dictConfig(json.load(config))
    except FileNotFoundError as e:
        logging.basicConfig(filename='./log/error.log',
                            level=logging.INFO,
                            format='%(asctime)s %(name)-12s %(levelname)-8s {%(pathname)s:%(funcName)s:%(lineno)d} %(message)s',
                            datefmt='%d.%m.%Y %H:%M:%S')
        console = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(name)-12s: %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)
        logging.exception(e)


def args_parser():
    """
    Parse mandatory and optional arguments
    """
    parser = argparse.ArgumentParser(
        description='Generative Adversarial Network training with Pytorch')
    parser.add_argument('--resume_generator', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--resume_discriminator', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from')
    return parser.parse_args()


def load_dataset(filepath='./data'):
    """loads and transforms the image dataset CIFAR10
    :param filepath: Folder of the dataset (default: data)
    :return: Images organized in mini batches
    """
    transformer = transforms.Compose([
        transforms.Scale(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    dataset = dset.CIFAR10(root=filepath, download=True, transform=transformer)
    return DataLoader(dataset, batch_size=BATCH_SIZE,
                      shuffle=True, num_workers=2)


def weights_init(neural_net):
    """ Initializes the weights of the neural network
    :param neural_net: (De-)Convolutional Neural Network where weights should be initialized
    """
    classname = neural_net.__class__.__name__
    if classname.find('Conv') != -1:
        neural_net.weight.data.normal_(0, 2e-2)
    elif classname.find('BatchNorm') != -1:
        neural_net.weight.data.normal_(1, 2e-2)
        neural_net.bias.data.fill_(0)


def save_model(state_dict, name, epoch):
    """Saves the trained neural net, optimizer and epoch
    :param state_dict: Dictionary of states from the (De-)Convolutional Neural Network, optimizer & epoch
    :param name: Name of the Neural Network
    :param epoch: Current epoch
    """
    logging.getLogger(__name__).info(
        'Saving trained model {} at epoch {}.'.format(name, epoch))
    if not os.path.exists('model'):
        os.makedirs('model')
    model_name = 'dcgan_{}_{}.pth'.format(name, epoch)
    model_path = os.path.join('./model', model_name)
    torch.save(state_dict, model_path)


def load_model(filepath, neural_net, optimizer):
    """Loads a trained neural net, optimizer and epoch
    :param filepath: Path of the saved state dict
    :param neural_net: The Neural Net of class Generator or Discriminator
    :param optimizer: Optimizer function of the Neural Network
    :return: Loaded Neural Network, optimizer and epoch 
    """
    if os.path.isfile(filepath):
        ckpt = torch.load(filepath)
        neural_net.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        logging.getLogger(__name__).info('Load trained model {} at epoch {} and resume training'.format(
            neural_net.__class__, ckpt['epoch']))
        return neural_net, optimizer, ckpt['epoch']


def save_images(imgs, name, epoch):
    """Saves images
    :param imgs: Images
    :param name: Name for images
    :param epoch: Current epoch
    """
    logging.getLogger(__name__).info(
        'Saving images {} at epoch {}.'.format(name, epoch))
    if not os.path.exists('results'):
        os.makedirs('results')
    img_name = '{}_{:03d}.png'.format(name, epoch)
    img_path = os.path.join('./results', img_name)
    vutils.save_image(imgs, img_path, normalize=True)


def train(args, dataloader):
    """Trains Generator Neural Network and Discriminiator Neural Network
    :param dataloader: DataLoader object with images
    """
    logger = logging.getLogger(__name__)

    logger.debug('Create Generator and Discriminitaor and initialize weights.')
    net_generator = Generator().apply(weights_init)
    net_discriminator = Discriminator().apply(weights_init)
    optimizer_generator = optim.Adam(net_generator.parameters(),
                                     lr=2e-4,
                                     betas=(.5, .999))
    optimizer_discriminator = optim.Adam(net_discriminator.parameters(),
                                         lr=2e-4,
                                         betas=(.5, .999))

    start_epoch = 0
    if args.resume_generator:
        net_generator, optimizer_generator, start_epoch = load_model(args.resume_generator,
                                                                     net_generator,
                                                                     optimizer_generator)
    if args.resume_discriminator:
        net_discriminator, optimizer_discriminator, start_epoch = load_model(args.resume_discriminator,
                                                                             net_discriminator,
                                                                             optimizer_discriminator)

    criterion = nn.BCELoss()
    max_data = len(dataloader)
    epochs = np.array(
        [i for i in range(start_epoch, EPOCH_SIZE)], dtype=np.uint8)

    for epoch in np.nditer(epochs):
        for idx, data in enumerate(dataloader, 0):
            ## 1. update weights of discriminator in respect of the gradient
            net_discriminator.zero_grad()

            ## train discriminiator on real images
            real_img = Variable(data[0])
            batch_size = real_img.size()[0]
            ones = Variable(torch.ones(batch_size))
            output = net_discriminator.forward(real_img)
            real_img_error = criterion(output, ones)

            ## train discriminiator on fake images
            noise_tensor = Variable(torch.randn(batch_size, 100, 1, 1))
            fake_img = net_generator.forward(noise_tensor)
            zeros = Variable(torch.zeros(batch_size))
            # detach the gradient from generator (saves computation)
            output = net_discriminator.forward(fake_img.detach())
            fake_img_error = criterion(output, zeros)

            ## backpropagate total error
            descriminator_error = real_img_error + fake_img_error
            descriminator_error.backward()
            optimizer_discriminator.step()

            ## 2. update weights of generator
            net_generator.zero_grad()
            # now we keep the gradient so we can update the weights of the generator
            output = net_discriminator.forward(fake_img)
            generator_error = criterion(output, ones)
            generator_error.backward()
            optimizer_generator.step()

            ## show loss
            logger.debug(
                'Epoch: {}/{}, Step: {}/{}'.format(int(epoch), EPOCH_SIZE, idx, max_data))
            logger.info('Loss Descriminator: {:.4f} | Loss Generator: {:.4f}'.format(
                descriminator_error.data[0], generator_error.data[0]))

            if idx != 0 and idx % 100 == 0:
                save_images(data[0], 'real_samples', 0)
                save_images(fake_img.data, 'fake_samples', int(epoch))

        save_model({
            'epoch': epoch + 1,
            'state_dict': net_generator.state_dict(),
            'optimizer': optimizer_generator.state_dict()
        }, 'generator', int(epoch + 1))
        save_model({
            'epoch': epoch + 1,
            'state_dict': net_discriminator.state_dict(),
            'optimizer': optimizer_discriminator.state_dict()
        }, 'discriminator', int(epoch + 1))


if __name__ == '__main__':
    load_logging_config()
    DATALOADER = load_dataset()
    ARGS = args_parser()
    train(ARGS, DATALOADER)
