import itertools

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lit_models.utils import ReplayBuffer, LambdaLR, weights_init_normal

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

class LitModel(pl.LightningModule):
    def __init__(
            self,
            epoch:       int   = 0,
            n_epochs:    int   = 200,
            decay_epoch: int   = 100,
            batch_size:  int   = 4,
            lr:          float = 0.0002,
            size:        int   = 256,
            input_nc:    int   = 3,
            output_nc:   int   = 3,
            *args, **kwargs
        ):
        super().__init__()
        self.save_hyperparameters()

        self.epoch       = epoch
        self.n_epochs    = n_epochs
        self.decay_epoch = decay_epoch
        self.batch_size  = batch_size
        self.lr          = lr
        self.size        = size
        self.input_nc    = input_nc
        self.output_nc   = output_nc

        # Networks
        self.netG_A2B = Generator(self.input_nc, self.output_nc).apply(weights_init_normal)
        self.netG_B2A = Generator(self.output_nc, self.input_nc).apply(weights_init_normal)
        self.netD_A   = Discriminator(self.input_nc).apply(weights_init_normal)
        self.netD_B   = Discriminator(self.output_nc).apply(weights_init_normal)

        # ReplayBuffer
        self.fake_A_buffer = ReplayBuffer()
        self.fake_B_buffer = ReplayBuffer()

    def forward(self, x):
        return 

    def training_step(self, batch, batch_idx, optimizer_idx):
        real_A = batch['A'].half() if self.half else batch['A']
        real_B = batch['B'].half() if self.half else batch['B']

        target_real = torch.ones(
                size=(len(real_A), 1),
                dtype=torch.float16,
                device=torch.device('cuda'),
                requires_grad=False,
        )
        target_fake = torch.zeros(
                size=(len(real_B), 1),
                dtype=torch.float16,
                device=torch.device('cuda'),
                requires_grad=False,
        )

        ###### Generators A2B and B2A ######
        if optimizer_idx == 0:
            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = self.netG_A2B(real_B)
            loss_identity_B = F.l1_loss(same_B, real_B) * 5.0
            # G_B2A(A) should equal A if real A is fed
            same_A = self.netG_B2A(real_A)
            loss_identity_A = F.l1_loss(same_A, real_A) * 5.0

            # GAN loss
            fake_B = self.netG_A2B(real_A)
            pred_fake = self.netD_B(fake_B)
            loss_GAN_A2B = F.mse_loss(pred_fake, target_real)

            fake_A = self.netG_B2A(real_B)
            pred_fake = self.netD_A(fake_A)
            loss_GAN_B2A = F.mse_loss(pred_fake, target_real)

            # Cycle loss
            recovered_A = self.netG_B2A(fake_B)
            loss_cycle_ABA = F.l1_loss(recovered_A, real_A) * 10.0

            recovered_B = self.netG_A2B(fake_A)
            loss_cycle_BAB = F.l1_loss(recovered_B, real_B) * 10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

            # Log
            self.log('loss_G', loss_G, prog_bar=True)
            self.log('loss_identity_A', loss_identity_A, prog_bar=True)
            self.log('loss_identity_B', loss_identity_B, prog_bar=True)
            self.log('loss_GAN_A2B', loss_GAN_A2B, prog_bar=True)
            self.log('loss_GAN_B2A', loss_GAN_B2A, prog_bar=True)
            return loss_G

        ######### Discriminator A #########
        if optimizer_idx == 1:
            fake_A = self.netG_B2A(real_B)

            # Real loss
            pred_real = self.netD_A(real_A)
            loss_D_real = F.mse_loss(pred_real, target_real)

            # Fake loss
            fake_A = self.fake_A_buffer.push_and_pop(fake_A)
            pred_fake = self.netD_A(fake_A.detach())
            loss_D_fake = F.mse_loss(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5

            # Log
            self.log("loss_D_A", loss_D_A, prog_bar=True)
            return loss_D_A

        ######### Discriminator B #########
        if optimizer_idx == 2:
            fake_B = self.netG_A2B(real_A)

            # Real loss
            pred_real = self.netD_B(real_B)
            loss_D_real = F.mse_loss(pred_real, target_real)
            
            # Fake loss
            fake_B = self.fake_B_buffer.push_and_pop(fake_B)
            pred_fake = self.netD_B(fake_B.detach())
            loss_D_fake = F.mse_loss(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5

            # Log
            self.log("loss_D_B", loss_D_B, prog_bar=True)
            return loss_D_B

    def validation_step(self, batch, batch_size):
        real_A = batch['A'].half()
        real_B = batch['B'].half()

        fake_A = self.netG_B2A(real_B)
        fake_B = self.netG_A2B(real_A)

        return {'fake_A': fake_A, 'fake_B': fake_B}

    def configure_optimizers(self):
# Optimizers & LR schedulers
        optimizer_G = torch.optim.Adam(
                itertools.chain(self.netG_A2B.parameters(), self.netG_B2A.parameters()),
                lr=self.lr, betas=(0.5, 0.999)
        )
        optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=self.lr, betas=(0.5, 0.999))
        optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=self.lr, betas=(0.5, 0.999))

        lr_scheduler_G   = torch.optim.lr_scheduler.LambdaLR(
                optimizer_G,   lr_lambda=LambdaLR(self.n_epochs, self.epoch, self.decay_epoch).step
        )
        lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
                optimizer_D_A, lr_lambda=LambdaLR(self.n_epochs, self.epoch, self.decay_epoch).step
        )
        lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
                optimizer_D_B, lr_lambda=LambdaLR(self.n_epochs, self.epoch, self.decay_epoch).step
        )

        return [optimizer_G, optimizer_D_A, optimizer_D_B], [lr_scheduler_G, lr_scheduler_D_A, lr_scheduler_D_B]
