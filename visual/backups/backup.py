        
import torch.nn as nn
import torchvision.models as models

import sys

class deconv_resnet(nn.Module):

    def __init__(self):
        super(deconv_resnet, self).__init__()

        self.features = nn.Sequential(
            ### decon layer 1  <-> con layer4 ###
            # block0 #
            nn.ConvTranspose2d(512, 512, 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # block1 #
            nn.ConvTranspose2d(512, 512, 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # block2 #
            nn.ConvTranspose2d(512, 512, 3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, 3,stride = 2, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            ### decon layer 2  <-> con layer3 ###
            # block0 #
            nn.ConvTranspose2d(512, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # block1 #
            nn.ConvTranspose2d(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # block2 #
            nn.ConvTranspose2d(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # block3 #
            nn.ConvTranspose2d(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # block4 #
            nn.ConvTranspose2d(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # block5 #
            nn.ConvTranspose2d(256, 256, 3, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, 3,stride = 2, padding = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ### decon layer 3  <-> con layer2 ###
            # block0 #
            nn.ConvTranspose2d(256, 128, 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # block1 #
            nn.ConvTranspose2d(128, 128, 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # block2 #
            nn.ConvTranspose2d(128, 128, 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # block3 #
            nn.ConvTranspose2d(128, 128, 3, padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 3, stride = 2,padding = 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ### decon layer 4  <-> con layer1 ###
            # block0 #
            nn.ConvTranspose2d(128, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # block1 #
            nn.ConvTranspose2d(64, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # block2 #
            nn.ConvTranspose2d(64, 64, 3, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, 3, stride = 2, padding = 1),
            nn.BatchNorm2d(64),

            ###### bottom ######
            nn.MaxUnpool2d(2, stride = 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, 7, stride = 2,padding = 1),
            
        )

        
    def forward(self, x):

        x = self.features(x)

        return x       

model = deconv_resnet()