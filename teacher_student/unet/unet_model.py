from .unet_parts import *

""" Full assembly of the parts to form the complete network """
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 32))
        self.down1 = (Down(32, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        factor = 2 if bilinear else 1
        self.down4 = (Down(256, 512 // factor))
        
        self.depth_up1 = (Up(512, 256 // factor, bilinear))
        self.depth_up2 = (Up(256, 128 // factor, bilinear))
        self.depth_up3 = (Up(128, 64 // factor, bilinear))
        self.depth_up4 = (Up(64, 32, bilinear))
        self.depth_outc = (OutConv(32, 1))
        
        self.seg_up1 = (Up(512, 256 // factor, bilinear))
        self.seg_up2 = (Up(256, 128 // factor, bilinear))
        self.seg_up3 = (Up(128, 64 // factor, bilinear))
        self.seg_up4 = (Up(64, 32, bilinear))
        self.seg_outc = (OutConv(32, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x_d = self.depth_up1(x5, x4)
        x_d = self.depth_up2(x_d, x3)
        x_d = self.depth_up3(x_d, x2)
        x_d = self.depth_up4(x_d, x1)
        logits_d = self.depth_outc(x_d)
        
        x = self.seg_up1(x5, x4)
        x = self.seg_up2(x, x3)
        x = self.seg_up3(x, x2)
        x = self.seg_up4(x, x1)
        logits = self.seg_outc(x)
        
        
        return logits, logits_d