import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        s = self.conv(x)
        p = self.pool(s)
        return s, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = conv_block(out_c + out_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        # Make sure the spatial size matches due to rounding
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, bias=False)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, bias=False)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, g, x):
        g1 = self.W_g(g)
        # Upsample g1 to match x spatial size
        g1 = F.interpolate(g1, size=x.shape[2:], mode='bilinear', align_corners=True)
        x1 = self.W_x(x)
        psi = self.sigmoid(self.psi(self.relu(g1 + x1)))
        return x * psi


class build_unet(nn.Module):
    def __init__(self, num_classes=5):  # your AV segmentation has 5 classes
        super().__init__()

        # Encoder
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        # Bottleneck
        self.b = conv_block(512, 1024)

        # Attention blocks
        self.att4 = AttentionBlock(F_g=1024, F_l=512, F_int=256)
        self.att3 = AttentionBlock(F_g=512, F_l=256, F_int=128)
        self.att2 = AttentionBlock(F_g=256, F_l=128, F_int=64)
        self.att1 = AttentionBlock(F_g=128, F_l=64, F_int=32)

        # Decoder
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        # Classifier
        self.outputs = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        # Bottleneck
        b = self.b(p4)

        # Decoder with attention
        s4_att = self.att4(b, s4)
        d1 = self.d1(b, s4_att)

        s3_att = self.att3(d1, s3)
        d2 = self.d2(d1, s3_att)

        s2_att = self.att2(d2, s2)
        d3 = self.d3(d2, s2_att)

        s1_att = self.att1(d3, s1)
        d4 = self.d4(d3, s1_att)

        # Classifier
        out = self.outputs(d4)
        return out


if __name__ == "__main__":
    model = build_unet()
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, input_res=(3, 256, 256), as_strings=True, print_per_layer_stat=False)
    print('      - Flops:  ' + flops)
    print('      - Params: ' + params)

