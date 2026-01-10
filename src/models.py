import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    """
    Implementacja wzoru (5) z artykułu: Hybrid Loss (Cross Entropy + Dice Loss).
    DCL(Q, F) = CrossEntropy + (1 - DiceScore)
    """
    def __init__(self, smooth=1e-6):
        super(HybridLoss, self).__init__()
        self.smooth = smooth
        # Używamy standardowej CrossEntropy, która w PyTorch jest bardzo stabilna
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        """
        logits: Wyjście z modelu (surowe, bez softmax) [Batch, Classes, H, W]
        targets: Prawdziwe maski (indeksy klas) [Batch, H, W]
        """
        
        # 1. Obliczenie Cross Entropy Loss
        # PyTorchowa funkcja CrossEntropyLoss oczekuje logits i targets jako long
        ce_loss = self.ce(logits, targets)
        
        # 2. Obliczenie Dice Loss (zgodnie z drugą częścią wzoru)
        # Musimy zamienić logits na prawdopodobieństwa (Softmax)
        probs = torch.softmax(logits, dim=1)
        
        # Musimy zamienić targets na One-Hot Encoding, żeby pasowały kształtem do probs
        # targets: [B, H, W] -> [B, C, H, W]
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Wzór z obrazka używa kwadratów w mianowniku: (y^2 + p^2)
        # Intersection (licznik): 2 * y * p
        intersection = 2.0 * (probs * targets_one_hot).sum(dim=(2, 3))
        
        # Union (mianownik): y^2 + p^2
        # Dla targets (0 lub 1) kwadrat nic nie zmienia, ale dla probs tak.
        denominator = (probs ** 2).sum(dim=(2, 3)) + (targets_one_hot ** 2).sum(dim=(2, 3))
        
        # Dice Score dla każdej klasy w każdym obrazie
        dice_score = (intersection + self.smooth) / (denominator + self.smooth)
        
        # Dice Loss to 1 - Dice Score (bo chcemy minimalizować stratę)
        # Uśredniamy po klasach i po batchu
        dice_loss = 1.0 - dice_score.mean()
        
        # Wynikowa strata to suma obu (wzór ma minus przed całą sumą, co w optymalizacji
        # oznacza minimalizację sumy strat składowych).
        return ce_loss + dice_loss

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ImprovedAttentionUNet(nn.Module):
    def __init__(self, img_ch=1, output_ch=4): # Zmieniono domyślnie na 4 klasy
        super(ImprovedAttentionUNet, self).__init__()
        
        filters = [64, 128, 256, 512, 1024]
        
        # Encoder
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.res1 = ResidualBlock(img_ch, filters[0])
        self.res2 = ResidualBlock(filters[0], filters[1])
        self.res3 = ResidualBlock(filters[1], filters[2])
        self.res4 = ResidualBlock(filters[2], filters[3])
        self.res5 = ResidualBlock(filters[3], filters[4])
        
        # Decoder
        self.up5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att5 = AttentionBlock(F_g=filters[4], F_l=filters[3], F_int=filters[3] // 2)
        self.up_res5 = ResidualBlock(filters[4] + filters[3], filters[3])
        
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att4 = AttentionBlock(F_g=filters[3], F_l=filters[2], F_int=filters[2] // 2)
        self.up_res4 = ResidualBlock(filters[3] + filters[2], filters[2])
        
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att3 = AttentionBlock(F_g=filters[2], F_l=filters[1], F_int=filters[1] // 2)
        self.up_res3 = ResidualBlock(filters[2] + filters[1], filters[1])
        
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.att2 = AttentionBlock(F_g=filters[1], F_l=filters[0], F_int=filters[0] // 2)
        self.up_res2 = ResidualBlock(filters[1] + filters[0], filters[0])
        
        self.conv_final = nn.Conv2d(filters[0], output_ch, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        x1 = self.res1(x)
        x2 = self.maxpool(x1); x2 = self.res2(x2)
        x3 = self.maxpool(x2); x3 = self.res3(x3)
        x4 = self.maxpool(x3); x4 = self.res4(x4)
        x5 = self.maxpool(x4); x5 = self.res5(x5)
        
        # Decoder
        d5 = self.up5(x5)
        x4 = self.att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.up_res5(d5)
        
        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.up_res4(d4)
        
        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_res3(d3)
        
        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_res2(d2)

        output = self.conv_final(d2)
        return output