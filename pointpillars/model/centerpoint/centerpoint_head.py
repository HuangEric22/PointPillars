import torch
import torch.nn as nn

class CenterHead(nn.Module):
    """
    Minimal CenterPoint head: predicts center heatmap + box attributes on BEV feature map.
    """
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.hm = nn.Conv2d(in_channels, num_classes, 3, padding=1)
        self.off = nn.Conv2d(in_channels, 2, 3, padding=1)
        self.z = nn.Conv2d(in_channels, 1, 3, padding=1)
        self.dims = nn.Conv2d(in_channels, 3, 3, padding=1)  # w,l,h
        self.rot = nn.Conv2d(in_channels, 2, 3, padding=1)   # sin,cos

        # CenterNet/CenterPoint trick: initialize hm bias low so early training predicts low probs
        nn.init.constant_(self.hm.bias, -2.19)

    def forward(self, x):
        return {
            "hm": torch.sigmoid(self.hm(x)),
            "off": self.off(x),
            "z": self.z(x),
            "dims": self.dims(x),
            "rot": self.rot(x),
        }
