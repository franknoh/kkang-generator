import torch
import torch.nn as nn


class CFA(nn.Module):
    def __init__(self, output_channel_num, checkpoint_name=None):
        super(CFA, self).__init__()

        self.output_channel_num = output_channel_num
        self.stage_channel_num = 128
        self.stage_num = 2

        self.features = nn.Sequential(
            nn.Conv2d(  3,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d( 64,  64, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d( 64, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, dilation=1, padding=1), nn.ReLU(inplace=True)
        )
        
        self.CFM_features = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, self.stage_channel_num, kernel_size=3, padding=1), nn.ReLU(inplace=True)
        )

        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.stage_channel_num, self.stage_channel_num, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.stage_channel_num, self.stage_channel_num, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.stage_channel_num, self.stage_channel_num, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.stage_channel_num, self.stage_channel_num, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.stage_channel_num, self.stage_channel_num, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.stage_channel_num, self.output_channel_num, kernel_size=3, padding=1)
            )
        ] + [
            nn.Sequential(
                nn.Conv2d(self.stage_channel_num + self.output_channel_num, self.stage_channel_num, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.stage_channel_num, self.stage_channel_num, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.stage_channel_num, self.stage_channel_num, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.stage_channel_num, self.stage_channel_num, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.stage_channel_num, self.stage_channel_num, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.stage_channel_num, self.output_channel_num, kernel_size=3, padding=1)
            ) for _ in range(1, self.stage_num)
        ])
        
        weights = torch.load(checkpoint_name, map_location='cpu')
        self.load_state_dict(weights['state_dict'])
    

    def forward(self, x):
        feature = self.features(x)
        feature = self.CFM_features(feature)
        for i, stage in enumerate(self.stages):
            input_tensor = feature if i == 0 else torch.cat([feature, heatmap], 1)
            heatmap = stage(input_tensor)
        
        return heatmap
