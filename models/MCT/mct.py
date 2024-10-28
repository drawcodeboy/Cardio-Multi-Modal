import torch
from torch import nn

from .Resnet1d_wang import ResNet1d_Wang, BasicBlock1d


class MCT_Model(nn.Module):
    def __init__(self):
        super(MCT_Model, self).__init__()
        ## 여기에 ResNet1d, VIT 등등 모델 다 작성하기

        self.ecg_feature_extractor = ResNet1d_Wang(BasicBlock1d, layers=[1, 1, 1], input_channels=1, token_dim=256)
        self.pcg_feature_extractor = ResNet1d_Wang(BasicBlock1d, layers=[1, 1, 1], input_channels=1, token_dim=256)
        # self.viT_block = MCT_ViT_Block()
        # self.fusion_block = FusionBlock()
        # self.classifier = nn.Linear(in,ou 작성)  

    def forward(self, ecg, pcg): ## pcg, ecg (batch_size, 5000, 1) 
        print(f'ecg_shape: {ecg.shape}, pcg_shape: {ecg.shape}')

        ecg = ecg.permute(0, 2, 1)  # 차원 변경
        pcg = pcg.permute(0, 2, 1) 

        ecg_tokens = self.ecg_feature_extractor(ecg) # (batch_size, 5000, 1) -> (batch_size, 256, 313) batch, token_dim, sequence_length
        pcg_tokens = self.pcg_feature_extractor(pcg) # (batch_size, 5000, 1) -> (batch_size, 256, 313)
        print(f'ecg_tokens_shape: {ecg_tokens.shape}, pcg_tokens_shape: {pcg_tokens.shape}')

        ecg_tokens = ecg_tokens.permute(0, 2, 1)  # 차원 변경
        pcg_tokens = pcg_tokens.permute(0, 2, 1)  # (batch_size, 313, 256) batch, sequence_length, token_dim 

        print(f'ecg_tokens_shape: {ecg_tokens.shape}, pcg_tokens_shape: {pcg_tokens.shape}')


        while(1):
            a = 1

        
        
        
        return 0