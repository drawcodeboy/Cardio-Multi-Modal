import torch
from torch import nn

from .Resnet1d_wang import ResNet1d_Wang, BasicBlock1d
from .generate_cls_token import generate_cls_token
from .MCT_VIT_Block import MCT_ViT_Block

class MCT_Model(nn.Module):
    def __init__(self):
        super(MCT_Model, self).__init__()
        ## 여기에 ResNet1d, VIT 등등 모델 다 작성하기

        self.ecg_feature_extractor = ResNet1d_Wang(BasicBlock1d, layers=[1, 1, 1], input_channels=1, token_dim=256) # input: (input_channels, sequence_length)
        self.pcg_feature_extractor = ResNet1d_Wang(BasicBlock1d, layers=[1, 1, 1], input_channels=1, token_dim=256)
        self.linear = nn.Linear(256, 256)
        # self.viT_block = MCT_ViT_Block()
        # self.fusion_block = FusionBlock()
        # self.classifier = nn.Linear(in,ou 작성)  

    def forward(self, ecg, pcg): ## pcg, ecg (batch_size, 5000, 1) 
        # print(f'ecg_shape: {ecg.shape}, pcg_shape: {ecg.shape}')

        ecg = ecg.permute(0, 2, 1)  # 차원 변경
        pcg = pcg.permute(0, 2, 1) 

        ecg_tokens = self.ecg_feature_extractor(ecg) # (batch_size, 1, 5000) -> (batch_size, 256, 313) -> batch, token_dim, sequence_length
        pcg_tokens = self.pcg_feature_extractor(pcg) # (batch_size, 1, 5000) -> (batch_size, 256, 313)
        # print(f'ecg_tokens_shape: {ecg_tokens.shape}, pcg_tokens_shape: {pcg_tokens.shape}')

        cls_token = generate_cls_token(ecg_tokens, pcg_tokens)

        ## MCT_VIT 작성하기
        # cls_token_shape: torch.Size([128, 256, 1])
        # ecg_tokens_shape: torch.Size([128, 313, 256]), pcg_tokens_shape: torch.Size([128, 313, 256])
        ecg_feature_output1, pcg_feature_output1, cls_token_output1 = MCT_ViT_Block(ecg_tokens, pcg_tokens, cls_token)




        while(1):
            a = 1

        
        
        
        return 0