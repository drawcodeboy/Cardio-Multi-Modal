import torch
from torch import nn

from .Resnet1d_wang import ResNet1d_Wang, BasicBlock1d
from .generate_cls_token import generate_cls_token
from .MCT_VIT_Block import MCT_VIT_Block

class MCT_Model(nn.Module):
    def __init__(self):
        super(MCT_Model, self).__init__()
        # 1. Resnet1d_wang 
        self.ecg_feature_extractor = ResNet1d_Wang(BasicBlock1d, layers=[1, 1, 1], input_channels=1, token_dim=32) # input: (batch_size, input_channels, input_length)
        self.pcg_feature_extractor = ResNet1d_Wang(BasicBlock1d, layers=[1, 1, 1], input_channels=1, token_dim=32) # output: (batch_size, token_dim, output_length)

        # 2. cls token creating
        self.generate_cls_token = generate_cls_token # input: ecg and pcg (batch_size, token_dim, sequence_length) * 2 # output: (batch_size, token_dim, 1)

        # 3. MCT_VIT_Block
        self.ecg_vit_block = MCT_VIT_Block() # input: ecg(batch_size, token_dim, sequence_length) and cls(batch_size, token_dim, 1)
        self.pcg_vit_block = MCT_VIT_Block() # output: (batch_size, token_dim, sequence_length) 

        # 4. FusionBlock
        # self.fusion_block = FusionBlock()
        
        # 5. Classifier
        # self.classifier = nn.Linear(in,ou 작성)  

    def forward(self, ecg, pcg): 
        ## pcg, ecg (batch_size, 5000, 1) 

        # 차원 변경
        ecg = ecg.permute(0, 2, 1)  # (batch_size, 1, 5000) 
        pcg = pcg.permute(0, 2, 1)

        # ResNet1d_wang 으로 토큰 추출
        ecg_tokens = self.ecg_feature_extractor(ecg) # (batch_size, 32, 313) (batch, token_dim, sequence_length)
        pcg_tokens = self.pcg_feature_extractor(pcg) 

        # cls_token 생성
        cls_token = self.generate_cls_token(ecg_tokens, pcg_tokens) # (batch_size, 32, 1) (batch_size, token_dim, 1)

        # MCT_VIT 
        '''
        # input
        cls_token_shape: torch.Size([128, 32, 1]) 
        ecg_tokens_shape: torch.Size([128, 32, 313]), pcg_tokens_shape: torch.Size([128, 32, 313]) = (batch_size, token_dim, sequence_length)      
        '''
        ecg_feature_output = self.ecg_vit_block(ecg_tokens, cls_token) # (batch_size, token_dim, sequence_length)
        pcg_feature_output = self.pcg_vit_block(pcg_tokens, cls_token) 


        # FusionBlock 들어가는 것
        # cls, ecg_feature_output, pcg_feature_output 퓨전


        while(1):
            a = 1
        
        
        
        return 0