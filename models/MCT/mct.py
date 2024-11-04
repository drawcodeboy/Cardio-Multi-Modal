import torch
from torch import nn

from .Resnet1d_wang import ResNet1d_Wang, BasicBlock1d
from .generate_cls_token import generate_cls_token
from .MCT_VIT_Block import MCT_VIT_Block
from .Fusion_Block import FusionBlock
from .classifer import Classifier

class MCT_Model(nn.Module):
    def __init__(self):
        super(MCT_Model, self).__init__()
        # 1. Resnet1d_wang 
        self.ecg_feature_extractor = ResNet1d_Wang(BasicBlock1d, layers=[1, 1, 1], input_channels=1, token_dim=32) # input: (batch_size, input_channels, input_length)
        self.pcg_feature_extractor = ResNet1d_Wang(BasicBlock1d, layers=[1, 1, 1], input_channels=1, token_dim=32) # output: (batch_size, token_dim, output_length)

        # 2. cls token creating
        self.generate_cls_token = generate_cls_token # input: ecg and pcg (batch_size, token_dim, sequence_length)  # output: (batch_size, token_dim, 1)

        # 3. MCT_VIT_Block
        self.ecg_vit_block_1 = MCT_VIT_Block() # input: ecg or pcg (batch_size, token_dim, sequence_length) and cls(batch_size, token_dim, 1)
        self.pcg_vit_block_1 = MCT_VIT_Block() # output: (batch_size, token_dim, sequence_length)

        self.ecg_vit_block_2 = MCT_VIT_Block() 
        self.pcg_vit_block_2 = MCT_VIT_Block() 

        self.ecg_vit_block_3 = MCT_VIT_Block() 
        self.pcg_vit_block_3 = MCT_VIT_Block() 

        # 4. FusionBlock
        self.fusion_block_1 = FusionBlock()
        self.fusion_block_2 = FusionBlock()
        self.fusion_block_3 = FusionBlock()
        
        # 5. Classifier
        self.classifier = Classifier(input_dim=32 * 3, hidden_dim=64, output_dim=7)

    def forward(self, ecg, pcg): 
        # pcg, ecg (batch_size, 5000, 1) 

        # ResNet1d_wang 으로 토큰 추출
        ecg_tokens = self.ecg_feature_extractor(ecg.permute(0, 2, 1)) # (batch_size, 32, 313) (batch, token_dim, sequence_length)
        pcg_tokens = self.pcg_feature_extractor(pcg.permute(0, 2, 1)) 

        # cls_token 생성
        cls_token_0 = self.generate_cls_token(ecg_tokens, pcg_tokens) # (batch_size, 32, 1) (batch_size, token_dim, 1)

        # MCT_VIT 
        '''
        # input
        cls_token_shape: torch.Size([128, 32, 1]) 
        ecg_tokens_shape: torch.Size([128, 32, 313]), pcg_tokens_shape: torch.Size([128, 32, 313]) = (batch_size, token_dim, sequence_length)      
        '''
        ecg_feature_output_1 = self.ecg_vit_block_1(ecg_tokens, cls_token_0) # (batch_size, token_dim, sequence_length)
        pcg_feature_output_1 = self.pcg_vit_block_1(pcg_tokens, cls_token_0) 


        # FusionBlock 들어가는 것
        # cls, ecg_feature_output, pcg_feature_output 퓨전
        '''
        # input
        cls_token_shape: (batch_size, 32, 1) (batch_size, token_dim, 1)
        ecg_feature_output_shape: (batch_size, token_dim, sequence_length) (128, 32, 313)
        '''

        fused_cls_token_1 = self.fusion_block_1(cls_token_0, ecg_feature_output_1, pcg_feature_output_1) # (batch_size, dim, 1)

        # 두번째 레이어
        ecg_feature_output_2 = self.ecg_vit_block_2(ecg_feature_output_1, fused_cls_token_1) # (batch_size, token_dim, sequence_length)
        pcg_feature_output_2 = self.pcg_vit_block_2(pcg_feature_output_1, fused_cls_token_1) 

        fused_cls_token_2 = self.fusion_block_2(fused_cls_token_1, ecg_feature_output_2, pcg_feature_output_2) # (batch_size, dim, 1)

        # 세번째 레이어
        ecg_feature_output_3 = self.ecg_vit_block_3(ecg_feature_output_2, fused_cls_token_2) # (batch_size, token_dim, sequence_length)
        pcg_feature_output_3 = self.pcg_vit_block_3(pcg_feature_output_2, fused_cls_token_2) 

        fused_cls_token_3 = self.fusion_block_3(fused_cls_token_2, ecg_feature_output_3, pcg_feature_output_3) # (batch_size, dim, 1)


        final_feature = torch.cat((fused_cls_token_1, fused_cls_token_2, fused_cls_token_3), dim=1)  # (batch_size, 3 * dim, 1)
        final_feature = final_feature.squeeze(-1)  # (batch_size, token_dim * 3)

        output = self.classifier(final_feature)

        return output