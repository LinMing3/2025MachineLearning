# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

def check_clip():
    """检查CLIP是否可用"""
    try:
        import clip
        # 测试基本功能
        clip.available_models()
        return True, clip
    except ImportError:
        print("Error: CLIP not installed. Run: pip install git+https://github.com/openai/CLIP.git")
        return False, None
    except Exception as e:
        print(f"Error: CLIP installation broken: {e}")
        return False, None

# 检查CLIP可用性
CLIP_OK, clip = check_clip()

class CLIPViTB32(nn.Module):
    """CLIP ViT-B/32 backbone"""
    
    def __init__(self, 
                 model_name="ViT-B/32", 
                 freeze_backbone=False,
                 output_dim=512,
                 **kwargs):
        super().__init__()
        
        if not CLIP_OK:
            raise RuntimeError("CLIP not available")
        
        self.model_name = model_name
        self.freeze_backbone = freeze_backbone
        self.output_dim = output_dim
        
        # 加载CLIP模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load(model_name, device=device)
        
        self.clip_model =self.clip_model.float()  # 确保模型是float类型

        # 只使用视觉编码器
        self.visual_encoder = self.clip_model.visual
        
        # 冻结backbone
        if freeze_backbone:
            for param in self.visual_encoder.parameters():
                param.requires_grad = False
        
        # 获取输出维度
        clip_dim = getattr(self.visual_encoder, 'output_dim', 512)
        
        # 特征投影层
        if output_dim != clip_dim:
            self.feature_projection = nn.Linear(clip_dim, output_dim)
        else:
            self.feature_projection = nn.Identity()
        
    def forward(self, x):
        """前向传播"""
        if x.dtype != torch.float32:
            x = x.float()
        
        visual_features = self.visual_encoder(x)
        output = self.feature_projection(visual_features)
        
        return output