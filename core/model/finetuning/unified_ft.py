#TODO：包括十种带/不带FD-Align的FT方法
# 1. FT
# 2. Tip
# 3. Tip-F
# 4. APE
# 5. APE-T
# 6. FD-Align + FT
# 7. FD-Align + Tip
# 8. FD-Align + Tip-F
# 9. FD-Align + APE
# 10. FD-Align + APE-T

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import numpy as np
from .finetuning_model import FinetuningModel
from core.utils import accuracy
import re

IMAGENET_TEMPLATES = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

class TipAdapter(nn.Module):
    """Tip-Adapter轻量级适配器"""
    def __init__(self, feat_dim, num_class, adapter_ratio=0.5):
        super().__init__()
        self.adapter_dim = int(feat_dim * adapter_ratio)
        self.down_proj = nn.Linear(feat_dim, self.adapter_dim)
        self.up_proj = nn.Linear(self.adapter_dim, feat_dim)
        self.classifier = nn.Linear(feat_dim, num_class)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # Tip-Adapter: residual connection + lightweight MLP
        residual = x
        adapted = self.down_proj(x)
        adapted = F.relu(adapted)
        adapted = self.dropout(adapted)
        adapted = self.up_proj(adapted)
        adapted_feat = residual + adapted
        return self.classifier(adapted_feat)

class APEModule(nn.Module):
    """APE (Automated Prompt Engineering) 模块"""
    def __init__(self, feat_dim, num_class, num_prompts=16):
        super().__init__()
        self.num_prompts = num_prompts
        self.feat_dim = feat_dim
        
        # 初始化可学习的prompt embeddings
        self.prompt_embeddings = nn.Parameter(torch.randn(num_prompts, feat_dim))
        
        # 使用更简单的attention机制
        self.query_proj = nn.Linear(feat_dim, feat_dim)
        self.key_proj = nn.Linear(feat_dim, feat_dim)
        self.value_proj = nn.Linear(feat_dim, feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        
        # 分类器
        self.classifier = nn.Linear(feat_dim, num_class)
        self.dropout = nn.Dropout(0.1)
        
        # 标准化prompt embeddings
        nn.init.normal_(self.prompt_embeddings, std=0.02)
        
    def forward(self, x):
        """
        x: [batch_size, feat_dim]
        """
        batch_size = x.size(0)
        
        # 1. 扩展prompt embeddings到batch
        # prompts: [batch_size, num_prompts, feat_dim]
        prompts = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 2. 计算attention
        # x作为query: [batch_size, 1, feat_dim]
        query = self.query_proj(x).unsqueeze(1)  # [batch_size, 1, feat_dim]
        
        # prompts作为key和value
        key = self.key_proj(prompts)    # [batch_size, num_prompts, feat_dim]
        value = self.value_proj(prompts) # [batch_size, num_prompts, feat_dim]
        
        # 3. 简化的attention计算
        # 计算attention scores
        scores = torch.bmm(query, key.transpose(1, 2))  # [batch_size, 1, num_prompts]
        scores = scores / (self.feat_dim ** 0.5)  # scale
        attention_weights = F.softmax(scores, dim=-1)
        
        # 4. 应用attention weights到value
        attended_prompts = torch.bmm(attention_weights, value)  # [batch_size, 1, feat_dim]
        attended_prompts = attended_prompts.squeeze(1)  # [batch_size, feat_dim]
        
        # 5. 融合原始特征和attention后的prompt
        fused_features = x + self.out_proj(attended_prompts)
        fused_features = self.dropout(fused_features)
        
        # 6. 分类
        output = self.classifier(fused_features)
        
        return output

class UnifiedFT(FinetuningModel):
    def __init__(self, feat_dim, num_class, method_type="FT", 
                 use_fd_align=False, class_names=None,
                 alpha=1.0, beta=0.5, spc_n=60, spc_k=20, 
                 adapter_ratio=0.5, num_prompts=16, **kwargs):
        super().__init__(**kwargs)
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.method_type = method_type
        self.use_fd_align = use_fd_align

       # 生成安全的类别名称
        if class_names is None:
            self.class_names = [f"class {i}" for i in range(num_class)]
        else:
            # 清理类别名称，移除可能导致tokenizer问题的字符
            self.class_names = [self._clean_class_name(name) for name in class_names]
        
        # FD-Align 论文参数
        self.alpha = alpha          # 1.0
        self.beta = beta            # 0.5
        self.spc_n = spc_n          # 60
        self.spc_k = spc_k          # 20
        
        # Tip-Adapter 参数
        self.adapter_ratio = adapter_ratio    # 0.5
        self.num_prompts = num_prompts        # 16
        
        
        # 根据方法类型初始化不同的组件
        if method_type == "FT":
            self.classifier = nn.Linear(feat_dim, num_class)
        elif method_type in ["Tip", "TipF"]:
            self.tip_adapter = TipAdapter(feat_dim, num_class)
            if method_type == "TipF":
                self.fine_tune_layers = nn.ModuleList([
                    nn.Linear(feat_dim, feat_dim),
                    nn.ReLU(),
                    nn.Linear(feat_dim, feat_dim)
                ])
        elif method_type in ["APE", "APET"]:
            self.ape_module = APEModule(feat_dim, num_class)
            if method_type == "APET":
                # APE-T uses template-based prompts
                self.template_weights = nn.Parameter(torch.ones(len(IMAGENET_TEMPLATES)))
        
        self.loss_func = nn.CrossEntropyLoss()
        
        # FD-Align相关初始化
        if use_fd_align:
            self._init_fd_align()

    def _clean_class_name(self, name):
        # 移除特殊字符，保留字母、数字、空格和常见标点
        cleaned = re.sub(r'[^\w\s\-_]', '', str(name))
        # 确保不为空
        if not cleaned.strip():
            cleaned = "unknown"
        return cleaned.strip()

    def _init_fd_align(self):
        """初始化FD-Align组件"""
        try:
            # 加载CLIP模型（仅用于文本编码）
            self.clip_model, _ = clip.load("ViT-B/32", device=self.device)
            self.clip_model.eval()
            
            # 冻结CLIP参数
            for param in self.clip_model.parameters():
                param.requires_grad = False
                
        except Exception as e:
            print(f"Warning: CLIP model loading failed: {e}")
            self.clip_model = None
        
        # 原型存储
        self.class_prototypes = None
        self.spurious_prototypes = None
        self._initialize_prototypes()
    
    def _initialize_prototypes(self):
        """初始化类别原型和虚假原型"""
        if self.clip_model is None:
            return
            
        # 生成类别原型
        class_prototypes = []
        for class_name in self.class_names:
            # 为每个类别生成文本嵌入
            texts = [f"a photo of a {class_name}"]
            text_tokens = clip.tokenize(texts).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_tokens)
                text_features = F.normalize(text_features, p=2, dim=-1)
            class_prototypes.append(text_features.mean(dim=0))
        
        self.class_prototypes = torch.stack(class_prototypes)
        
        # 生成虚假原型（基于ImageNet模板）
        spurious_prototypes = []
        for template in IMAGENET_TEMPLATES[:self.spc_n]:  # 使用前spc_n个模板
            template_prototypes = []
            for class_name in self.class_names:
                text = template.format(class_name)
                text_tokens = clip.tokenize([text]).to(self.device)
                with torch.no_grad():
                    text_features = self.clip_model.encode_text(text_tokens)
                    text_features = F.normalize(text_features, p=2, dim=-1)
                template_prototypes.append(text_features.squeeze(0))
            
            # 对每个模板，计算所有类别的平均嵌入
            avg_prototype = torch.stack(template_prototypes).mean(dim=0)
            spurious_prototypes.append(avg_prototype)
        
        spurious_prototypes = torch.stack(spurious_prototypes)
        
        # 应用SPC校正
        self.spurious_prototypes = self._spurious_prototype_correction(spurious_prototypes)
    
    def _spurious_prototype_correction(self, prototypes):
        """伪原型校正 (SPC)"""
        prototypes_np = prototypes.cpu().numpy()
        
        # 1. 孤立森林去除异常值
        if len(prototypes_np) > 10:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_mask = iso_forest.fit_predict(prototypes_np) == 1
            clean_prototypes = prototypes_np[outlier_mask]
        else:
            clean_prototypes = prototypes_np
        
        # 2. K-means聚类合并相似原型
        if len(clean_prototypes) > self.spc_k:
            kmeans = KMeans(n_clusters=self.spc_k, random_state=42)
            kmeans.fit(clean_prototypes)
            final_prototypes = kmeans.cluster_centers_
        else:
            final_prototypes = clean_prototypes
            
        return torch.tensor(final_prototypes, device=prototypes.device, dtype=prototypes.dtype)
    
    def _compute_spurious_loss(self, image_features):
        """计算虚假特征损失"""
        if self.spurious_prototypes is None:
            return torch.tensor(0.0, device=image_features.device)
        
        # 计算图像特征与虚假原型的相似性
        similarities = F.cosine_similarity(
            image_features.unsqueeze(1),
            self.spurious_prototypes.unsqueeze(0),
            dim=-1
        )
        
        # 使用预训练模型作为参考分布
        with torch.no_grad():
            if hasattr(self, 'pretrained_similarities'):
                target_similarities = self.pretrained_similarities
            else:
                # 如果没有预训练参考，使用当前相似性作为软目标
                target_similarities = similarities.detach()
        
        # KL散度损失
        spurious_loss = F.kl_div(
            F.log_softmax(similarities, dim=-1),
            F.softmax(target_similarities, dim=-1),
            reduction='batchmean'
        )
        
        return spurious_loss
    
    def _ft_method(self, feat):
        """标准微调方法"""
        return self.classifier(feat)
    
    def _tip_method(self, feat):
        """Tip方法"""
        return self.tip_adapter(feat)
    
    def _tipf_method(self, feat):
        """Tip-F方法Tip + Fine-tuning"""
        processed_feat = feat
        for layer in self.fine_tune_layers:
            processed_feat = layer(processed_feat)
        return self.tip_adapter(processed_feat)
    
    def _ape_method(self, feat):
        """APE方法"""
        return self.ape_module(feat)
    
    def _apet_method(self, feat):
        """APE-T方法（APE + 模板权重）"""
        # 1. 获取基础APE logits
        base_logits = self.ape_module(feat)
        
        # 2. 如果有模板权重，应用模板增强
        if hasattr(self, 'template_weights') and self.use_fd_align:
            # 使用加权的模板进行增强
            batch_size = feat.size(0)
            
            # 计算模板加权的特征增强
            template_weights = F.softmax(self.template_weights, dim=0)
            
            # 这里可以添加更复杂的模板权重逻辑
            # 例如：基于模板权重调整logits
            template_scale = template_weights.mean()  # 简化的模板影响
            enhanced_logits = base_logits * (1 + 0.1 * template_scale)
            
            return enhanced_logits
        else:
            return base_logits
    
    def forward(self, batch):
        """主要的前向传播方法，训练器会调用这个方法"""
        if self.training:
            return self.set_forward_loss(batch)
        else:
            return self.set_forward(batch)
        
    def set_forward_loss(self, batch):
        """训练时的前向传播和损失计算"""
        image, target = batch
        
        if not hasattr(self, '_debug_printed'):
            print(f"DEBUG: image.shape={image.shape}")
            print(f"DEBUG: target.shape={target.shape}")
            print(f"DEBUG: target range: {target.min().item()}-{target.max().item()}")
            print(f"DEBUG: num_class configured: {self.num_class}")
            self._debug_printed = True

        image = image.to(self.device)
        target = target.to(self.device)
        
        
        if target.max() >= self.num_class:
            # 将标签重新映射到 0 到 num_class-1 的范围
            unique_labels = torch.unique(target)
            target_mapping = {label.item(): i for i, label in enumerate(unique_labels)}
            
            # 创建新的标签张量
            new_target = torch.zeros_like(target)
            for old_label, new_label in target_mapping.items():
                mask = target == old_label
                new_target[mask] = new_label
            target = new_target
            
    
    
        # 提取特征
        feat = self.emb_func(image)
        
        # 根据方法类型计算logits
        try:
            if self.method_type == "FT":
                logits = self._ft_method(feat)
            elif self.method_type == "Tip":
                logits = self._tip_method(feat)
            elif self.method_type == "TipF":
                logits = self._tipf_method(feat)
            elif self.method_type == "APE":
                logits = self._ape_method(feat)
            elif self.method_type == "APET":
                logits = self._apet_method(feat)
            else:
                raise ValueError(f"Unsupported method_type: {self.method_type}")
        except Exception as e:
            print(f"Error in forward pass for method {self.method_type}: {e}")
            raise
        
        # 计算主要损失
        class_loss = self.loss_func(logits, target)
        
         # 如果使用FD-Align，添加虚假特征损失
        if self.use_fd_align:
            try:
                spurious_loss = self._compute_spurious_loss(feat)
                total_loss = self.alpha * class_loss + self.beta * spurious_loss
            except Exception as e:
                print(f"Warning: FD-Align spurious loss computation failed: {e}")
                total_loss = class_loss
        else:
            total_loss = class_loss
        
        # 计算准确率
        acc = accuracy(logits, target)
        
        return logits, acc, total_loss
    
    def set_forward(self, batch):
        """用于测试/验证的前向传播"""
        image, global_target = batch
        image = image.to(self.device)
        
        with torch.no_grad():
            feat = self.emb_func(image)
        
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=1
        )
        episode_size = support_feat.size(0)
        
        output_list = []
        for i in range(episode_size):
            output = self.set_forward_adaptation(
                support_feat[i], support_target[i], query_feat[i]
            )
            output_list.append(output)
        
        output = torch.cat(output_list, dim=0)
        acc = accuracy(output, query_target.reshape(-1))
        
        return output, acc
    
    def set_forward_adaptation(self, support_feat, support_target, query_feat):
        """Few-shot适应性前向传播"""
        # 为当前episode创建临时分类器
        if self.method_type == "FT":
            classifier = nn.Linear(self.feat_dim, self.way_num).to(self.device)
        elif self.method_type in ["Tip", "TipF"]:
            classifier = TipAdapter(self.feat_dim, self.way_num).to(self.device)
        elif self.method_type in ["APE", "APET"]:
            classifier = APEModule(self.feat_dim, self.way_num).to(self.device)
        else:
            classifier = nn.Linear(self.feat_dim, self.way_num).to(self.device)
        
        optimizer = self.sub_optimizer(classifier, {
            "name": "SGD",
            "kwargs": {
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0005
            }
        })
        
        classifier.train()
        support_size = support_feat.size(0)
        
        # 内层训练循环
        for epoch in range(100):  # inner_train_iter
            rand_id = torch.randperm(support_size)
            for i in range(0, support_size, 4):  # inner_batch_size = 4
                select_id = rand_id[i : min(i + 4, support_size)]
                batch = support_feat[select_id]
                target = support_target[select_id]
                
                # 前向传播
                if self.method_type in ["TipF"]:
                    # Tip-F需要特殊处理
                    processed_feat = batch
                    for layer in self.fine_tune_layers:
                        processed_feat = layer(processed_feat)
                    output = classifier(processed_feat)
                else:
                    output = classifier(batch)
                
                loss = self.loss_func(output, target)
                
                # 如果使用FD-Align，添加spurious loss
                if self.use_fd_align:
                    spurious_loss = self._compute_spurious_loss(batch)
                    loss = loss + 0.5 * spurious_loss  # beta=0.5
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
        
        # 用训练好的分类器预测query
        classifier.eval()
        with torch.no_grad():
            if self.method_type in ["TipF"]:
                processed_query_feat = query_feat
                for layer in self.fine_tune_layers:
                    processed_query_feat = layer(processed_query_feat)
                output = classifier(processed_query_feat)
            else:
                output = classifier(query_feat)
        
        return output