# -*- coding: utf-8 -*-
"""
WiSE-FT: Robust Fine-Tuning of Zero-Shot Models
论文: https://arxiv.org/abs/2109.01903

核心思想：通过权重插值的方式结合预训练模型和微调模型的权重，
在保持zero-shot能力的同时提升few-shot性能。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .finetuning_model import FinetuningModel
from core.utils import accuracy


class WiSEFT(FinetuningModel):
    def __init__(self, feat_dim, num_class, alpha=0.5, 
                 use_weight_interpolation=True, **kwargs):
        super().__init__(**kwargs)
        self.feat_dim = feat_dim
        self.num_class = num_class
        self.alpha = alpha  # 插值参数，0.5表示均等权重
        self.use_weight_interpolation = use_weight_interpolation
        
        # 分类器
        self.classifier = nn.Linear(feat_dim, num_class)
        self.loss_func = nn.CrossEntropyLoss()
        
        # 保存初始权重（预训练状态）
        self.pretrained_state = None
        self._save_pretrained_weights()
        
    def _save_pretrained_weights(self):
        """保存预训练权重"""
        self.pretrained_state = copy.deepcopy(self.state_dict())
        
    def _interpolate_weights(self, alpha=None):
        """执行权重插值
        
        Args:
            alpha: 插值参数，None时使用self.alpha
                   alpha=1.0: 完全使用预训练权重
                   alpha=0.0: 完全使用微调权重
                   alpha=0.5: 均等混合
        """
        if alpha is None:
            alpha = self.alpha
            
        if self.pretrained_state is None:
            print("Warning: No pretrained weights saved, skipping interpolation")
            return
            
        # 获取当前微调后的权重
        current_state = self.state_dict()
        
        # 执行权重插值
        interpolated_state = {}
        for name in current_state:
            if name in self.pretrained_state:
                interpolated_state[name] = (
                    alpha * self.pretrained_state[name] + 
                    (1 - alpha) * current_state[name]
                )
            else:
                # 新参数保持当前值
                interpolated_state[name] = current_state[name]
                
        # 加载插值后的权重
        self.load_state_dict(interpolated_state)
        
    def forward(self, batch):
        """主要的前向传播方法"""
        if self.training:
            return self.set_forward_loss(batch)
        else:
            return self.set_forward(batch)
            
    def set_forward_loss(self, batch):
        """训练时的前向传播和损失计算"""
        image, target = batch
        image = image.to(self.device)
        target = target.to(self.device)
        
        # 标签转换（如果需要）
        if target.min() >= 1:
            target = target - 1
            
        # 提取特征
        feat = self.emb_func(image)
        
        # 分类
        logits = self.classifier(feat)
        
        # 计算损失
        loss = self.loss_func(logits, target)
        
        # 计算准确率
        acc = accuracy(logits, target)
        
        return logits, acc, loss
        
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
        # 创建临时分类器
        classifier = nn.Linear(self.feat_dim, self.way_num).to(self.device)
        optimizer = self.sub_optimizer(classifier, {
            "name": "SGD",
            "kwargs": {
                "lr": 0.01,
                "momentum": 0.9,
                "weight_decay": 0.0005
            }
        })
        
        # 保存临时分类器的初始权重
        initial_classifier_state = copy.deepcopy(classifier.state_dict())
        
        classifier.train()
        support_size = support_feat.size(0)
        
        # 内层训练循环
        for epoch in range(100):  # inner_train_iter
            rand_id = torch.randperm(support_size)
            for i in range(0, support_size, 4):  # inner_batch_size = 4
                select_id = rand_id[i : min(i + 4, support_size)]
                batch = support_feat[select_id]
                target = support_target[select_id]
                
                output = classifier(batch)
                loss = self.loss_func(output, target)
                
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
        
        # WiSE-FT: 对分类器执行权重插值
        if self.use_weight_interpolation:
            current_classifier_state = classifier.state_dict()
            interpolated_classifier_state = {}
            
            for name in current_classifier_state:
                interpolated_classifier_state[name] = (
                    self.alpha * initial_classifier_state[name] + 
                    (1 - self.alpha) * current_classifier_state[name]
                )
            
            classifier.load_state_dict(interpolated_classifier_state)
        
        # 预测
        classifier.eval()
        with torch.no_grad():
            output = classifier(query_feat)
            
        return output
        
    def train_epoch_end(self):
        """每个epoch结束后执行权重插值"""
        if self.use_weight_interpolation and not self.training:
            self._interpolate_weights()
            
    def set_alpha(self, alpha):
        """动态调整插值参数"""
        self.alpha = alpha
        if not self.training:
            self._interpolate_weights()
