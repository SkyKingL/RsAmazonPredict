import torch
import torch.nn as nn
import torch.nn.functional as F

class MF(nn.Module):
    def __init__(self, user_count, item_count, dim=64, gpu='cpu'):
        super(MF, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.device = gpu
        
        # 用户和物品的嵌入层（确保使用float32）
        self.user_emb = nn.Embedding(user_count, dim).float()
        self.item_emb = nn.Embedding(item_count, dim).float()
        
        # 添加偏置项
        self.user_bias = nn.Embedding(user_count, 1).float()
        self.item_bias = nn.Embedding(item_count, 1).float()
        self.global_bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        
        # 初始化权重
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user, item):
        # 确保输入是LongTensor类型
        user = user.long()
        item = item.long()
        
        # 获取嵌入向量（已确保是float32类型）
        user_embed = self.user_emb(user)
        item_embed = self.item_emb(item)
        
        # 获取偏置项
        user_bias = self.user_bias(user).squeeze()
        item_bias = self.item_bias(item).squeeze()
        
        # 计算预测评分
        rating = (user_embed * item_embed).sum(dim=1)
        rating += user_bias + item_bias + self.global_bias
        
        return rating