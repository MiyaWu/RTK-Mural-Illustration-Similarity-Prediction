import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertModel, BertConfig
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import os
from tqdm import tqdm
import torch.nn.functional as F
import logging
from nlpaug.augmenter.word import RandomWordAug,ContextualWordEmbsAug
from torch.cuda.amp import autocast, GradScaler

class MultimodalSimilarityModel(nn.Module):
    def __init__(self, bert_model_path, pooler, dropout, device, feature_dim):
        super(MultimodalSimilarityModel, self).__init__()

        # 文本编码器（BERT）
        self.text_encoder = SimcseModel(pretrained_model=bert_model_path, pooling=pooler, dropout=dropout).to(device)
        self.text_projection = nn.Linear(self.text_encoder.bert.config.hidden_size, feature_dim)

        # 图像编码器（EfficientNet）
        self.image_encoder = models.efficientnet_b3(pretrained=False)  # 不加载预训练权重
        self.image_encoder.classifier = torch.nn.Identity()
        self.image_projection = nn.Linear(1536, feature_dim)  # EfficientNet输出为1536维

        # # logit_scale（一个可训练的标量，用于调整相似度的大小）
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1/0.07)))

    def forward(self, image_input, text_input):
        # 获取文本特征
        text_features = self.text_encoder(input_ids=text_input['input_ids'],
                                          attention_mask=text_input['attention_mask'],
                                          token_type_ids=text_input.get('token_type_ids'))  # 使用字典中的 token_type_ids
        text_features = self.text_projection(text_features)  # 投影到相同的特征空间

        # 获取图像特征
        image_features = self.image_encoder(image_input).view(image_input.size(0), -1)
        image_features = self.image_projection(image_features)  # 投影到相同的特征空间

        text_features = F.normalize(text_features, p=2, dim=-1)
        image_features = F.normalize(image_features, p=2, dim=-1)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)  # 梯度裁剪

        return image_features, text_features,logit_scale



class SimcseModel(nn.Module):
    """Simcse无监督模型定义"""

    def __init__(self, pretrained_model, pooling, dropout):
        super(SimcseModel, self).__init__()
        config = BertConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout  # 修改config的dropout系数
        config.hidden_dropout_prob = dropout
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask, token_type_ids=None):  # token_type_ids is optional
        out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                        output_hidden_states=True, return_dict=True)

        if self.pooling == 'cls':
            return out.last_hidden_state[:, 0]  # [batch, 768]
        if self.pooling == 'pooler':
            return out.pooler_output  # [batch, 768]
        if self.pooling == 'last-avg':
            last = out.last_hidden_state.transpose(1, 2)  # [batch, 768, seqlen]
            return torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
        if self.pooling == 'first-last-avg':
            first = out.hidden_states[1].transpose(1, 2)  # [batch, 768, seqlen]
            last = out.hidden_states[-1].transpose(1, 2)  # [batch, 768, seqlen]
            first_avg = torch.avg_pool1d(first, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            last_avg = torch.avg_pool1d(last, kernel_size=last.shape[-1]).squeeze(-1)  # [batch, 768]
            avg = torch.cat((first_avg.unsqueeze(1), last_avg.unsqueeze(1)), dim=1)  # [batch, 2, 768]
            return torch.avg_pool1d(avg.transpose(1, 2), kernel_size=2).squeeze(-1)  # [batch, 768]


class MultimodalSimilarityData:
    def __init__(self, image_paths, text_data, tokenizer, transform=None, max_length=20, text_aug_prob=0.3):
        self.image_paths = image_paths
        self.text_data = text_data
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.text_aug = [
            RandomWordAug(
                action='delete',
                aug_p=text_aug_prob  # 每个汉字被删除的概率

            ),

            # 中文 BERT 上下文替换（替换为中文预训练模型）
            ContextualWordEmbsAug(
                model_path='./pretrain_model/bert-base-chinese',  # 使用中文 BERT 模型
                action="substitute",
                device='cuda' if torch.cuda.is_available() else 'cpu'
            ),
        ]
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 图像增强（双视图）
        image = Image.open(self.image_paths[idx]).convert('RGB')
        image1 = self.transform(image) if self.transform else image
        image2 = self.transform(image) if self.transform else image

        # 文本增强（双视图）
        text = self.text_data[idx]
        aug = np.random.choice(self.text_aug)
        text_aug1 = aug.augment(text)
        text_aug2 = aug.augment(text)

        # 分词处理
        text_input1 = self.tokenizer(
            text_aug1,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        text_input1['token_type_ids'] = torch.zeros_like(text_input1['input_ids'])

        text_input2 = self.tokenizer(
            text_aug2,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        text_input2['token_type_ids'] = torch.zeros_like(text_input2['input_ids'])

        return image1, image2, text_input1, text_input2

def compute_loss( image1, text1, logit_scale,  image2, text2, tau=0.07):
    """

    """

    # # 计算图像-文本之间的损失

    logit_scale = logit_scale.exp()  # 转换为线性空间缩放因子

    logits_per_image1 = logit_scale * image1 @ text1.t()
    logits_per_text1 = logit_scale * text1 @ image1.t()

    # 图像→文本相似度（增强视图）
    logits_per_image2 = logit_scale * image2 @ text2.t()
    logits_per_text2 = logit_scale * text2 @ image2.t()


    labels1 = torch.arange(len(image1), device=device)
    labels2 = torch.arange(len(image2), device=device)

    loss_i1t1 = (F.cross_entropy(logits_per_image1, labels1) + F.cross_entropy(logits_per_text1, labels1)) / 2
    loss_i2t2 = (F.cross_entropy(logits_per_image2, labels2) + F.cross_entropy(logits_per_text2, labels2)) / 2
    loss_image_text = (loss_i1t1 + loss_i2t2) * 0.5  # 模态间损失

    # 计算模态内损失：图像-图像损失和文本-文本损失
    image_sim_loss = compute_info_nce_loss(image1, image2, tau)
    text_sim_loss = compute_info_nce_loss(text1, text2, tau)

    total_loss_batch = loss_image_text*0.6 + (image_sim_loss + text_sim_loss)*0.2

    return loss_image_text, image_sim_loss, text_sim_loss, total_loss_batch



def compute_info_nce_loss(features, features_aug, tau):
    """

    """
    features = F.normalize(features, dim=-1)  # [batch, dim]
    features_aug = F.normalize(features_aug, dim=-1)  # [batch, dim]

    batch_size = features.size(0)
    # 计算相似度矩阵，使用余弦相似度
    sim_matrix = F.cosine_similarity(features.unsqueeze(1), features_aug.unsqueeze(0), dim=-1)
    labels = torch.arange(batch_size).to(features.device)


    sim_matrix = sim_matrix / tau  # 对比损失的温度缩放
    loss_fct = nn.CrossEntropyLoss()
    return loss_fct(sim_matrix, labels)


def train_model(model, data_loader, optimizer, device, epochs, checkpoint_dir):
    scaler = GradScaler()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_image_text_loss = 0
        total_image_sim_loss = 0
        total_text_sim_loss = 0

        for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            image1, image2, text_input1, text_input2 = batch
            # 移至设备
            image1 = image1.to(device)
            image2 = image2.to(device)
            text_input1 = {k: v.squeeze(1).to(device) for k, v in text_input1.items()}
            text_input2 = {k: v.squeeze(1).to(device) for k, v in text_input2.items()}

            optimizer.zero_grad()
            with autocast():  # 自动混合精度上下文
                image_features_1, text_features_1, logit_scale = model(image1, text_input1)
                image_features_2, text_features_2, logit_scale = model(image2, text_input2)

                loss_image_text, image_sim_loss, text_sim_loss, total_loss_batch = compute_loss(
                    image_features_1, text_features_1, model.logit_scale,
                    image_features_2, text_features_2, tau=0.07
                )

            scaler.scale(total_loss_batch).backward()
            scaler.step(optimizer)
            scaler.update()

            # 累加损失
            total_loss += total_loss_batch.item()
            total_image_text_loss += loss_image_text.item()
            total_image_sim_loss += image_sim_loss.item()
            total_text_sim_loss += text_sim_loss.item()

            del image1, image2, text_input1, text_input2, image_features_1, image_features_2, text_features_1, text_features_2
            torch.cuda.empty_cache()

        # 记录每个 epoch 的平均损失
        avg_total_loss = total_loss / len(data_loader)
        avg_image_text_loss = total_image_text_loss / len(data_loader)
        avg_image_sim_loss = total_image_sim_loss / len(data_loader)
        avg_text_sim_loss = total_text_sim_loss / len(data_loader)

        logging.info(f"Epoch {epoch + 1}, Average Loss: Total Loss = {avg_total_loss}, "
                     f"Image_text Loss = {avg_image_text_loss},  "
                     f"Image Similarity Loss = {avg_image_sim_loss}, "
                     f"Text Similarity Loss = {avg_text_sim_loss}")


        # 保存模型
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"checkpoint.pth"))


def load_model(bert_model_path, pooler, dropout, device, feature_dim=512, checkpoint_path=None):
    model = MultimodalSimilarityModel(bert_model_path, pooler, dropout, device, feature_dim=feature_dim)
    if checkpoint_path:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Model loaded from {checkpoint_path}")
    return model


def prepare_data(image_paths, text_data, tokenizer, batch_size, transform=None, max_length=20):
    dataset = MultimodalSimilarityData(image_paths, text_data, tokenizer, transform, max_length=max_length)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


def load_json_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    image_paths = [item["image_data"] for item in data]
    text_data = [item["text_data"] for item in data]
    return image_paths, text_data


if __name__ == "__main__":
    # 加载数据
    json_path = "./pretrain/output_pairs.json"  # 你的JSON文件路径
    image_paths, text_data = load_json_data(json_path)

    # 设置参数
    pooler = 'cls'
    dropout = 0.1
    batch_size = 16
    epoch = 30
    checkpoint_dir = 'checkpoints/new/0.2-0.6'
    bert_model_path = './pretrain_model/bert-base-chinese'
    tokenizer = BertTokenizerFast.from_pretrained(bert_model_path)

    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # 配置日志记录
    log_file_path = os.path.join(checkpoint_dir, 'training_log.txt')
    logging.basicConfig(filename=log_file_path, level=logging.INFO,
                        format='%(asctime)s - %(message)s')


    color_jitter = transforms.ColorJitter(
        0.4, 0.4, 0.4, 0.1
    )
    transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224),
        transforms.RandomHorizontalFlip(),  # with 0.5 probability
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ])

    # 准备数据
    data_loader = prepare_data(image_paths, text_data, tokenizer, batch_size=batch_size, transform=transform)

    # 初始化模型
    model = load_model(bert_model_path, pooler, dropout, device, feature_dim=512)

    model.to(device)

    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=3e-5)

    # 开始训练
    train_model(model, data_loader, optimizer, device, epochs=epoch, checkpoint_dir=checkpoint_dir)
