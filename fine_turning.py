import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, BertConfig
from PIL import Image
from torchvision import transforms
import torchvision.models as models
import json
import torch.nn.functional as F
from collections import defaultdict
import re
from tqdm import tqdm
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.cuda.amp import autocast, GradScaler

def parse_filename(filename):
    """解析文件名，返回基础名称和版本号"""
    # 匹配格式：基础名称+数字+.扩展名 或 纯数字文件名
    base_match = re.match(r'^(.*?)(\d+)?\.(jpg|png)$', filename)
    if not base_match:
        return filename, "0"  # 默认处理

    base_part = base_match.group(1).rstrip('-')  # 去除结尾的短横线
    version = base_match.group(2) or "0"

    # 处理纯数字文件名（如"123.jpg"）
    if not base_part and base_match.group(2):
        return f"{base_match.group(2)}.jpg", "0"

    return f"{base_part}.{base_match.group(3)}", version


class MultimodalDataset(Dataset):
    def __init__(self, pair_file, image_sim_file, text_sim_file, json_file, image_transform,text_tokenizer):
        # 加载元数据并解析版本号
        self.metadata = {}
        with open(json_file, 'r', encoding='utf-8') as f:
            for item in json.load(f):
                # 解析文件名和版本
                filename = os.path.basename(item['image_data'])
                base_name, version = parse_filename(filename)

                # 更新元数据
                self.metadata[filename] = {
                    'image_data': item['image_data'],
                    'text_data': item['text_data'],
                    'base_name': base_name,
                    'version': version
                }

        # 创建基础名称到所有版本的映射
        self.base_to_versions = defaultdict(list)
        for filename in self.metadata:
            base = self.metadata[filename]['base_name']
            self.base_to_versions[base].append(filename)

        # 加载各相似度数据
        self.pair_file = self.load_data(pair_file)
        self.image_sim = self.load_image(image_sim_file)
        self.text_sim = self.load_text(text_sim_file)

        self.transform = image_transform
        self.tokenizer = text_tokenizer

    @staticmethod
    def load_data(file_path):
        """加载训练/验证对数据"""
        pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(', ')
                if len(parts) == 3:
                    mural, book, sim = parts
                    pairs.append({
                        'mural': mural.strip(),
                        'book': book.strip(),
                        'overall_sim': float(sim)
                    })
        return pairs

    @staticmethod
    def load_image(file_path):
        """加载图像相似度数据"""
        sim_dict = defaultdict(dict)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(', ')
                if len(parts) == 3:
                    mural, book, sim = parts
                    mural_base, _ = parse_filename(mural)
                    book_base, _ = parse_filename(book)
                    sim_dict[mural_base][book_base] = float(sim)
        return sim_dict

    @staticmethod
    def load_text(file_path):
        """加载文本相似度数据"""
        sim_dict = defaultdict(dict)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 使用正则表达式解析复杂格式
                match = re.match(r'^([^:]+):\s*(.*?),\s*([^:]+):\s*(.*?),\s*([0-9.]+)$', line)
                if match:
                    mural, mural_text, book, book_text, sim = match.groups()
                    mural_base, _ = parse_filename(mural)
                    book_base, _ = parse_filename(book)
                    sim_dict[mural_base][book_base] = {
                        'sim': float(sim),
                        'mural_text': mural_text,
                        'book_text': book_text
                    }
        return sim_dict

    def __len__(self):
        return len(self.pair_file)

    def __getitem__(self, idx):
        pair = self.pair_file[idx]

        # 获取实际文件名（处理版本）
        mural_files = self.base_to_versions.get(pair['mural'], [pair['mural']])
        book_files = self.base_to_versions.get(pair['book'], [pair['book']])

        # 选择最新版本（版本号最大的）
        mural_file = max(mural_files, key=lambda x: int(self.metadata[x]['version']))
        book_file = max(book_files, key=lambda x: int(self.metadata[x]['version']))

        # 加载数据
        mural_data = self.metadata[mural_file]
        book_data = self.metadata[book_file]

        # 获取相似度（基于基础名称）
        image_sim = self.image_sim.get(mural_data['base_name'], {}).get(book_data['base_name'], 0.0)
        text_sim_data = self.text_sim.get(mural_data['base_name'], {}).get(book_data['base_name'], {})
        text_sim = text_sim_data.get('sim', 0.0)

        # 加载图像
        mural_img = Image.open(mural_data['image_data']).convert('RGB')
        book_img = Image.open(book_data['image_data']).convert('RGB')

        # 文本编码
        def encode_text(text):
            return self.tokenizer(
                text,
                padding='max_length',
                max_length=128,
                return_tensors='pt',
                truncation=True
            )

        return {
            'mural_image': self.transform(mural_img),
            'book_image': self.transform(book_img),
            'mural_text': encode_text(mural_data['text_data']),
            'book_text': encode_text(book_data['text_data']),
            'labels': {
                'image': torch.tensor(image_sim, dtype=torch.float),
                'text': torch.tensor(text_sim, dtype=torch.float),
                'overall': torch.tensor(pair['overall_sim'], dtype=torch.float)
            }
        }


def collate_fn(batch):
    return {
        'mural_image': torch.stack([x['mural_image'] for x in batch]),
        'book_image': torch.stack([x['book_image'] for x in batch]),
        'mural_text': {
            'input_ids': torch.stack([x['mural_text']['input_ids'].squeeze(0) for x in batch]),
            'attention_mask': torch.stack([x['mural_text']['attention_mask'].squeeze(0) for x in batch])
        },
        'book_text': {
            'input_ids': torch.stack([x['book_text']['input_ids'].squeeze(0) for x in batch]),
            'attention_mask': torch.stack([x['book_text']['attention_mask'].squeeze(0) for x in batch])
        },
        'labels': {
            'image': torch.stack([x['labels']['image'] for x in batch]),
            'text': torch.stack([x['labels']['text'] for x in batch]),
            'overall': torch.stack([x['labels']['overall'] for x in batch])
        }
    }

class MultimodalSimilarityData:
    def __init__(self, image_sim_file, text_sim_file, overall_sim_file):
        self.image_sim = self._load_txt_similarity(image_sim_file)
        self.text_sim = self._load_txt_similarity(text_sim_file)
        self.overall_sim = self._load_json_similarity(overall_sim_file)

    def _load_txt_similarity(self, file_path):
        sim_dict = defaultdict(dict)
        with open(file_path, 'r') as f:
            for line in f:
                mural_id, book_id, sim = line.strip().split(',')
                sim_dict[mural_id][book_id] = float(sim)
        return sim_dict

    def _load_json_similarity(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    def get_sample(self, mural_id, book_id):
        return {
            'image_sim': self.image_sim[mural_id][book_id],
            'text_sim': self.text_sim[mural_id][book_id],
            'overall_sim': self.overall_sim[mural_id][book_id]
        }



class SimcseModel(nn.Module):
    def __init__(self, pretrained_model, pooling='cls', dropout=0.3):
        super().__init__()
        config = BertConfig.from_pretrained(pretrained_model)
        config.attention_probs_dropout_prob = dropout
        config.hidden_dropout_prob = dropout
        self.bert = BertModel.from_pretrained(pretrained_model, config=config)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                          output_hidden_states=True, return_dict=True)
        # 简化后的池化方法
        if self.pooling == 'cls':
            return outputs.last_hidden_state[:, 0]
        elif self.pooling == 'mean':
            return outputs.last_hidden_state.mean(dim=1)
        else:
            raise ValueError("Unsupported pooling method")


class MultimodalEncoder(nn.Module):
    def __init__(self, text_encoder, image_encoder):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.text_proj = nn.Linear(768, 512)
        self.image_proj = nn.Linear(1536, 512)

    def forward(self, text_input, image_input):
        text_feat = self.text_encoder(
            input_ids=text_input['input_ids'],
            attention_mask=text_input['attention_mask']
        )
        text_feat = self.text_proj(text_feat)

        image_feat = self.image_encoder(image_input)
        image_feat = self.image_proj(image_feat)

        return text_feat, image_feat


class CrossModalAttention(nn.Module):
    def __init__(self, dim=512, num_heads=8):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads)
        self.norm = nn.LayerNorm(dim)  # 添加层归一化

    def forward(self, query, key_value):
        # 维度调整 (batch_size, seq_len, features) → (seq_len, batch_size, features)
        assert query.dim() == 2, f"Expected 2D query, got {query.shape}"
        assert key_value.dim() == 2, f"Expected 2D key_value, got {key_value.shape}"

        query = query.unsqueeze(0)  # (seq_len=1, batch_size, features)
        key_value = key_value.unsqueeze(0)

        query = query.permute(1, 0, 2)
        key_value = key_value.permute(1, 0, 2)

        attn_output, _ = self.mha(
            query=query,
            key=key_value,
            value=key_value
        )

        # 恢复维度并添加残差连接
        output = attn_output.permute(1, 0, 2) + query.permute(1, 0, 2)
        output = output.squeeze(0)
        return self.norm(output)

class MultimodalSimilarityModel(nn.Module):
    def __init__(self, text_encoder, image_encoder):
        super().__init__()
        self.encoder = MultimodalEncoder(text_encoder, image_encoder)

        self.image_to_text = CrossModalAttention(dim=512, num_heads=16)
        self.text_to_image = CrossModalAttention(dim=512, num_heads=16)
        # 单独定义LayerNorm
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(512)

        self.overall_head = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, mural_inputs, book_inputs):
        # 编码特征
        mural_text, mural_image = self.encoder(mural_inputs['text'], mural_inputs['image'])
        book_text, book_image = self.encoder(book_inputs['text'], book_inputs['image'])

        # 跨模态交互
        mural_cross = self.norm1(self.image_to_text(mural_text, mural_image))
        book_cross = self.norm2(self.text_to_image(book_text, book_image))

        # 特征融合
        combined = torch.cat([mural_cross, book_cross], dim=1)
        similarity = self.overall_head(combined)

        return {
            'overall': similarity.squeeze(),
            'features': {
                'mural_text': mural_text,
                'mural_image': mural_image,
                'book_text': book_text,
                'book_image': book_image
            }
        }


class MultimodalConsistencyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.3, learnable=True):
        super().__init__()
        if learnable:
            self.alpha = nn.Parameter(torch.tensor(alpha))
            self.beta = nn.Parameter(torch.tensor(beta))
        else:
            self.alpha = alpha
            self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, preds, features, labels):
        # 主损失
        loss_main = self.mse(preds['overall'], labels['overall'])

        # 跨模态对齐损失
        img2text = F.cosine_similarity(features['mural_image'], features['book_text'])
        text2img = F.cosine_similarity(features['mural_text'], features['book_image'])
        loss_cross = (self.mse(img2text, labels['overall']) + self.mse(text2img, labels['overall'])) / 2

        # 模态内一致性
        loss_image = self.mse(F.cosine_similarity(features['mural_image'], features['book_image']), labels['image'])
        loss_text = self.mse(F.cosine_similarity(features['mural_text'], features['book_text']), labels['text'])

        margin = 0.2
        pos_sim = F.cosine_similarity(features['mural_text'], features['book_text'])
        neg_sim = F.cosine_similarity(features['mural_text'], features['book_text'].roll(shifts=1, dims=0))
        loss_contrastive = torch.mean(torch.clamp(margin - pos_sim + neg_sim, min=0))

        return loss_main + 0.3*loss_cross + 0.2*(loss_image+loss_text) + 0.1*loss_contrastive


def load_models(checkpoint_path, device):
    # 初始化模型组件
    text_encoder = SimcseModel('pretrain_model/bert-base-chinese').to(device)
    image_encoder = models.efficientnet_b3(pretrained=False)
    image_encoder.classifier = nn.Identity()

    # 构建完整模型
    model = MultimodalSimilarityModel(text_encoder, image_encoder).to(device)

    # 加载checkpoint

    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    return model

# 训练函数
def train(model, dataloader, optimizer, criterion, device):
    scaler = GradScaler()
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch in dataloader:
        # 数据准备（修改后的结构）
        mural_inputs = {
            'text': {
                'input_ids': batch['mural_text']['input_ids'].to(device),
                'attention_mask': batch['mural_text']['attention_mask'].to(device)
            },
            'image': batch['mural_image'].to(device)
        }
        book_inputs = {
            'text': {
                'input_ids': batch['book_text']['input_ids'].to(device),
                'attention_mask': batch['book_text']['attention_mask'].to(device)
            },
            'image': batch['book_image'].to(device)
        }
        labels = {
            'image': batch['labels']['image'].to(device),
            'text': batch['labels']['text'].to(device),
            'overall': batch['labels']['overall'].to(device)
        }

        # 梯度清零
        optimizer.zero_grad()
        # （混合精度）
        with autocast():  # AMP上下文
            outputs = model(mural_inputs, book_inputs)
            loss = criterion(outputs, outputs['features'], labels)


        # 反向传播
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss / (len(progress_bar) + 1e-7):.4f}"
        })

    return total_loss / len(dataloader)

# 测试和评估函数
def evaluate(model, dataloader, device):
    model.eval()
    mae, mse = 0.0, 0.0
    progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in dataloader:
            # 数据准备与train函数保持一致
            mural_inputs = {
                'text': {
                    'input_ids': batch['mural_text']['input_ids'].to(device),
                    'attention_mask': batch['mural_text']['attention_mask'].to(device)
                },
                'image': batch['mural_image'].to(device)
            }
            book_inputs = {
                'text': {
                    'input_ids': batch['book_text']['input_ids'].to(device),
                    'attention_mask': batch['book_text']['attention_mask'].to(device)
                },
                'image': batch['book_image'].to(device)
            }
            labels = batch['labels']['overall'].to(device)

            outputs = model(mural_inputs, book_inputs)
            preds = outputs['overall']

            mae += F.l1_loss(preds, labels).item()
            mse += F.mse_loss(preds, labels).item()

            progress_bar.set_postfix({
                'current_mae': f"{F.l1_loss(preds, labels).item():.4f}",
                'current_mse': f"{F.mse_loss(preds, labels).item():.4f}"
            })
    return mae / len(dataloader), mse / len(dataloader)


# 主程序
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    epochs = 20
    lr = 1e-4

    checkpoint_path = 'D:\PYCHARM\\壁画\\checkpoints\\new\\0.2-0.6\\checkpoint.pth'

    data_config = {
        'train': {
            'pair_file': 'D:/PYCHARM/壁画/data_make/6-2-2/train.txt',
            'image_sim_file': 'D:/PYCHARM/壁画/data_make/image_similarity.txt',
            'text_sim_file': 'D:/PYCHARM/壁画/data_make/text_similarity.txt',
            'json_file': 'D:/PYCHARM/壁画/data_make/pre_data.json'
        },
        'val': {
            'pair_file': 'D:/PYCHARM/壁画/data_make/6-2-2/val.txt',
            'image_sim_file': 'D:/PYCHARM/壁画/data_make/image_similarity.txt',
            'text_sim_file': 'D:/PYCHARM/壁画/data_make/text_similarity.txt',
            'json_file': 'D:/PYCHARM/壁画/data_make/pre_data.json'
        },
        'test': {
            'pair_file': 'D:/PYCHARM/壁画/data_make/6-2-2/test.txt',
            'image_sim_file': 'D:/PYCHARM/壁画/data_make/image_similarity.txt',
            'text_sim_file': 'D:/PYCHARM/壁画/data_make/text_similarity.txt',
            'json_file': 'D:/PYCHARM/壁画/data_make/pre_data.json'
        }
    }


    output_dir = 'D:/PYCHARM/壁画/save_model/new/622'
    # 检查 output_dir 是否存在，如果不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory {output_dir} created.")
    else:
        print(f"Directory {output_dir} already exists.")

    # 加载BERT分词器
    pretrained_model = 'pretrain_model\\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    best_loss = 0

    # 模型和优化器
    model = load_models(checkpoint_path,device)
    criterion = MultimodalConsistencyLoss(learnable=True)
    optimizer = optim.Adam([
        {'params': model.parameters()},
        {'params': criterion.parameters()}
    ], lr=lr)

    # 加载数据集
    train_dataset = MultimodalDataset(
        pair_file=data_config['train']['pair_file'],
        image_sim_file=data_config['train']['image_sim_file'],
        text_sim_file=data_config['train']['text_sim_file'],
        json_file=data_config['train']['json_file'],
        image_transform=transform,
        text_tokenizer=tokenizer
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)

    val_dataset = MultimodalDataset(
        pair_file=data_config['val']['pair_file'],
        image_sim_file=data_config['val']['image_sim_file'],
        text_sim_file=data_config['val']['text_sim_file'],
        json_file=data_config['val']['json_file'],
        image_transform=transform,
        text_tokenizer=tokenizer
    )
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn)

    test_dataset = MultimodalDataset(
        pair_file=data_config['test']['pair_file'],
        image_sim_file=data_config['test']['image_sim_file'],
        text_sim_file=data_config['test']['text_sim_file'],
        json_file=data_config['test']['json_file'],
        image_transform=transform,
        text_tokenizer=tokenizer
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,collate_fn=collate_fn)

    log_file = os.path.join(output_dir, "training_log.txt")
    with open(log_file, 'w') as log:
        log.write("Epoch, Train Loss, Validation MSE, Validation MAE\n")

    # 训练和评估过程
    with tqdm(range(epochs), desc="Total Progress") as epoch_pbar:
        for epoch in epoch_pbar:

            print(f'Epoch {epoch + 1}/{epochs}')

            # 训练模型
            train_loss = train(model, train_dataloader, optimizer,criterion, device)
            print(f"Training Loss: {train_loss:.4f}")

            # 评估模型
            eval_mae, eval_mse = evaluate(model, val_dataloader, device)
            print(f"Evaluation MAE: {eval_mae:.4f}, MSE: {eval_mse:.4f}")

            epoch_pbar.set_postfix({
                'train_loss': f"{train_loss:.4f}",
                'val_mae': f"{eval_mae:.4f}",
                'val_mse': f"{eval_mse:.4f}"
            })

            with open(log_file, 'a') as log:
                log.write(f"{epoch}, {train_loss}, {eval_mse}, {eval_mae}\n")

            # 保存模型
            model_path = os.path.join(output_dir, "model.pth")  # 保存在models文件夹中
            torch.save(model.state_dict(), model_path)
            # 检查是否为最好的模型
            if eval_mse < best_loss:
                best_loss = eval_mse
                best_epoch = epoch + 1  # 记录最好的epoch

                # 保存最好的模型
                torch.save(model.state_dict(), model_path)
                print(f"Best model saved at epoch {epoch + 1} with MSE: {best_loss:.4f}")

    print("Training completed.")

    # 在所有训练完成后进行最终的测试
    print("Training completed. Starting testing phase...")

    # 测试模型
    test_mae, test_mse = evaluate(model, test_dataloader, device)
    print(f"Test MAE: {test_mae:.4f}, MSE: {test_mse:.4f}")
    # 记录日志
    with open(log_file, 'a') as log:
        log.write(f"Test MSE: {test_mse}, MAE: {test_mae}\n")

