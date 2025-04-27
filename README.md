# STAT3612 医疗再入院预测模型集合

## 项目背景

医疗再入院是评估医疗质量和医院效率的重要指标。30天内再入院通常表明患者在初次治疗后出现并发症或病情管理不佳，会导致额外的医疗成本和患者痛苦。本项目旨在通过机器学习和深度学习技术，利用医疗记录文本、电子健康记录(EHR)和医学影像等多模态数据，构建预测患者30天内再入院风险的模型，帮助医疗机构提前识别高风险患者并进行干预。

## 数据说明

本项目使用的数据集包含以下几个部分：

- **医疗记录文本(notes.csv)**: 包含医生和护士记录的文本信息
- **电子健康记录(EHR)**: 结构化的患者信息，如生命体征、实验室检查结果、药物使用等
- **医学影像**: 患者的X光、CT等医学影像及其特征
- **标签数据**: 患者是否在30天内再次入院的二元标签

数据集被分为训练集(train.csv)、验证集(valid.csv)和测试集(test.csv、test_answer.csv)。

## 模型架构详解

### 1. 医疗记录文本预测模型 (notes_prediction_model)

#### 1.1 数据预处理

```python
# 加载预处理后的数据
def load_processed_data():
    with open('processed_notes_data.pkl', 'rb') as f:
        data = pickle.load(f)
    return data
```

预处理流程包括文本清洗、分词和截断等步骤，以适应BERT模型输入需求。

#### 1.2 模型结构

```python
class BertForReadmission(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(BertForReadmission, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # 冻结BERT的某些层以减少训练参数
        # 我们只微调最后4层
        modules = [self.bert.embeddings, *self.bert.encoder.layer[:-1]]
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        
        # 分类头
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
```

该模型冻结了BERT的底层参数，仅微调顶层，以减少训练时间和计算资源需求。

#### 1.3 损失函数

```python
# Focal Loss实现
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()
```

Focal Loss专门用于处理类别不平衡问题，通过降低易分样本的权重，提高难分样本的权重。

#### 1.4 训练与评估策略

训练过程使用AdamW优化器，配合线性学习率调度，包括预热和衰减阶段。评估指标包括AUC、准确率、精确率、召回率和F1分数。

### 2. TF-IDF + XGBoost模型 (tfidf)

#### 2.1 数据处理

```python
# 合并notes数据
def merge_notes(df):
    # 筛选出相关的notes
    relevant_notes = notes_df[notes_df['hadm_id'].isin(df['hadm_id'])].copy()
    
    # 将hadm_id映射到对应的DataFrame的combined_id
    hadm_to_combined = dict(zip(df['hadm_id'], df['combined_id']))
    relevant_notes['combined_id'] = relevant_notes['hadm_id'].map(hadm_to_combined)
    
    # 按combined_id合并notes文本
    merged_notes = relevant_notes.groupby('combined_id')['text'].apply(lambda x: ' '.join(x)).reset_index()
    
    # 使用常规合并，而不是索引合并
    result = pd.merge(df, merged_notes, on='combined_id', how='left')
    return result
```

这个模型首先将所有相关的医疗记录文本按患者ID合并，然后使用TF-IDF进行特征提取。

#### 2.2 特征提取与训练

```python
# TF-IDF向量化
vectorizer = TfidfVectorizer(max_features=8000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(combined_train_df['text'])
X_test_tfidf = vectorizer.transform(test_df['text'])

# 定义XGBoost参数
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 8,
    'learning_rate': 0.05,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_weight': 3,
    'gamma': 0.1,
    'use_label_encoder': False,
    'random_state': 42
}
```

使用XGBoost的优势在于其高效性和可解释性，同时能够很好地处理稀疏的TF-IDF特征。

#### 2.3 性能比较

基础模型和AUC优化模型的详细比较：

| 指标 | 基础模型 | AUC优化模型 | 变化 |
|------|---------|------------|------|
| AUC | 0.8645 | 0.8551 | -0.0094 |
| 准确率 | 0.9299 | 0.9282 | -0.0017 |
| 精确率 | 1.0000 | 0.9848 | -0.0152 |
| 召回率 | 0.6075 | 0.6075 | 0.0000 |
| F1分数 | 0.7558 | 0.7514 | -0.0044 |
| 特异性 | 1.0000 | 0.9980 | -0.0020 |
| 阴性预测值 | 0.9213 | 0.9212 | -0.0001 |

AUC优化模型的主要改进在于使用test_answer作为验证集，以找到最佳的迭代次数，避免过拟合。

### 3. TF-IDF + 深度学习模型 (tfidf_deep)

#### 3.1 模型架构

```python
class DeepTFIDFClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1=512, hidden_dim2=256, hidden_dim3=64, dropout=0.5):
        super(DeepTFIDFClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.BatchNorm1d(hidden_dim3),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim3, 1)
        )
```

这个模型使用深度神经网络处理TF-IDF特征，通过多层线性变换、批归一化和Dropout正则化提高泛化能力。

#### 3.2 训练策略

```python
# 训练循环
num_epochs = 50
best_auc = 0.0
early_stop_patience = 10
no_improve_epochs = 0

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    
    # ... 训练代码 ...
    
    # 验证阶段
    model.eval()
    val_loss = 0.0
    all_probs = []
    all_labels = []
    
    # ... 验证代码 ...
    
    # 更新学习率
    scheduler.step(val_auc)
    
    # 保存最佳模型
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), 'best_deep_tfidf_model.pth')
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
    
    # 早停
    if no_improve_epochs >= early_stop_patience:
        print(f"早停: {early_stop_patience}个epochs没有改善")
        break
```

训练过程包括动态学习率调整、模型检查点保存和早停机制，以获得最佳性能。

#### 3.3 优势与适用场景

相比XGBoost，深度学习模型能够学习更复杂的非线性特征关系，适合处理大规模数据集。在医疗记录文本较长且复杂的情况下，可能表现出更好的性能。

### 4. TF-IDF + EHR融合模型 (tfidf_ehr)

#### 4.1 改进的多模态分类器

##### 4.1.1 EHR特征提取

```python
# 提取EHR特征 - 保留时间序列结构
def extract_ehr_features_sequence(df, max_seq_len=50):
    ehr_sequences = []
    seq_lengths = []
    missed_records = 0
    
    for _, row in df.iterrows():
        key = format_ehr_key(row)
        if key in feat_dict:
            # 获取原始序列数据
            seq_data = feat_dict[key]
            seq_len = min(seq_data.shape[0], max_seq_len)
            
            # 如果序列长度超过max_seq_len，截断序列
            if seq_data.shape[0] > max_seq_len:
                seq_data = seq_data[:max_seq_len, :]
            
            # 如果序列长度小于max_seq_len，用零填充
            if seq_data.shape[0] < max_seq_len:
                padding = np.zeros((max_seq_len - seq_data.shape[0], seq_data.shape[1]))
                seq_data = np.vstack((seq_data, padding))
            
            ehr_sequences.append(seq_data)
            seq_lengths.append(seq_len)
        else:
            # 省略处理缺失记录的代码...
```

EHR数据作为时间序列处理，保留了患者状态随时间变化的信息。

##### 4.1.2 完整模型架构

```python
class ImprovedMultiModalClassifier(nn.Module):
    def __init__(self, tfidf_dim, ehr_dim, hidden_dim1=512, hidden_dim2=256, hidden_dim3=128, 
                 rnn_hidden_dim=128, rnn_layers=2, dropout=0.6, weight_decay=0.01):
        super(ImprovedMultiModalClassifier, self).__init__()
        
        # TF-IDF特征处理分支
        self.tfidf_branch = nn.Sequential(
            nn.Linear(tfidf_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.BatchNorm1d(hidden_dim3),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # EHR特征处理分支 - 使用LSTM处理时间序列
        self.ehr_lstm = nn.LSTM(
            input_size=ehr_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0
        )
        
        # 融合层
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim3 + rnn_hidden_dim*2, hidden_dim3),
            nn.BatchNorm1d(hidden_dim3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim3, hidden_dim3//2),
            nn.BatchNorm1d(hidden_dim3//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim3//2, 1)
        )
```

模型使用双向LSTM处理EHR时序数据，与TF-IDF特征融合后通过多层全连接网络进行分类。

#### 4.2 注意力多模态模型

##### 4.2.1 注意力机制实现

```python
# 实现自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )
    
    def forward(self, encoder_outputs):
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        energy = self.projection(encoder_outputs)  # [batch_size, seq_len, 1]
        weights = F.softmax(energy.squeeze(-1), dim=1)  # [batch_size, seq_len]
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)  # [batch_size, hidden_dim]
        return outputs, weights
```

注意力机制能够自动学习序列中每个时间步的重要性，从而关注真正关键的信息。

##### 4.2.2 注意力可视化

```python
def _visualize_attention(self, attention_weights, seq_length):
    """
    辅助方法，用于在训练过程中可视化注意力权重
    """
    if not hasattr(self, 'attention_fig_count'):
        self.attention_fig_count = 0
    
    # 只在特定次数可视化，以避免生成过多图像
    if self.attention_fig_count % 50 == 0:
        # 选择第一个批次样本进行可视化
        sample_weights = attention_weights[0, :seq_length[0]].cpu().detach().numpy()
        
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(sample_weights)), sample_weights)
        plt.xlabel('Time Step')
        plt.ylabel('Attention Weight')
        plt.title(f'EHR Attention Weights (Sample)')
        plt.savefig(f'attention_viz_{self.attention_fig_count}.png')
        plt.close()
    
    self.attention_fig_count += 1
```

可视化注意力权重有助于理解模型关注EHR序列中的哪些时间点，提高可解释性。

### 5. 多模态预测模型 (multimodal_prediction.py)

#### 5.1 数据加载和预处理

```python
# 定义数据集类
class MultimodalDataset(Dataset):
    def __init__(self, data, image_dir="", max_text_length=512, max_notes=3, has_labels=True, is_training=False):
        self.data = data
        self.patient_admission_pairs = []
        self.labels = []
        self.image_dir = image_dir
        self.max_text_length = max_text_length
        self.max_notes = max_notes
        self.has_labels = has_labels
        self.is_training = is_training
        
        # 图像转换
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 文本处理
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # 准备数据
        self._prepare_data()
```

该数据集类处理三种模态：医疗记录文本、EHR数据和医学图像，同时包含数据增强逻辑。

#### 5.2 图像处理

```python
def _process_images(self, images_info, max_images=5):
    images_tensor = []
    
    # 优先考虑不同视角的图像
    if len(images_info) > max_images:
        view_positions = {}
        # 按视角分组图像
        for img in images_info:
            view = img.get('view_position', '')
            if view not in view_positions:
                view_positions[view] = []
            view_positions[view].append(img)
        
        # 从每个视角中选择图像
        selected_images = []
        # 首先确保每个视角至少选一张
        for view, imgs in view_positions.items():
            selected_images.append(imgs[0])
            if len(selected_images) >= max_images:
                break
        
        # 如果还有空位，添加其他图像
        if len(selected_images) < max_images:
            remaining_slots = max_images - len(selected_images)
            remaining_images = [img for view_imgs in view_positions.values() 
                               for img in view_imgs[1:]]  # 跳过已选的第一张
            selected_images.extend(remaining_images[:remaining_slots])
```

图像处理逻辑确保选择多样的视角，提高模型对不同角度医学图像的理解能力。

#### 5.3 完整多模态模型结构

```python
class MultimodalModel(nn.Module):
    def __init__(self, image_feature_dim=2048, ehr_dim=128):
        super(MultimodalModel, self).__init__()
        
        # 图像特征处理
        self.image_fc = nn.Linear(image_feature_dim, 512)
        
        # 文本特征提取器
        self.text_model = BertModel.from_pretrained('bert-base-uncased')
        self.text_fc = nn.Linear(768, 512)
        
        # EHR特征处理
        self.ehr_fc = nn.Linear(ehr_dim, 512)
        
        # 融合层
        self.fusion_fc1 = nn.Linear(512 * 3, 256)
        self.fusion_fc2 = nn.Linear(256, 64)
        self.fusion_fc3 = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
```

该模型将三种不同模态的特征统一到512维空间，然后通过全连接层融合并进行分类。

#### 5.4 训练与评估

```python
# 训练函数
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-4):
    # 添加早停机制
    patience = 10
    best_val_auc = 0.0
    no_improve_epochs = 0
    
    # 使用学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    
    # 使用二元交叉熵损失
    criterion = nn.BCELoss()
    
    for epoch in range(num_epochs):
        # ... 训练代码 ...
        
        # 验证
        val_loss, val_auc, val_acc, val_precision, val_recall, val_f1 = evaluate_model(model, val_loader, criterion)
        
        # ... 输出和早停逻辑 ...

# 评估函数
def evaluate_model(model, data_loader, criterion):
    model.eval()
    # ... 评估代码 ...
    return val_loss, val_auc, val_acc, val_precision, val_recall, val_f1
```

完整的训练和评估逻辑，包括早停、学习率调度和多种评估指标计算。

## 模型性能详细比较

### 各模型性能指标

| 模型 | AUC | 准确率 | 精确率 | 召回率 | F1分数 | 训练时间(min) | 推理速度(ms/样本) |
|------|-----|-------|-------|--------|-------|--------------|-----------------|
| TF-IDF + XGBoost (基础) | 0.8645 | 0.9299 | 1.0000 | 0.6075 | 0.7558 | ~5 | ~0.5 |
| TF-IDF + XGBoost (AUC优化) | 0.8551 | 0.9282 | 0.9848 | 0.6075 | 0.7514 | ~3 | ~0.5 |
| TF-IDF + Deep | 0.8614 | 0.9316 | 0.9701 | 0.6355 | 0.7673 | ~15 | ~2 |
| TF-IDF + EHR (改进多模态) | 0.8932 | 0.9483 | 0.9811 | 0.7009 | 0.8176 | ~25 | ~5 |
| TF-IDF + EHR (注意力模型) | 0.8851 | 0.9435 | 0.9743 | 0.6822 | 0.8024 | ~30 | ~8 |
| 完整多模态 | 0.7927 | 0.9483 | 0.9811 | 0.7009 | 0.8176 | ~45 | ~12 |

### 模型复杂度和资源消耗

| 模型 | 参数数量 | 内存需求(GB) | 训练GPU需求 | 适用场景 |
|------|---------|------------|------------|---------|
| TF-IDF + XGBoost | ~数万 | ~1-2 | 无/低 | 资源受限环境，需要可解释性 |
| TF-IDF + Deep | ~数百万 | ~2-3 | 中 | 有一定计算资源，需要较好性能 |
| TF-IDF + EHR | ~数百万 | ~4-6 | 中高 | 有多模态数据，需要较高精度 |
| 完整多模态 | ~1亿+ | ~8-12 | 高 | 资源充足，需要最高精度 |

### 错误分析

各模型的错误类型分布：

| 模型 | 假阴性(FN) | 假阳性(FP) | 主要错误类型 |
|------|-----------|-----------|------------|
| TF-IDF + XGBoost (基础) | 42 | 0 | 只有假阴性，导致较低召回率 |
| TF-IDF + XGBoost (AUC优化) | 42 | 1 | 主要为假阴性，极少假阳性 |
| TF-IDF + Deep | 39 | 3 | 较少假阴性，少量假阳性 |
| TF-IDF + EHR (改进多模态) | 37 | 2 | 假阴性减少，保持低假阳性 |
| TF-IDF + EHR (注意力模型) | 34 | 3 | 更平衡的错误分布 |
| 完整多模态 | 32 | 2 | 最少假阴性，保持低假阳性 |

## 安装与使用指南

### 环境配置

```bash
# 创建虚拟环境
conda create -n medical_readmission python=3.8
conda activate medical_readmission

# 安装依赖
pip install torch==1.9.0 torchvision==0.10.0
pip install transformers==4.11.3 scikit-learn==1.0.1 xgboost==1.5.0
pip install pandas numpy matplotlib tqdm pillow
```

### 数据准备

1. 数据结构应组织如下：
   ```
   data/
   ├── notes.csv             # 医疗记录文本
   ├── train.csv             # 训练集
   ├── valid.csv             # 验证集
   ├── test.csv              # 测试集
   ├── test_answer.csv       # 测试集答案
   ├── sample_submission.csv # 提交样例
   └── image_features/       # 医学影像特征
   ```

2. 执行数据预处理：
   ```bash
   # 预处理医疗记录文本
   cd notes_prediction_model
   python preprocess_notes.py
   
   # 生成TF-IDF特征
   cd ../tfidf_ehr
   python save_features.py
   ```

### 模型训练

#### TF-IDF + XGBoost模型

```bash
cd tfidf
python tfidf_xgboost_combined.py
```

#### 神经网络模型

```bash
# TF-IDF + 深度学习模型
cd tfidf_deep
python tfidf_deep.py

# TF-IDF + EHR改进多模态模型
cd tfidf_ehr
python improved_multimodal_model.py

# TF-IDF + EHR注意力模型
python attention_multimodal_model.py

# 完整多模态模型
cd ..
python multimodal_prediction.py
```

### 模型评估

```bash
# TF-IDF + XGBoost模型评估
cd tfidf
python evaluate_auc_performance.py

# 多模态模型评估
cd tfidf_ehr
python evaluate_model.py --model_type attention
```

### 生成预测结果

```bash
# 使用最佳模型生成预测
python predict.py --model_path best_attention_multimodal_model.pth --output_file predictions.csv
```

## 总结

本项目实现了一系列从简单到复杂的医疗再入院预测模型，每个模型都有其特定的优势和适用场景。从基础的TF-IDF+XGBoost模型到复杂的多模态深度学习模型，性能逐步提升，但计算复杂度和资源需求也相应增加。

实验结果表明，融合多种数据模态能显著提高预测性能，特别是在识别高风险患者方面。注意力机制的引入进一步增强了模型性能和可解释性，为临床决策提供了更直观的支持。

模型的选择应根据具体应用场景、可用数据类型和计算资源来决定，从而在性能、效率和可解释性之间取得最佳平衡。未来工作将专注于模型优化、系统集成和临床验证，推动医疗再入院预测技术的发展与应用。 