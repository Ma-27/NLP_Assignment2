import pandas as pd
import torch
import torch.nn as nn
from TorchCRF import CRF
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW

# 定义标签到ID的映射
labels_to_ids = {
    'O': 0,
    'B-gpe': 1,
    'I-gpe': 2,
    'B-per': 3,
    'I-per': 4,
    'B-org': 5,
    'I-org': 6,
    'B-loc': 7,
    'I-loc': 8,
    "B-tim": 9,
    "I-tim": 10,
    "B-art": 11,
    "I-art": 12,
    "B-eve": 13,
    "I-eve": 14,
    "B-geo": 15,
    "I-geo": 16,
    "B-nat": 17,
    "I-nat": 18,
    # 添加其他标签的编码...
}

# 反映射ID到标签
ids_to_labels = {id: label for label, id in labels_to_ids.items()}


# 定义数据集类
class new_dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        # 获取句子和对应的词标签
        sentence = self.data.sentence[index].split(separator)
        word_labels = self.data.word_labels[index].split(separator)

        # 检查句子和词标签长度是否匹配
        if len(sentence) != len(word_labels):
            print(
                f"Index {index} has mismatched lengths: len(sentence)={len(sentence)}, len(word_labels)={len(word_labels)}")
            return None  # skip this example

        # 使用tokenizer对句子进行编码
        encoding = self.tokenizer(
            sentence,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        labels = []
        word_ids = encoding.word_ids(batch_index=0)
        previous_word_idx = None

        for idx, word_idx in enumerate(word_ids):
            if word_idx is None:
                labels.append(-100)
            else:
                word_label = word_labels[word_idx]
                if word_idx != previous_word_idx:
                    labels.append(labels_to_ids[word_label])
                else:
                    if word_label.startswith('B-'):
                        new_label = 'I-' + word_label[2:]
                        labels.append(labels_to_ids.get(new_label, labels_to_ids['O']))
                    else:
                        labels.append(labels_to_ids[word_label])
            previous_word_idx = word_idx

        # 处理 -100 error
        labels = torch.tensor(labels, dtype=torch.long)
        labels[labels == -100] = 0  # 将 -100 替换为有效标签 0

        # 将所有内容转换为PyTorch张量
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(labels, dtype=torch.long)

        return item

    def __len__(self):
        return self.len


# 定义带有CRF层的模型
class BertCRF(nn.Module):
    def __init__(self, num_labels):
        super(BertCRF, self).__init__()
        self.num_labels = num_labels
        self.bert = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        self.crf = CRF(num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # 获取BERT的输出
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        emissions = self.bert.classifier(sequence_output)  # [batch_size, seq_len, num_labels]

        if labels is not None:
            # 计算CRF损失
            loss = -self.crf(emissions, labels, mask=attention_mask.byte())
            return loss
        else:
            # 使用CRF进行解码
            prediction = self.crf_model.viterbi_decode(emissions, mask=attention_mask.byte())
            return prediction


# 定义训练函数
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="训练中"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # 计算平均损失，或者可以根据需要选择 loss.sum()
        loss = loss.mean()

        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    print(f"平均训练损失: {avg_loss:.4f}")


# 定义BIO规则转移矩阵的函数
def is_valid_transition(label_from, label_to):
    # 如果当前标签为 'O' (Outside)，则下一个标签必须是 'O' 或 'B-' 开头的标签（新的实体的开始）
    if label_from == 'O':
        return label_to == 'O' or label_to.startswith('B-')

    # 如果当前标签以 'B-' 开头（表示实体的开始），则下一个标签有三种可能：
    # 1. 'O'：结束实体并回到 Outside
    # 2. 另一个 'B-'：表示开始一个新的实体
    # 3. 对应的 'I-' 标签：表示继续同一类型的实体
    elif label_from.startswith('B-'):
        tag_from = label_from[2:]  # 获取当前标签的实体类型
        return label_to == 'O' or label_to.startswith('B-') or (label_to.startswith('I-') and label_to[2:] == tag_from)

    # 如果当前标签以 'I-' 开头（表示实体的内部），则下一个标签也有三种可能：
    # 1. 'O'：结束当前实体并回到 Outside
    # 2. 另一个 'B-'：表示开始一个新的实体
    # 3. 对应的 'I-' 标签：继续当前类型的实体
    elif label_from.startswith('I-'):
        tag_from = label_from[2:]  # 获取当前标签的实体类型
        return label_to == 'O' or label_to.startswith('B-') or (label_to.startswith('I-') and label_to[2:] == tag_from)

    # 如果当前标签不属于上述任何一种情况，则认为是非法的
    else:
        return False


# 定义验证函数
def joint_model_valid(model, dataloader, device):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    eval_preds, eval_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 获取模型预测结果
            predictions = model(input_ids=input_ids, attention_mask=attention_mask)
            # 将预测结果转换为张量并对齐
            max_len = input_ids.size(1)
            predictions_padded = [p + [0] * (max_len - len(p)) for p in predictions]
            predictions_tensor = torch.tensor(predictions_padded).to(device)

            # 调整 labels 和 attention_mask 的尺寸
            labels = labels[:, :predictions_tensor.size(1)]
            attention_mask = attention_mask[:, :predictions_tensor.size(1)]

            # 计算准确率
            flattened_targets = labels.view(-1)
            flattened_predictions = predictions_tensor.view(-1)

            active_accuracy = attention_mask.view(-1) != 0

            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            eval_labels.append(labels)
            eval_preds.append(predictions)

            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    eval_accuracy = eval_accuracy / len(dataloader)
    print(f"验证准确率: {eval_accuracy:.4f}")

    # 转换ID到标签
    labels = [
        [ids_to_labels[id.item()] for id in labels]
        for labels in eval_labels
    ]
    predictions = [
        [ids_to_labels[id.item()] for id in preds]
        for preds in eval_preds
    ]

    return labels, predictions


# 定义BIO违例统计函数
def BIO_violations(predictions):
    violations = 0
    total_preds = 0

    for pred_sequence in predictions:
        previous_label = 'O'
        for label in pred_sequence:
            if label.startswith('I-'):
                if not (previous_label.endswith(label[2:]) and (
                        previous_label.startswith('B-') or previous_label.startswith('I-'))):
                    violations += 1
            previous_label = label
            total_preds += 1

    return violations, total_preds


# 自定义的collate函数，跳过返回None的样本
def collate_fn(batch):
    # 过滤掉返回 None 的样本
    batch = [item for item in batch if item is not None]
    # 使用默认的 collate 函数将剩余的样本组合成一个批量
    return default_collate(batch)


# 主函数
if __name__ == "__main__":
    print("========= Question4 =========")
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # 读取数据，使用实验课的数据集
    # data = pd.read_csv('ner_dataset.csv', encoding='latin1', on_bad_lines='skip')
    # fixme 如果在Google Colab中，可以使用以下路径
    data = pd.read_csv('/content/sample_data/ner_dataset.csv', encoding='latin1', on_bad_lines='skip')

    # 使用前向填充填补空的'Sentence #'列
    data['Sentence #'] = data['Sentence #'].ffill()

    # 将非字符串的单词和标签替换为缺失标记'O'
    data['Word'] = data['Word'].fillna('O').astype(str)
    data['Tag'] = data['Tag'].fillna('O').astype(str)

    # 数据预处理
    sentences = data.groupby('Sentence #')['Word'].apply(list).values
    labels = data.groupby('Sentence #')['Tag'].apply(list).values

    # 将句子和标签存入DataFrame todo: separator = '|||'
    separator = '|||'
    df = pd.DataFrame({
        'sentence': [separator.join(s) for s in sentences],
        'word_labels': [separator.join(l) for l in labels]
    })
    # 测试df是否正确
    print("The sentence in df are:")
    print(df['sentence'])
    print("The word_labels in df are:")
    print(df['word_labels'])

    # 划分训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    MAX_LEN = 128
    # fixme 批次大小，在PC上调试设置为36，Google Colab上设置为64
    BATCH_SIZE = 64
    # fixme 训练轮次，调试时可以设置为1
    EPOCHS = 1

    # 创建数据集和数据加载器
    training_set = new_dataset(train_df.reset_index(drop=True), tokenizer, MAX_LEN)
    testing_set = new_dataset(test_df.reset_index(drop=True), tokenizer, MAX_LEN)
    # 打印数据集大小
    print(f"训练数据集规模: {len(training_set)}")
    print(f"测试数据集规模: {len(testing_set)}")

    train_params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 0, 'collate_fn': collate_fn}
    test_params = {'batch_size': BATCH_SIZE, 'shuffle': False, 'num_workers': 0, 'collate_fn': collate_fn}
    print(f"训练集加载器参数: {train_params}")
    print(f"测试集加载器参数: {test_params}")

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    # 打印第一个训练批次的键和形状
    first_train_batch = next(iter(training_loader))
    print(f"第一个训练批次的Key: {first_train_batch.keys()}")
    print(f"第一个训练批次 'input_ids' shape: {first_train_batch['input_ids'].shape}")
    print(f"第一个训练批次 batch 'labels' shape: {first_train_batch['labels'].shape}")

    # 初始化带有CRF层的模型
    num_labels = len(labels_to_ids)
    model = BertCRF(num_labels=num_labels)
    model.to(device)
    # 打印模型初始化信息
    print(f"模型初始化，使用 {num_labels} 个标签.")
    print(f"模型使用  {device} 设备.")

    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=3e-5)

    # 训练模型
    for epoch in range(EPOCHS):
        print(f"\n========= 训练Epoch {epoch + 1} =========")
        train_epoch(model, training_loader, optimizer, device)

    # 保存模型的状态字典
    torch.save(model.state_dict(), 'joint_model_weights.pth')
    print("包含CRF层的模型权重已保存")

    # 验证模型
    print("\n========= 验证模型 =========")
    labels_list, preds_list = joint_model_valid(model, testing_loader, device)

    # 生成分类报告
    flattened_true_labels = [label for sublist in labels_list for label in sublist]
    flattened_predictions = [pred for sublist in preds_list for pred in sublist]

    # 获取数据中的唯一标签
    unique_labels = sorted(set(flattened_true_labels + flattened_predictions))
    # 打印数据中唯一标签的数量
    print(f"数据中的唯一标签数量: {len(unique_labels)}")

    report = classification_report(
        flattened_true_labels,
        flattened_predictions,
        labels=unique_labels,
        target_names=unique_labels,
        zero_division=0
    )
    print("\n========= 分类报告（使用BERT与CRF混合模型） =========")
    print(report)

    # 统计BIO规则违例
    print("\n========= 统计BIO规则违例 =========")
    violations, total_preds = BIO_violations(preds_list)
    violation_ratio = violations / total_preds
    print(f"BIO规则违例数: {violations}")
    print(f"预测标签总数: {total_preds}")
    print(f"BIO规则违例比例: {violation_ratio:.4f}")