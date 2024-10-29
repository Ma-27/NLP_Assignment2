import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from transformers import BertTokenizerFast, BertForTokenClassification

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
    'B-tim': 9,
    'I-tim': 10,
    'B-art': 11,
    'I-art': 12,
    'B-eve': 13,
    'I-eve': 14,
    'B-geo': 15,
    'I-geo': 16,
    'B-nat': 17,
    'I-nat': 18,
    # 添加其他标签的编码...
}

# 反映射ID到标签
ids_to_labels = {id: label for label, id in labels_to_ids.items()}


# 定义数据集类
class NERDataset(Dataset):
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
            print(f"索引 {index} 的句子和标签长度不匹配")
            return None  # 跳过该样本

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

        # 将所有内容转换为PyTorch张量
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(labels, dtype=torch.long)

        return item

    def __len__(self):
        return self.len


# 定义验证函数
def validate(model, dataloader, device):
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    eval_preds, eval_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            # 获取输入数据并移动到设备上
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs.loss, outputs.logits
            eval_loss += loss.item()

            # 计算预测结果
            active_logits = logits.view(-1, model.num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)

            # 只考虑非填充部分的标签
            labels_flat = labels.view(-1)
            active_labels = labels_flat != -100
            labels_flat = labels_flat[active_labels]
            predictions = flattened_predictions[active_labels]

            eval_labels.extend(labels_flat.cpu().numpy())
            eval_preds.extend(predictions.cpu().numpy())

    avg_loss = eval_loss / len(dataloader)
    print(f"验证损失: {avg_loss}")

    # 将ID转换为标签
    eval_labels = [ids_to_labels[id] for id in eval_labels]
    eval_preds = [ids_to_labels[id] for id in eval_preds]

    return eval_labels, eval_preds


# 定义BIO规则违例统计函数
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


# 自定义collate函数，跳过返回None的样本
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    return default_collate(batch)


# 主函数
if __name__ == '__main__':
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # 读取数据集
    data = pd.read_csv('ner_dataset.csv', encoding='latin1', on_bad_lines='skip')

    # 数据预处理
    data['Sentence #'] = data['Sentence #'].ffill()
    data['Word'] = data['Word'].fillna('O').astype(str)
    data['Tag'] = data['Tag'].fillna('O').astype(str)

    sentences = data.groupby('Sentence #')['Word'].apply(list).values
    labels = data.groupby('Sentence #')['Tag'].apply(list).values

    # 将句子和标签存入DataFrame
    separator = '|||'
    df = pd.DataFrame({
        'sentence': [separator.join(s) for s in sentences],
        'word_labels': [separator.join(l) for l in labels]
    })

    # 划分训练集和测试集
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)

    MAX_LEN = 128
    BATCH_SIZE = 32

    # 创建测试集和数据加载器
    test_dataset = NERDataset(test_df.reset_index(drop=True), tokenizer, MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 加载模型并加载保存的权重
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(labels_to_ids))
    model.load_state_dict(torch.load('model_weights.pth', map_location=device))
    model.to(device)

    # 验证模型
    print("\n========= 验证模型 =========")
    true_labels, predictions = validate(model, test_loader, device)

    # 生成分类报告
    print("\n========= 分类报告 =========")
    print(classification_report(true_labels, predictions, labels=list(labels_to_ids.values()), zero_division=0))

    # 统计BIO规则违例
    print("\n========= 统计BIO规则违例 =========")
    # 将预测结果分组
    grouped_preds = []
    idx = 0
    for seq in test_dataset:
        seq_len = len(seq['labels'][seq['labels'] != -100])
        pred_seq = predictions[idx:idx + seq_len]
        grouped_preds.append(pred_seq)
        idx += seq_len

    violations, total_preds = BIO_violations(grouped_preds)
    violation_ratio = violations / total_preds if total_preds > 0 else 0
    print(f"BIO规则违例数: {violations}")
    print(f"预测标签总数: {total_preds}")
    print(f"BIO规则违例比例: {violation_ratio:.4f}")
