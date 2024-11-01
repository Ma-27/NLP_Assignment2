import pandas as pd
import torch
import torch.nn as nn
from TorchCRF import CRF
from sklearn.metrics import accuracy_score, classification_report
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
        # 第一步：获取句子和对应的词标签
        sentence = self.data.sentence[index].split(separator)
        word_labels = self.data.word_labels[index].split(separator)

        # 检查句子和词标签长度是否匹配
        if len(sentence) != len(word_labels):
            print(
                f"Index {index} has mismatched lengths: len(sentence)={len(sentence)}, len(word_labels)={len(word_labels)}")
            return None  # 跳过此样本

        # 第二步：使用tokenizer对句子进行编码，包括填充和截断到最大长度
        encoding = self.tokenizer(
            sentence,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        # 第三步：为每个分词后的词片段创建标签
        labels = []
        word_ids = encoding.word_ids(batch_index=0)
        previous_word_idx = None

        for idx, word_idx in enumerate(word_ids):
            # 对于特殊字符，如[CLS]和[SEP]，以及填充的词片段，标签设为"O"
            if word_idx is None:
                labels.append(labels_to_ids['O'])
            else:
                word_label = word_labels[word_idx]  # 获取当前词的标签
                if word_idx != previous_word_idx:
                    # 当前词片段是新词的开始，保留B-标签
                    labels.append(labels_to_ids[word_label])
                else:
                    # 如果当前词片段属于同一个词，需要判断标签是否需要转换
                    if word_label.startswith('B-'):
                        # 如果是B-标签，后续词片段标签改为I-标签
                        new_label = 'I-' + word_label[2:]
                        labels.append(labels_to_ids.get(new_label, labels_to_ids['O']))
                    else:
                        labels.append(labels_to_ids[word_label])
            previous_word_idx = word_idx  # 更新前一个词索引

        # 处理 -100 error
        labels = torch.tensor(labels, dtype=torch.long)
        labels[labels == -100] = 0  # 将 -100 替换为有效标签 0

        # 第四步：将所有内容转换为PyTorch张量
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = labels

        return item

    def __len__(self):
        return self.len


# 定义带有CRF层的BERT模型
class BertCRF(nn.Module):
    def __init__(self, num_labels):
        # 初始化BertCRF模型，包括BERT编码层和CRF层
        super(BertCRF, self).__init__()
        # 标签数量
        self.num_labels = num_labels
        # 加载预训练的BERT模型用于Token分类任务，并设置分类标签数量
        self.bert = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
        # 初始化CRF层，标签数量与模型的分类标签数量一致
        self.crf = CRF(num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # 获取BERT的输出
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask)
        # 提取最后一层的隐层输出 [batch_size, seq_len, hidden_size]
        sequence_output = outputs.last_hidden_state
        # 将隐层输出经过分类层转换为标记的logits分数 [batch_size, seq_len, num_labels]
        emissions = self.bert.classifier(sequence_output)

        # 如果提供labels参数，则模型会计算损失；否则进入预测模式。
        if labels is not None:
            # 训练中，计算CRF损失
            # 使用CRF层计算损失，将损失取负（CRF层返回的是log-likelihood），函数返回的是负对数似然
            # 使用attention_mask来忽略填充标记位置
            loss = -self.crf(emissions, labels, mask=attention_mask.bool())
            return loss  # 返回损失值
        else:
            # 预测时，使用CRF进行解码
            # CRF层的decode方法进行解码，返回最可能的标签路径
            # 使用attention_mask忽略填充标记位置
            prediction = self.crf.viterbi_decode(emissions, mask=attention_mask.bool())
            return prediction  # 返回预测的标签序列


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
    # 将模型设置为评估模式
    model.eval()

    eval_accuracy = 0
    eval_preds, eval_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
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

            # 只计算活跃标签的准确率
            active_accuracy = attention_mask.view(-1) != 0

            # 只选择有效的标签进行计算
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            # 将当前批次的标签和预测结果分别添加到列表中
            eval_labels.append(labels)
            eval_preds.append(predictions)

            # 计算当前批次的准确率
            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    # 计算平均准确率
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


# 统计BIO违例的函数
def BIO_violations(predictions):
    violations = 0
    total_preds = 0

    # 遍历每个预测序列
    for pred_sequence in predictions:
        previous_label = 'O'  # 初始上一个标签设为 'O'

        # 遍历当前序列的每个标签
        for label in pred_sequence:
            # 检查从 previous_label 到当前 label 的转移是否符合BIO规则
            if not is_valid_transition(previous_label, label):
                violations += 1  # 若违反规则，违例计数增加

            # 更新上一个标签为当前标签
            previous_label = label
            # 总预测标签数量增加
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
    print("========= Question4 -使用预训练模型验证=========")
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
    _, test_df = train_test_split(df, test_size=0.2, random_state=42)

    MAX_LEN = 128
    # fixme 批次大小，在PC上调试设置为36，Google Colab上设置为84
    BATCH_SIZE = 84

    # 创建测试数据集和数据加载器
    testing_set = new_dataset(test_df.reset_index(drop=True), tokenizer, MAX_LEN)

    test_params = {'batch_size': BATCH_SIZE, 'shuffle': False, 'num_workers': 0, 'collate_fn': collate_fn}

    testing_loader = DataLoader(testing_set, **test_params)

    # 初始化模型
    num_labels = len(labels_to_ids)
    model = BertCRF(num_labels=num_labels)
    model.to(device)
    print(f"模型初始化，使用 {num_labels} 个标签.")
    print(f"模型使用 {device} 设备.")

    # 加载模型的状态字典
    # model.load_state_dict(torch.load('joint_model_weights.pth', map_location=device))
    # fixme 使用Google Colab
    model.load_state_dict(torch.load('/content/joint_model_weights.pth', map_location=device))
    print("包含CRF层的模型权重已成功加载")

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
