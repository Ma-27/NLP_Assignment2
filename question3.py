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


# 重写验证函数，按照原题目要求
def crf_valid(model, crf_model, testing_loader):
    # 将模型设置为评估模式
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            ids = batch['input_ids'].to(device, dtype=torch.long)
            mask = batch['attention_mask'].to(device, dtype=torch.bool)  # 注意这里修改使用 bool 类型，防止-100撞上
            labels = batch['labels'].to(device, dtype=torch.long)

            # 获取Bert模型的输出
            outputs = model(input_ids=ids, attention_mask=mask)

            # 获取发射分数
            emissions = outputs.logits  # (batch_size, seq_len, num_labels)

            # 使用CRF模型获取预测和损失
            # outpus[1]  use crf_model to get predictions
            # 在CRF解码时，直接使用有效的attention_mask
            predictions = crf_model.decode(emissions, mask=mask)

            # outpus[0] should also come from crf_model
            loss = -crf_model(emissions, labels, mask=mask, reduction='mean')

            eval_loss += loss.item()
            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)

            if idx % 100 == 0:
                loss_step = eval_loss / nb_eval_steps
                print(f"每100步的验证损失: {loss_step}")

            # 将预测结果转换为张量并对齐
            predictions = [torch.tensor(p, device=device) for p in predictions]
            predictions = torch.nn.utils.rnn.pad_sequence(predictions, batch_first=True, padding_value=-100)

            # 调整 labels 和 mask 的尺寸，使其与 predictions 匹配
            labels = labels[:, :predictions.size(1)]
            mask = mask[:, :predictions.size(1)]

            # 计算准确率
            flattened_targets = labels.reshape(-1)  # shape (batch_size * seq_len,)
            flattened_predictions = predictions.view(-1)  # shape (batch_size * seq_len,)

            # 只计算活跃标签的准确率
            active_accuracy = mask.reshape(-1) != 0  # shape (batch_size, seq_len)

            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            eval_labels.append(labels)
            eval_preds.append(predictions)

            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    labels = [[ids_to_labels[id.item()] for id in labels] for labels in eval_labels]
    predictions = [[ids_to_labels[id.item()] for id in preds] for preds in eval_preds]

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"验证损失: {eval_loss}")
    print(f"验证准确率: {eval_accuracy}")

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
    print("========= Question3 =========")
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # 读取数据，使用实验课的数据集
    # data = pd.read_csv('ner_dataset.csv', encoding='latin1', on_bad_lines='skip')
    # 真正在Google Colab中运行使用以下路径 todo 修改路径
    data = pd.read_csv('/content/sample_data/ner_dataset.csv', encoding='latin1', on_bad_lines='skip')

    # 使用前向填充填补空的'Sentence #'列
    data['Sentence #'] = data['Sentence #'].ffill()

    # 将非字符串的单词和标签替换为缺失标记'O'
    data['Word'] = data['Word'].fillna('O').astype(str)
    data['Tag'] = data['Tag'].fillna('O').astype(str)

    # 数据预处理
    sentences = data.groupby('Sentence #')['Word'].apply(list).values
    labels = data.groupby('Sentence #')['Tag'].apply(list).values

    # 将句子和标签存入DataFrame，fixme: notice separator = '|||'
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
    # todo 修改BATCH_SIZE,如果在PC上跑，该数字为24，如果在Google Colab上跑，该数字为96
    BATCH_SIZE = 96

    # 创建数据集和数据加载器
    training_set = new_dataset(train_df.reset_index(drop=True), tokenizer, MAX_LEN)
    testing_set = new_dataset(test_df.reset_index(drop=True), tokenizer, MAX_LEN)

    train_params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 0, 'collate_fn': collate_fn}
    test_params = {'batch_size': BATCH_SIZE, 'shuffle': False, 'num_workers': 0, 'collate_fn': collate_fn}

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    # 初始化Bert模型
    num_labels = len(labels_to_ids)
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    model.to(device)

    # 加载已训练的Bert模型权重 todo 如在Google Colab上运行，需要修改路径
    # saved_state_dict = torch.load('model_weights.pth', map_location=device)
    saved_state_dict = torch.load('/content/model_weights.pth', map_location=device)

    model.load_state_dict(saved_state_dict)
    print("已成功加载训练好的Bert模型权重")

    # 初始化CRF层
    crf_model = CRF(num_labels)
    crf_model.to(device)

    # 创建BIO标签列表
    labels_list = [ids_to_labels[i] for i in range(len(ids_to_labels))]

    # 设置CRF层的转移矩阵
    num_labels = len(labels_list)
    transition_matrix = torch.zeros(num_labels, num_labels)

    for i, label_from in enumerate(labels_list):
        for j, label_to in enumerate(labels_list):
            if not is_valid_transition(label_from, label_to):
                transition_matrix[i][j] = 1  # 非法转移赋予小正值
            else:
                transition_matrix[i][j] = 96.0  # 合法转移得分为略大的正值

    crf_model.transitions = nn.Parameter(transition_matrix.to(device))
    print("已设置CRF模型的转移矩阵以消除BIO违例")

    # 验证模型
    print("\n========= 验证模型 =========")
    labels_list, preds_list = crf_valid(model, crf_model, testing_loader)

    # 统计BIO规则违例
    print("\n========= 统计BIO规则违例 =========")
    violations, total_preds = BIO_violations(preds_list)
    violation_ratio = violations / total_preds
    print(f"BIO规则违例数: {violations}")
    print(f"预测标签总数: {total_preds}")
    print(f"BIO规则违例比例: {violation_ratio:.4f}")

    # 生成分类报告
    flattened_true_labels = [label for sublist in labels_list for label in sublist]
    flattened_predictions = [pred for sublist in preds_list for pred in sublist]
    unique_labels = sorted(set(flattened_true_labels + flattened_predictions))

    report = classification_report(
        flattened_true_labels,
        flattened_predictions,
        labels=unique_labels,
        target_names=unique_labels,
        zero_division=0
    )
    print("\n========= 分类报告（使用CRF模型） =========")
    print(report)
