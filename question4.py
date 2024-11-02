import pandas as pd
import torch
import torch.nn as nn
from TorchCRF import CRF
from datasets import load_dataset, Dataset  # 注意这里
from seqeval.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
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
            # 使用CRF层计算损失，将损失取负（CRF层返回的是log-likelihood）,函数返回的是负对数
            # 使用attention_mask来忽略填充标记位置
            loss = -self.crf(emissions, labels, mask=attention_mask.bool())
            return loss  # 返回损失值
        else:
            # 预测时，使用CRF进行解码
            # CRF层的viterbi_decode方法进行解码，返回最可能的标签路径
            # 使用attention_mask忽略填充标记位置
            prediction = self.crf.viterbi_decode(emissions, mask=attention_mask.byte())
            return prediction  # 返回预测的标签序列


# 加载数据集并进行预处理
def load_and_preprocess_data(file_path):
    # 使用 Hugging Face 的 datasets 库加载数据
    dataset = load_dataset('csv', data_files={'train': file_path}, encoding='latin1', on_bad_lines='skip')

    # 填充缺失值
    dataset = dataset['train']
    dataset = dataset.to_pandas()
    # 使用前向填充填补空的'Sentence #'列
    dataset['Sentence #'] = dataset['Sentence #'].ffill()

    # 将非字符串的单词和标签替换为缺失标记'O'
    dataset['Word'] = dataset['Word'].fillna('O').astype(str)
    dataset['Tag'] = dataset['Tag'].fillna('O').astype(str)

    # 按句子分组
    sentences = dataset.groupby('Sentence #')['Word'].apply(list).values
    labels = dataset.groupby('Sentence #')['Tag'].apply(list).values

    # 创建 DataFrame
    df = pd.DataFrame({'words': sentences, 'labels': labels})
    # 测试df是否正确
    print("The sentence in df are:")
    print(df['words'])
    print("The word_labels in df are:")
    print(df['labels'])

    # 划分训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    return train_df, test_df


# 定义分词和标签对齐函数
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['words'],
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding='max_length',
        truncation=True,
        max_length=128
    )

    # 初始化标签列表
    labels = []
    # 获取批次中每个样本的分词结果
    for batch_index in range(len(examples['words'])):
        # 第一步：获取当前句子的词标签
        word_labels = examples['labels'][batch_index]
        # 第二步：获取当前句子的 word_ids
        word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
        previous_word_idx = None
        label_ids = []

        # 第三步：为每个分词后的词片段创建标签
        # 如果标签是 B- 开头，只有第一个词片段保留 B- 标签，其他的词片段改为 I- 标签
        for idx, word_idx in enumerate(word_ids):
            # 对于特殊字符，如 [CLS]、[SEP]，以及填充的词片段，标签设为 "O"
            if word_idx is None:
                label_ids.append(labels_to_ids['O'])
            else:
                word_label = word_labels[word_idx]  # 获取当前词的标签
                if word_idx != previous_word_idx:
                    # 当前词片段是新词的开始，保留原始标签
                    label_ids.append(labels_to_ids[word_label])
                else:
                    # 当前词片段属于同一个词，需要判断标签是否需要转换
                    if word_label.startswith('B-'):
                        # 如果是 B- 标签，后续词片段标签改为 I- 标签
                        new_label = 'I-' + word_label[2:]
                        label_ids.append(labels_to_ids.get(new_label, labels_to_ids['O']))
                    else:
                        label_ids.append(labels_to_ids[word_label])
            previous_word_idx = word_idx  # 更新前一个词索引

        # 将处理好的标签添加到 labels 列表中
        labels.append(label_ids)

    # 处理标签中的 -100 错误（如果有的话）
    for i in range(len(labels)):
        label_ids = labels[i]
        label_ids = [label if label != -100 else labels_to_ids['O'] for label in label_ids]
        labels[i] = label_ids

    # 第四步：将标签添加到 tokenized_inputs 中
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# 定义训练函数，执行一个训练周期的操作。
def train_epoch(model, dataloader, optimizer, device):
    # 设置模型为训练模式
    model.train()
    total_loss = 0

    # 逐批次遍历数据加载器
    for batch in tqdm(dataloader, desc="训练中"):
        optimizer.zero_grad()  # 清零优化器的梯度

        # 将输入数据、注意力掩码和标签移至目标设备 (如 CUDA)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # 执行前向传播，提取输出的损失值
        loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # 计算平均损失，或者可以根据需要选择 loss.sum()
        loss = loss.mean()
        # 累加损失值
        total_loss += loss.item()

        # 反向传播，计算梯度
        loss.backward()

        # 优化器更新模型权重
        optimizer.step()

    # 计算平均损失
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
    # 将模型设置为评估模式
    model.eval()
    eval_preds, eval_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 获取模型预测结果
            predictions = model(input_ids=ids, attention_mask=mask)

            # 将预测结果对齐并转换为标签
            labels = labels.cpu().numpy()  # 将标签张量移至 CPU 并转换为 NumPy 数组，以便后续处理
            mask = mask.cpu().numpy()  # 将注意力掩码张量移至 CPU 并转换为 NumPy 数组

            # 对于在一批中的每个样本
            for i in range(len(predictions)):
                pred = predictions[i]
                label = labels[i]
                mask_i = mask[i]

                # 初始化列表来存储 true labels 和 predictions
                true_labels = []
                pred_labels = []

                # 遍历当前句子的每个token
                for j in range(len(pred)):
                    if mask_i[j]:
                        # 只考虑 attention_mask 为 1 的 token，忽略填充的 token
                        label_id = label[j]
                        if label_id != -100:
                            true_labels.append(ids_to_labels[label_id])
                            pred_labels.append(ids_to_labels[pred[j]])

                # 将当前句子的 true labels 和 predictions 添加到全部的列表中
                eval_labels.append(true_labels)
                eval_preds.append(pred_labels)

    # 返回未展平的标签列表，适用于 seqeval.metrics
    return eval_labels, eval_preds


# 定义BIO违例统计函数
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
    print("========= Question4 =========")
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # 划分训练集和测试集
    # 本地文件路径
    # file_path = 'ner_dataset.csv'
    # fixme 如果在Google Colab中，可以使用以下路径
    file_path = '/content/sample_data/ner_dataset.csv'
    train_df, test_df = load_and_preprocess_data(file_path)

    # fixme 批次大小，在PC上调试设置为36，Google Colab上设置为84
    BATCH_SIZE = 84
    # fixme 训练轮次，调试时可以设置为1
    EPOCHS = 1

    # 创建数据集和数据加载器
    training_set = Dataset.from_pandas(train_df)
    testing_set = Dataset.from_pandas(test_df)
    # 打印数据集大小
    print(f"训练数据集规模: {len(training_set)}")
    print(f"测试数据集规模: {len(testing_set)}")

    # 应用分词和标签对齐（使用更新的函数）
    training_set = training_set.map(tokenize_and_align_labels, batched=True)
    testing_set = testing_set.map(tokenize_and_align_labels, batched=True)

    # 设置格式为 PyTorch 张量
    training_set.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    testing_set.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # 创建数据加载器
    training_loader = DataLoader(training_set, batch_size=BATCH_SIZE, shuffle=True)
    testing_loader = DataLoader(testing_set, batch_size=BATCH_SIZE)

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

    report = classification_report(
        labels_list,
        preds_list,
        zero_division=0
    )
    print("\n========= 分类报告（使用BERT与CRF混合模型） =========")
    print(report)

    # 计算 F1 分数
    f1 = f1_score(labels_list, preds_list)
    print(f"\n========= F1 分数 =========")
    print(f"F1 分数: {f1:.4f}")

    # 计算准确率
    accuracy = accuracy_score(labels_list, preds_list)
    print(f"\n========= 准确率 =========")
    print(f"准确率: {accuracy:.4f}")

    # 统计BIO规则违例
    print("\n========= 统计BIO规则违例 =========")
    violations, total_preds = BIO_violations(preds_list)
    violation_ratio = violations / total_preds
    print(f"BIO规则违例数: {violations}")
    print(f"预测标签总数: {total_preds}")
    print(f"BIO规则违例比例: {violation_ratio:.4f}")
