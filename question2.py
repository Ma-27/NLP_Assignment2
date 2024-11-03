import pandas as pd
import torch
# 注意这里
from seqeval.metrics import classification_report, f1_score, accuracy_score
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
            return None  # skip this example

        # 第二步：使用tokenizer对句子进行编码，包括填充和截断到最大长度
        # BertTokenizerFast提供了方便的“return_offsets_mapping”功能
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
        # 如果标签是B-开头，只有第一个词片段保留B-标签，其他的词片段改为I-标签
        # old labels = [labels_to_ids[label] for label in word_labels]
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

        # 第四步：将所有内容转换为PyTorch张量
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(labels, dtype=torch.long)

        return item

    def __len__(self):
        return self.len


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

        # 执行模型的前向传播操作
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        # 提取输出的损失值
        loss = outputs.loss
        total_loss += loss.item()  # 累加损失值

        # 反向传播，计算梯度
        loss.backward()

        # 优化器更新模型权重
        optimizer.step()

    # 计算平均损失
    avg_loss = total_loss / len(dataloader)
    print(f"平均训练损失: {avg_loss}")


# 定义验证函数
# Active Accuracy can no longer be based on label != -100, we use attention_mask. No need to fix anything here.
def new_valid(model, testing_loader):
    # put model in evaluation mode
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):

            ids = batch['input_ids'].to(device, dtype=torch.long)
            mask = batch['attention_mask'].to(device, dtype=torch.long)
            labels = batch['labels'].to(device, dtype=torch.long)

            outpus = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = outpus[0]
            eval_logits = outpus[1]
            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)

            if idx % 100 == 0:
                loss_step = eval_loss / nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")

            # compute evaluation accuracy
            flattened_targets = labels.view(-1)  # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels)  # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)  # shape (batch_size * seq_len,)

            # only compute accuracy at active labels
            active_accuracy = mask.view(-1) != 0  # shape (batch_size, seq_len)

            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            eval_labels.append(labels)
            eval_preds.append(predictions)

            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    labels = [
        [ids_to_labels[id.item()] if id.item() != -100 else 'O' for id in labels]
        for labels in eval_labels
    ]
    predictions = [
        [ids_to_labels[id.item()] if id.item() != -100 else 'O' for id in preds]
        for preds in eval_preds
    ]

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions


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


# 定义自定义的collate函数,跳过返回None的样本
def collate_fn(batch):
    # 过滤掉返回 None 的样本
    batch = [item for item in batch if item is not None]
    # 使用默认的 collate 函数将剩余的样本组合成一个批量
    return default_collate(batch)


# 主函数
if __name__ == '__main__':
    print("========= Question2 =========")

    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # 读取数据,使用实验课的数据集
    # data = pd.read_csv('ner_dataset.csv', encoding='latin1', on_bad_lines='skip')
    # 使用Google Colab
    data = pd.read_csv('/content/sample_data/ner_dataset.csv', encoding='latin1', on_bad_lines='skip')

    # data = data.fillna(method='ffill')
    # 使用前向填充填补空的'Sentence #'列
    data['Sentence #'] = data['Sentence #'].ffill()

    # 将非字符串的单词和标签替换为缺失标记'O'
    data['Word'] = data['Word'].fillna('O').astype(str)
    data['Tag'] = data['Tag'].fillna('O').astype(str)

    # 数据预处理
    sentences = data.groupby('Sentence #')['Word'].apply(list).values
    labels = data.groupby('Sentence #')['Tag'].apply(list).values

    # 将句子和标签存入DataFrame fixme: separator = '|||'
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
    # fixme 批次大小，在PC上调试设置为36，Google Colab上设置为192
    BATCH_SIZE = 192
    # fixme 训练轮次，调试时可以设置为1
    EPOCHS = 5

    # 创建数据集和数据加载器
    training_set = new_dataset(train_df.reset_index(drop=True), tokenizer, MAX_LEN)
    testing_set = new_dataset(test_df.reset_index(drop=True), tokenizer, MAX_LEN)
    # 打印数据集大小
    print(f"训练数据集规模: {len(training_set)}")
    print(f"测试数据集规模: {len(testing_set)}")

    # 创建数据加载器
    train_params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 0, 'collate_fn': collate_fn}
    test_params = {'batch_size': BATCH_SIZE, 'shuffle': False, 'num_workers': 0, 'collate_fn': collate_fn}
    print(f"训练集加载器参数: {train_params}")
    print(f"测试集加载器参数: {test_params}")

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    # 打印第一个训练批次的键和形状
    first_train_batch = next(iter(training_loader))
    print(f"First training batch keys: {first_train_batch.keys()}")
    print(f"First training batch 'input_ids' shape: {first_train_batch['input_ids'].shape}")
    print(f"First training batch 'labels' shape: {first_train_batch['labels'].shape}")

    # 初始化模型
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(labels_to_ids))
    model.to(device)
    # 打印模型初始化信息
    print(f"Model initialized with {len(labels_to_ids)} labels.")
    print(f"Model is using device: {device}")

    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=3e-5)

    # 训练模型
    for epoch in range(EPOCHS):
        print(f"\n========= 训练Epoch {epoch + 1} =========")
        train_epoch(model, training_loader, optimizer, device)

    # 保存模型的状态字典
    torch.save(model.state_dict(), 'model_weights.pth')
    print("模型权重已保存")

    # 验证模型
    print("\n========= 验证模型 =========")
    labels_list, preds_list = new_valid(model, testing_loader)

    # 获取数据中的唯一标签
    unique_labels = sorted({label for sequence in labels_list for label in sequence} |
                           {label for sequence in preds_list for label in sequence})

    # 打印数据中唯一标签的数量
    print(f"数据中的唯一标签数量: {len(unique_labels)}")

    # 打印分类报告
    report = classification_report(
        labels_list,
        preds_list,
        zero_division=0
    )
    print("\n========= 分类报告 =========")
    print(report)

    # 计算并打印F1分数和准确率
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
