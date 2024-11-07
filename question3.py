import pandas as pd
import torch
from datasets import load_dataset, Dataset  # 注意这里
from seqeval.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
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


# 加载数据集并进行预处理（使用datasets库）
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


# 定义自定义的解码函数，强制BIO约束
def viterbi_decode_with_constraints(emissions, mask):
    # 获取批次大小、序列长度和标签数
    batch_size, seq_length, num_tags = emissions.size()

    # 初始化 scores 张量，用于存储从起始位置到各标签的得分
    # 初始得分设为极小值，只有 'O' 标签的得分设为 0，其他标签初始得分为极低值，确保解码从 'O' 开始
    scores = emissions.new_full((batch_size, num_tags), -10000.0)
    scores[:, labels_to_ids['O']] = 0.0  # 只有 'O' 标签初始得分不为极小值

    # pointers 张量用于记录回溯路径，形状为 [batch_size, seq_length, num_tags]
    pointers = emissions.new_zeros((batch_size, seq_length, num_tags), dtype=torch.long)

    # 创建转移分数矩阵，大小为 [num_tags, num_tags]，用于存储合法和非法转移得分
    # 初始值为 0 表示合法转移，非法转移的得分将设为极低值
    transition_scores = emissions.new_full((num_tags, num_tags), 0.0)
    for i in range(num_tags):
        for j in range(num_tags):
            label_from = ids_to_labels[i]
            label_to = ids_to_labels[j]
            if not is_valid_transition(label_from, label_to):  # 检查BIO规则是否允许该转移
                transition_scores[i, j] = -10000.0  # 非法转移的分数设为极低值

    # 逐时间步计算每个标签的最优得分和路径
    for t in range(seq_length):
        # 获取当前时间步的掩码，形状为 [batch_size, 1]，用于忽略填充位置
        mask_t = mask[:, t].unsqueeze(1)

        # 获取当前时间步的发射分数，形状为 [batch_size, num_tags]
        emit_scores = emissions[:, t]

        # 扩展 scores 和 emit_scores 的维度，以便在计算时匹配 [batch_size, num_tags, num_tags]
        scores_expanded = scores.unsqueeze(2)  # 形状变为 [batch_size, num_tags, 1]
        emit_scores_expanded = emit_scores.unsqueeze(1)  # 形状变为 [batch_size, 1, num_tags]

        # 计算总得分：前一时间步的得分 + 转移得分 + 当前时间步的发射得分
        # transition_scores 添加到 scores_expanded 和 emit_scores_expanded 之和上，形状为 [batch_size, num_tags, num_tags]
        scores_t = scores_expanded + transition_scores.unsqueeze(0) + emit_scores_expanded

        # 在前一时刻的标签上取得分最高的路径，得到最佳得分 scores_t 和对应的前一标签 indices
        # scores_t: [batch_size, num_tags], indices: [batch_size, num_tags]
        scores_t, indices = scores_t.max(1)

        # 更新 scores，考虑 mask，如果 mask 为 1，则用新 scores，否则保持旧 scores
        scores = scores_t * mask_t + scores * (1 - mask_t)

        # 记录路径，便于后续回溯找到最佳标签序列
        pointers[:, t] = indices

    # 回溯找到最优路径
    # seq_ends 表示每个序列的有效长度减1（因为序列从 0 开始计数）
    seq_ends = mask.long().sum(1) - 1  # 形状为 [batch_size]
    best_tags_list = []

    # 对每个序列逐个回溯，找到最优路径
    for idx in range(batch_size):
        # 从最后一个时间步开始，找到最佳的最后一个标签
        seq_end = seq_ends[idx]
        best_last_tag = scores[idx].argmax().item()  # 获取得分最高的标签索引
        best_tags = [best_last_tag]

        # 逐步回溯找到最优路径，直到序列开头
        for back_t in range(seq_end, 0, -1):
            best_last_tag = pointers[idx, back_t, best_last_tag]
            best_tags.append(best_last_tag.item())

        # 由于回溯过程从后向前，最终路径需要反转
        best_tags.reverse()
        best_tags_list.append(best_tags)  # 将路径添加到结果列表中

    return best_tags_list  # 返回包含所有序列最优标签路径的列表


# 定义验证函数，使用自定义的解码函数
def crf_valid(model, testing_loader):
    # 将模型设置为评估模式
    model.eval()

    eval_preds, eval_labels = [], []
    eval_loss = 0.0
    nb_eval_steps = 0

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 获取Bert模型的输出，添加 labels 参数以计算损失
            outputs = model(input_ids=ids, attention_mask=mask, labels=labels)

            # 获取发射分数和损失
            emissions = outputs.logits  # (batch_size, seq_len, num_labels)
            loss = outputs.loss

            # 累加损失和步数
            eval_loss += loss.item()
            nb_eval_steps += 1

            # 使用自定义解码函数获取预测结果
            predictions = viterbi_decode_with_constraints(emissions, mask)

            # 将预测结果对齐并转换为标签
            labels = labels.cpu().numpy()
            mask = mask.cpu().numpy()

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
                        pred_id = pred[j]
                        if label_id != -100:
                            true_labels.append(ids_to_labels[label_id])
                            pred_labels.append(ids_to_labels[pred_id])

                # 将当前句子的 true labels 和 predictions 添加到全部的列表中
                eval_labels.append(true_labels)
                eval_preds.append(pred_labels)

    # 计算平均损失
    eval_loss = eval_loss / nb_eval_steps

    # 计算准确率
    eval_accuracy = accuracy_score(eval_labels, eval_preds)

    # 打印验证损失和验证准确率
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    # 返回未展平的标签列表，适用于 seqeval.metrics
    return eval_labels, eval_preds


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
    file_path = '/content/sample_data/ner_dataset.csv'
    # file_path = 'ner_dataset.csv'  # 根据实际路径修改

    # 加载并预处理数据
    train_df, test_df = load_and_preprocess_data(file_path)

    # 创建数据集和数据加载器
    testing_set = Dataset.from_pandas(test_df)
    # 打印数据集大小
    print(f"测试数据集规模: {len(testing_set)}")

    # 应用分词和标签对齐（使用更新的函数）
    testing_set = testing_set.map(tokenize_and_align_labels, batched=True)

    # 设置格式为 PyTorch 张量
    testing_set.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # todo 修改BATCH_SIZE,如果在PC上跑，该数字为24，如果在Google Colab上跑，该数字为96
    BATCH_SIZE = 96  # 根据实际情况调整

    # 创建数据加载器
    testing_loader = DataLoader(testing_set, batch_size=BATCH_SIZE)

    # 初始化Bert模型
    num_labels = len(labels_to_ids)
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)
    model.to(device)

    # 加载已训练的Bert模型权重 todo 如在Google Colab上运行，需要修改路径
    # saved_state_dict = torch.load('model_weights.pth', map_location=device)
    saved_state_dict = torch.load('/content/model_weights.pth', map_location=device)
    model.load_state_dict(saved_state_dict)
    print("已成功加载训练好的Bert模型权重")

    # 创建BIO标签列表
    labels_list = [ids_to_labels[i] for i in range(len(ids_to_labels))]

    # 验证模型
    print("\n========= 验证模型 =========")
    labels_list_eval, preds_list = crf_valid(model, testing_loader)

    # 生成分类报告
    report = classification_report(
        labels_list_eval,
        preds_list,
        zero_division=0
    )
    print("\n========= 分类报告（使用CRF模型） =========")
    print(report)

    # 计算 F1 分数
    f1 = f1_score(labels_list_eval, preds_list)
    print(f"\n========= F1 分数 =========")
    print(f"F1 分数: {f1:.4f}")

    # 计算准确率
    accuracy = accuracy_score(labels_list_eval, preds_list)
    print(f"\n========= 准确率 =========")
    print(f"准确率: {accuracy:.4f}")

    # 统计BIO规则违例
    print("\n========= 统计BIO规则违例 =========")
    violations, total_preds = BIO_violations(preds_list)
    violation_ratio = violations / total_preds
    print(f"BIO规则违例数: {violations}")
    print(f"预测标签总数: {total_preds}")
    print(f"BIO规则违例比例: {violation_ratio:.4f}")
