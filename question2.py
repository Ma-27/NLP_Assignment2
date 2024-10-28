import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
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
        sentence = self.data.sentence[index].strip().split()
        word_labels = self.data.word_labels[index].split(",")

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


# 定义训练函数
def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="训练中"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

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

    labels = [[ids_to_labels[id.item()] for id in labels] for labels in eval_labels]
    predictions = [[ids_to_labels[id.item()] for id in preds] for preds in eval_preds]

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

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


# 主函数
if __name__ == '__main__':
    # 初始化设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # 读取数据,使用实验课的数据集
    data = pd.read_csv('ner_dataset.csv', encoding='latin1')
    # 使用Google Colab
    # data = pd.read_csv('/content/sample_data/ner_dataset.csv', encoding='latin1')

    # data = data.fillna(method='ffill')
    # 使用前向填充填补空的'Sentence #'列
    data['Sentence #'] = data['Sentence #'].ffill()

    # 将非字符串的单词和标签替换为缺失标记'O'
    data['Word'] = data['Word'].fillna('O').astype(str)
    data['Tag'] = data['Tag'].fillna('O').astype(str)

    # 数据预处理
    sentences = data.groupby('Sentence #')['Word'].apply(list).values
    labels = data.groupby('Sentence #')['Tag'].apply(list).values

    # 将句子和标签存入DataFrame
    df = pd.DataFrame({'sentence': [' '.join(s) for s in sentences], 'word_labels': [','.join(l) for l in labels]})

    print(df['sentence'])

    # 划分训练集和测试集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    MAX_LEN = 128
    BATCH_SIZE = 64
    EPOCHS = 3

    # 创建数据集和数据加载器
    training_set = new_dataset(train_df.reset_index(drop=True), tokenizer, MAX_LEN)
    testing_set = new_dataset(test_df.reset_index(drop=True), tokenizer, MAX_LEN)

    train_params = {'batch_size': BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
    test_params = {'batch_size': BATCH_SIZE, 'shuffle': False, 'num_workers': 0}

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    # 初始化模型
    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(labels_to_ids))
    model.to(device)

    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=3e-5)

    # 训练模型
    for epoch in range(EPOCHS):
        print(f"\n========= 训练Epoch {epoch + 1} =========")
        train_epoch(model, training_loader, optimizer, device)

    # 验证模型
    print("\n========= 验证模型 =========")
    labels_list, preds_list = new_valid(model, testing_loader, device)

    # 统计BIO规则违例
    print("\n========= 统计BIO规则违例 =========")
    violations = 0
    total_preds = 0
    previous_label = 'O'

    for i in range(len(labels_list)):
        label = labels_list[i]
        pred = preds_list[i]

        if pred.startswith('I-'):
            if not (previous_label.endswith(pred[2:]) and (
                    previous_label.startswith('B-') or previous_label.startswith('I-'))):
                violations += 1
        previous_label = pred
        total_preds += 1

    violation_ratio = violations / total_preds
    print(f"BIO规则违例数: {violations}")
    print(f"预测标签总数: {total_preds}")
    print(f"BIO规则违例比例: {violation_ratio:.4f}")

    # 保存模型的状态字典
    torch.save(model.state_dict(), 'model_weights.pth')
