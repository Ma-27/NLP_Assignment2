import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

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
class new_dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        # 定义分隔符，与主函数中的分隔符保持一致
        separator = '|||'

        # 第一步：获取句子和对应的词标签
        sentence = self.data.sentence[index].split(separator)
        word_labels = self.data.word_labels[index].split(separator)

        # 检查句子和词标签长度是否匹配
        if len(sentence) != len(word_labels):
            print(
                f"Index {index} has mismatched lengths: len(sentence)={len(sentence)}, len(word_labels)={len(word_labels)}")
            return None  # 跳过这个样本

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
        word_ids = encoding.word_ids(batch_index=0)  # 获取每个词片段对应的词索引
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

        # set labels according to offset_mapping
        # 由于我们已经使用了word_ids，这部分代码可以忽略或删除
        # i = 0
        # for idx, mapping in enumerate(encoding["offset_mapping"]):
        #     if mapping[0] == 0 and mapping[1] != 0:
        #         # overwrite label
        #         encoded_labels[idx] = labels[i]
        #         i += 1

        # 确保标签长度与输入ID长度一致，填充标签到max_len长度
        labels += [labels_to_ids['O']] * (self.max_len - len(labels))  # 填充标签到max_len长度

        # 第四步：将所有内容转换为PyTorch张量
        # 创建一个长度为max_len的标签数组，初始化为"O"标签
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(labels, dtype=torch.long)
        item['sentence'] = sentence  # 添加原始句子，供后续打印使用

        return item

    def __len__(self):
        return self.len


# 主函数
if __name__ == '__main__':
    print("========= Question1 =========")

    # 初始化tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # 读取数据集
    df = pd.read_csv('ner_dataset.csv', encoding='latin1', on_bad_lines='skip')

    # 使用前向填充填补空的'Sentence #'列
    df['Sentence #'] = df['Sentence #'].ffill()

    # 将非字符串的单词和标签替换为缺失标记'O'
    df['Word'] = df['Word'].fillna('O').astype(str)
    df['Tag'] = df['Tag'].fillna('O').astype(str)

    # 数据预处理
    sentences = df.groupby('Sentence #')['Word'].apply(list).values
    labels = df.groupby('Sentence #')['Tag'].apply(list).values

    # 定义分隔符
    separator = '|||'

    # 将句子和标签存入DataFrame
    df = pd.DataFrame({
        'sentence': [separator.join(s) for s in sentences],
        'word_labels': [separator.join(l) for l in labels]
    })
    # 测试df是否正确
    print("The sentence in df are:")
    print(df['sentence'])
    print("The word_labels in df are:")
    print(df['word_labels'])

    # 定义最大序列长度
    MAX_LEN = 128

    # 打印完整数据集大小
    print("完整数据集大小: {}".format(df.shape))

    # 划分训练集和测试集
    train_size = 0.8
    train_dataset = df.sample(frac=train_size, random_state=200)
    test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("全部数据集: {}".format(df.shape))
    print("训练集大小: {}".format(train_dataset.shape))
    print("测试集大小: {}".format(test_dataset.shape))

    # 创建数据集实例
    training_set = new_dataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = new_dataset(test_dataset, tokenizer, MAX_LEN)

    # 查看第一个训练样本
    sample = training_set[0]
    if sample is not None:
        tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'])
        labels = sample['labels']

        # 将标签ID转换为标签名称
        token_labels = [ids_to_labels[label_id.item()] for label_id in labels]

        # 打印原始句子
        original_sentence = sample['sentence']
        print("\n原始句子：")
        print(' '.join(original_sentence))

        # 打印令牌和对应的标签
        print("\n令牌和对应的标签：")
        for token, label in zip(tokens, token_labels):
            print(f"{token} : {label}")

        # 检查标签和输入ID的长度是否一致
        print("\n输入ID长度:", len(sample['input_ids']))
        print("标签长度:", len(sample['labels']))
    else:
        print("第一个训练样本为 None，无法显示令牌和标签。")
