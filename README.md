###### Q3的模型 model_weights.pth 没上传

###### Q4的模型 joint_model_weights.pth 没上传

###### **重写了Q3和Q4的代码，重新训练了模型**

# Assignment 2

We have 4 questions with a total of 100 Marks.

### Question Breakdown:

1. **Full Labels for Wordpieces** (30 Marks)

   **对词片段的完整标注**

2. **Retrain with New Labels** (20 Marks)

   **使用新标签重新训练**（20 分）

3. **Add Transition Violation Scores** (20 Marks)

   **添加转移违例分数**（20 分）

4. **Train Transition Scores** (30 Marks)

## Q1: Full Labels for Word pieces  对词片段的完整标注

Recall that there is a design decision regarding how to convert labels at token level.

请回忆之前讲过的，有一个关于如何在词片段级别转换标签的设计决策。

You need to propagate the original label of the word to all of its wordpieces and let the model train on this. For
beginning tags, the first wordpiece should have a **B** tag, and the remaining should have **I** tags.

==你需要将原始单词的标签传播到所有的词片段上==，并让模型基于这些标注进行训练。对于起始标签，第一个词片段应该被标记为**B**
标签，其余的词片段则应标记为**I**标签。

For example, if a word like **Washington** is labeled as **b-gpe**, and it gets tokenized into
`["Wash", "##ing", "##ton"]`, the labels should be `["b-gpe", "i-gpe", "i-gpe"]`.

例如，如果像**Washington**这样的单词被标注为**b-gpe**，并且经过分词后变成了`["Wash", "##ing", "##ton"]`，那么这些标签应该为
`["b-gpe", "i-gpe", "i-gpe"]`。

Implement this version of label conversion by creating a new dataset class.

通过创建一个新的数据集类来实现这种版本的标签转换。

30 Marks

30分

```python
class new_dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        # step 1: get the sentence and word labels
        sentence = self.data.sentence[index].strip().split()
        word_labels = self.data.word_labels[index].split(",")

        # step 2: use tokenizer to encode sentence (includes padding/truncation up to max length)
        # BertTokenizerFast provides a handy "return_offsets_mapping" functionality for individual tokens
        encoding = self.tokenizer(sentence,
                                  is_split_into_words=True,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)

        # step 3: create token labels for all word pieces of each tokenized word
        # If the label is B tag, only the first wordpiece label is B, others are i
        # old labels = [labels_to_ids[label] for label in word_labels]
        labels =

        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of 0 of length max_length
        # the reason is that torch-crf (getting into it later) cannot take tags out side of number of labels
        # it can accept mask, so we will be fine
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype=int) * labels_to_ids["O"]

        # set labels according to offset_mapping
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
        # if mapping[0] == 0 and mapping[1] != 0:
        # overwrite label
        #     encoded_labels[idx] = labels[i]
        #     i += 1

        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)

        return item

    def __len__(self):
        return self.len
```

### Testing Code

```python
### Testing Code Here
train_size = 0.8
train_dataset = data.sample(frac=train_size, random_state=200)
test_dataset = data.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print("FULL Dataset: {}".format(data.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = new_dataset(train_dataset, tokenizer, MAX_LEN)
testing_set = new_dataset(test_dataset, tokenizer, MAX_LEN)

print(training_set[0])
```

## Q2: Train the Model and Report BIO Violations

###### **训练模型并且报告非法BIO**（20 分）

1. Train the model on the new labels, report the testing set performance with `classification_report` (10 Marks).

   使用新标签训练模型，并报告测试集的`classification_report`（10 分）。

2. Gather the statistics of BIO rule violations. $\frac{\#Violations}{\#PredictedLabels}$ (10 Marks).

   收集非法BIO规则的统计数据。$\frac{\#Violations}{\#PredictedLabels}$（10 分）。

A violation happens when "I-tag" is not preceded by "I-tag" or "B-tag".

当"I-tag"标签前面没有"I-tag"或"B-tag"时，就会发生违例。

20 Marks in total for Q2

Q2总计20分。

### Train and Validate Code

```python
# Active Accuracy can no longer be based on label != -100, we use attention_mask
# No need to fix anything here.
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
```

```python
# Retrain and Produce Classification Report
```

```
# Gather the statistics of BIO rule violations. Is there violations in labels?

def BIO_violations(predictions):
    return # violations, # predictions != 0
```

## Q3 Use pytorch-crf to add transition scores to ==rule out== violations.

###### Q3 使用pytorch-crf添加转移分数来消除违例。

This can be achieved via manully setting the ==crf_model.transitions.data==, then rewrite valid method.

这可以通过手动设置`crf_model.transitions.data`来实现，然后重写`valid`方法。

Then re-evaluate with the same trained Bert model as in Q2。

接着使用Q2中相同的已训练Bert模型重新评估。

Add transition scores to rule out violations and re-evaluate (10 Marks)

添加转移分数以消除违例并重新评估（10 分）。

Recompute new BIO violations. You should find no violation (10 Marks)

重新计算新的BIO规则违例。你应该发现没有违例（10 分）。

20 Marks in total for Q3。

Q3总计20分。

```
#Rewrite the valid method, get predictions and loss from crf_model
def crf_valid(model, crf_model, testing_loader):
    # put model in evaluation mode
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):

            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)

            outpus = model(input_ids=ids, attention_mask=mask, labels=labels)
            
             # outpus[1]  use crf_model to get predictions
            predictions =
             #outpus[0] should also come from crf_model
            loss =   

            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)

            if idx % 100==0:
                loss_step = eval_loss/nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")

            # compute evaluation accuracy
            flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
            flattened_predictions = predictions.view(-1) # shape (batch_size * seq_len,)

            # only compute accuracy at active labels
            active_accuracy = mask.view(-1) != 0 # shape (batch_size, seq_len)

            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            eval_labels.append(labels)
            eval_preds.append(predictions)

            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    labels = [ [ids_to_labels[id.item()] for id  in labels] for labels in eval_labels]
    predictions = [[ids_to_labels[id.item()] for id in preds] for preds in  eval_preds]

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions
```

```
#Set CRF model and evaluate
```

## Q4 Use pytorch-crf to jointly re-train the Bert model with transition score.

The idea is that you try joint training crf and bert and see how it works.

目标是你试着联合训练CRF和BERT模型并且看看能否工作。

###### Q4 使用 pytorch-crf 对 Bert 模型和转移分数进行联合训练。

Then re-evaluate with the same trained Bert model as in Q2

然后使用与 Q2 中相同的已训练 Bert 模型重新评估。

1. Train transition scores jointly with Bert model (20 Marks)

   将转移分数与 Bert 模型联合训练（20 分）

2. Report Result and Recompute new BIO violations. (10 Marks)

   报告结果并重新计算新的 BIO 违规情况（10 分）

30 Marks in total for Q4

###### Code Example

```python
# Re write the train method here, and then re-evaluate
def crf_train(model, crf_model, training_loader):
    return model, crf_model
```

###### 提示

公式表达了如何利用CRF进行联合训练，结合转移分数（transition score）和观察分数（emission
score）来计算概率。在你的问题中，这种方法是将CRF与BERT模型联合训练的核心思路。

###### 概念介绍

- **Emission** $E$: ==这是每个标签与BERT生成的特征（即emissions）相关联的得分。==表示模型对某个token属于某个标签的置信度。*
  *emission**
  （观测得分）是原始模型（例如BERT）在每个token位置上的输出得分。具体来说，emission是BERT模型输出的每个token在每个可能标签（如命名实体识别中的 "
  B-PER", "I-PER", "O" 等标签）上的预测得分，通常以logits的形式表示。
- **Transition** $M$: 表示CRF中标签之间的转移得分，反映从标签 $y_t$ 转移到标签 $y_{t+1}$ 的可能性。
- **Score**: 这个函数将给定的标签序列和特征（emissions）一起，计算整个序列的得分。

###### 公式详解

1. **Emission Scores**:
   $$
   E = \text{Emission Score} \in \mathbb{R}^{B \times L \times T}
   $$
   这里的 $B$ 表示batch size，$L$ 表示序列的长度，$T$ 表示标签的数量。即，每个token在每个标签上的得分。

2. **总得分计算**:
   $$
   \text{Score}(y, E; M) = \sum_{t=0}^{L-1} E_{t, y_t} + \sum_{t=0}^{L-2} M_{y_t, y_{t+1}}
   $$
   这个公式表示，对于给定的标签序列 $y$，计算其在特定的emission score矩阵 $E$ 和 transition matrix $M$
   上的得分。第一项是将标签 $y_t$ 在第 $t$ 个位置的得分加总，第二项是将相邻标签之间的转移得分加总。

3. **概率计算**:
   $$
   \text{P}(Y \mid E, M) = \frac{\exp(\text{Score}(Y, E; M))}{\sum_{y} \exp(\text{Score}(y, E; M))}
   $$
   这个公式定义了给定emission score $E$ 和 transition matrix $M$ 的条件下，标签序列 $Y$
   的概率。它使用softmax计算，分子是特定标签序列 $Y$ 的得分，分母是所有可能标签序列的得分的归一化项。

###### 直观理解

- 该公式展示了在给定特征 $E$ 和转移矩阵 $M$ 的情况下，如何通过联合计算标签序列的得分来获得其概率。
- 这种方法可以很好地建模序列标注任务中的标签依赖性，因此在BERT模型的输出基础上加入CRF，可以进一步捕捉到上下文中的依赖关系。

在你使用PyTorch中的CRF库时，主要工作就是利用上面的公式，在BERT的输出上计算 emission
scores，并结合CRF中的转移矩阵，按照公式来计算整个序列的得分和归一化的概率。在训练过程中，你将最大化正确标签序列的概率，从而优化模型的参数。

