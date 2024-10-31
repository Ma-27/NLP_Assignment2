# Question 2 - BERT模型

#### 训练+测试结果

```
========= Question2 =========
/usr/local/lib/python3.10/dist-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(

The sentence in df are:
0        Thousands|||of|||demonstrators|||have|||marche...
1        Iranian|||officials|||say|||they|||expect|||to...
2        Helicopter|||gunships|||Saturday|||pounded|||m...
3        They|||left|||after|||a|||tense|||hour-long|||...
4        U.N.|||relief|||coordinator|||Jan|||Egeland|||...
                               ...                        
47954    Opposition|||leader|||Mir|||Hossein|||Mousavi|...
47955    On|||Thursday|||,|||Iranian|||state|||media|||...
47956    Following|||Iran|||'s|||disputed|||June|||12||...
47957    Since|||then|||,|||authorities|||have|||held||...
47958    The|||United|||Nations|||is|||praising|||the||...
Name: sentence, Length: 47959, dtype: object

The word_labels in df are:
0        O|||O|||O|||O|||O|||O|||B-geo|||O|||O|||O|||O|...
1        B-gpe|||O|||O|||O|||O|||O|||O|||O|||O|||O|||O|...
2        O|||O|||B-tim|||O|||O|||O|||O|||O|||B-geo|||O|...
3                O|||O|||O|||O|||O|||O|||O|||O|||O|||O|||O
4        B-geo|||O|||O|||B-per|||I-per|||O|||B-tim|||O|...
                               ...                        
47954    O|||O|||O|||B-per|||I-per|||O|||O|||O|||O|||O|...
47955    O|||B-tim|||O|||B-gpe|||O|||O|||O|||O|||O|||O|...
47956    O|||B-geo|||O|||O|||B-tim|||I-tim|||O|||O|||O|...
47957    O|||O|||O|||O|||O|||O|||O|||O|||O|||O|||O|||O|...
47958    O|||B-org|||I-org|||O|||O|||O|||O|||O|||O|||O|...
Name: word_labels, Length: 47959, dtype: object

训练数据集规模: 38367
测试数据集规模: 9592

训练集加载器参数: {'batch_size': 256, 'shuffle': True, 'num_workers': 0, 'collate_fn': <function collate_fn at 0x7fab8018e5f0>}
测试集加载器参数: {'batch_size': 256, 'shuffle': False, 'num_workers': 0, 'collate_fn': <function collate_fn at 0x7fab8018e5f0>}
First training batch keys: dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'labels'])

First training batch 'input_ids' shape: torch.Size([256, 128])
First training batch 'labels' shape: torch.Size([256, 128])

Some weights of BertForTokenClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
/usr/local/lib/python3.10/dist-packages/transformers/optimization.py:591: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  warnings.warn(
  
Model initialized with 19 labels.
Model is using device: cuda

========= 训练Epoch 1 =========
训练中: 100%|██████████| 600/600 [11:50<00:00,  1.18s/it]
平均训练损失: 0.24445816195259493

========= 训练Epoch 2 =========
训练中: 100%|██████████| 600/600 [11:50<00:00,  1.18s/it]
平均训练损失: 0.1279214130093654

========= 训练Epoch 3 =========
训练中: 100%|██████████| 600/600 [11:49<00:00,  1.18s/it]
平均训练损失: 0.10099836981544892

========= 训练Epoch 4 =========
训练中: 100%|██████████| 600/600 [11:49<00:00,  1.18s/it]
平均训练损失: 0.08067721255123615

========= 训练Epoch 5 =========
训练中: 100%|██████████| 600/600 [11:50<00:00,  1.18s/it]
平均训练损失: 0.06349939783414205
模型权重已保存

========= 验证模型 =========
Validation loss per 100 evaluation steps: 0.22014446556568146
Validation loss per 100 evaluation steps: 0.14874395388777895
Validation Loss: 0.14322802498936654
Validation Accuracy: 0.8886775942739653
数据中的唯一标签数量: 17

========= 分类报告 =========
              precision    recall  f1-score   support

       B-art       0.31      0.17      0.22        94
       B-eve       0.51      0.34      0.41        70
       B-geo       0.87      0.90      0.88      7558
       B-gpe       0.95      0.95      0.95      3142
       B-nat       0.44      0.38      0.41        40
       B-org       0.78      0.72      0.75      4151
       B-per       0.87      0.84      0.86      3400
       B-tim       0.92      0.90      0.91      4077
       I-art       0.27      0.20      0.23       161
       I-eve       0.15      0.05      0.08        77
       I-geo       0.82      0.86      0.84      5816
       I-gpe       0.94      0.78      0.85       310
       I-nat       0.43      0.19      0.27        31
       I-org       0.78      0.71      0.74      6892
       I-per       0.86      0.94      0.90      8236
       I-tim       0.77      0.77      0.77      1662
           O       0.99      0.99      0.99    209170

    accuracy                           0.96    254887
   macro avg       0.69      0.63      0.65    254887
weighted avg       0.96      0.96      0.96    254887


========= 统计BIO规则违例 =========
BIO规则违例数: 783
预测标签总数: 254887
BIO规则违例比例: 0.0031

```

#### 加载预训练模型，测试结果

```
========= Question2 -使用预训练模型验证=========
/usr/local/lib/python3.10/dist-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
数据已加载
Some weights of BertForTokenClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
<ipython-input-4-55d0167f5ffa>:236: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load('/content/model_weights.pth', map_location=device))
模型权重已加载

========= 验证模型 =========
Validation loss per 100 evaluation steps: 0.18346908688545227
Validation loss per 100 evaluation steps: 0.14735342244995703
Validation Loss: 0.14388508558273316
Validation Accuracy: 0.8844718546109912

展平标签和预测
数据中的唯一标签数量: 17

========= 分类报告 =========
              precision    recall  f1-score   support

       B-art       0.00      0.00      0.00        94
       B-eve       0.83      0.14      0.24        70
       B-geo       0.84      0.91      0.87      7558
       B-gpe       0.96      0.93      0.94      3142
       B-nat       0.00      0.00      0.00        40
       B-org       0.78      0.64      0.70      4151
       B-per       0.81      0.85      0.83      3400
       B-tim       0.90      0.89      0.90      4077
       I-art       0.00      0.00      0.00       161
       I-eve       0.00      0.00      0.00        77
       I-geo       0.85      0.81      0.83      5816
       I-gpe       0.80      0.76      0.78       310
       I-nat       0.00      0.00      0.00        31
       I-org       0.75      0.66      0.70      6892
       I-per       0.83      0.96      0.89      8236
       I-tim       0.75      0.71      0.73      1662
           O       0.99      0.99      0.99    209170

    accuracy                           0.96    254887
   macro avg       0.59      0.54      0.55    254887
weighted avg       0.96      0.96      0.96    254887


========= 统计BIO规则违例 =========
BIO规则违例数: 972
预测标签总数: 254887
BIO规则违例比例: 0.0038
```
