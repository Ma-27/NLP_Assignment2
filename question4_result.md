# Question 4 - BERT+CRF联合模型

#### 训练+测试结果

```
========= Question4 =========
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
训练集加载器参数: {'batch_size': 64, 'shuffle': True, 'num_workers': 0, 'collate_fn': <function collate_fn at 0x783312141cf0>}
测试集加载器参数: {'batch_size': 64, 'shuffle': False, 'num_workers': 0, 'collate_fn': <function collate_fn at 0x783312141cf0>}
第一个训练批次的Key: dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'labels'])
第一个训练批次 'input_ids' shape: torch.Size([64, 128])
第一个训练批次 batch 'labels' shape: torch.Size([64, 128])
Some weights of BertForTokenClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
/usr/local/lib/python3.10/dist-packages/transformers/optimization.py:591: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  warnings.warn(
模型初始化，使用 19 个标签.
模型使用  cuda 设备.

========= 训练Epoch 1 =========
训练中: 100%|██████████| 600/600 [13:34<00:00,  1.36s/it]
平均训练损失: 6.1605

========= 训练Epoch 2 =========
训练中: 100%|██████████| 600/600 [13:35<00:00,  1.36s/it]
平均训练损失: 3.1475

========= 训练Epoch 3 =========
训练中: 100%|██████████| 600/600 [13:34<00:00,  1.36s/it]
平均训练损失: 2.4715

========= 训练Epoch 4 =========
训练中: 100%|██████████| 600/600 [13:33<00:00,  1.36s/it]
平均训练损失: 1.9149

========= 训练Epoch 5 =========
训练中: 100%|██████████| 600/600 [13:33<00:00,  1.36s/it]
平均训练损失: 1.5086
包含CRF层的模型权重已保存

========= 验证模型 =========
验证准确率: 0.9642
数据中的唯一标签数量: 17

========= 分类报告（使用BERT与CRF混合模型） =========
              precision    recall  f1-score   support

       B-art       0.35      0.18      0.24        94
       B-eve       0.58      0.36      0.44        70
       B-geo       0.86      0.91      0.88      7558
       B-gpe       0.97      0.94      0.95      3142
       B-nat       0.59      0.25      0.35        40
       B-org       0.77      0.73      0.75      4151
       B-per       0.88      0.83      0.86      3400
       B-tim       0.91      0.91      0.91      4077
       I-art       0.28      0.13      0.18       161
       I-eve       0.45      0.25      0.32        77
       I-geo       0.84      0.84      0.84      5816
       I-gpe       0.95      0.78      0.85       310
       I-nat       0.54      0.23      0.32        31
       I-org       0.74      0.75      0.75      6892
       I-per       0.89      0.90      0.90      8236
       I-tim       0.78      0.78      0.78      1662
           O       0.99      0.99      0.99    209170

    accuracy                           0.96    254887
   macro avg       0.73      0.63      0.67    254887
weighted avg       0.96      0.96      0.96    254887


========= 统计BIO规则违例 =========
BIO规则违例数: 561
预测标签总数: 254887
BIO规则违例比例: 0.0022

```

#### 训练+测试结果(另一个，只训练一个epoch)

```
========= Question4 =========
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
训练集加载器参数: {'batch_size': 84, 'shuffle': True, 'num_workers': 0, 'collate_fn': <function collate_fn at 0x7ca3fe581ab0>}
测试集加载器参数: {'batch_size': 84, 'shuffle': False, 'num_workers': 0, 'collate_fn': <function collate_fn at 0x7ca3fe581ab0>}
第一个训练批次的Key: dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'labels'])
第一个训练批次 'input_ids' shape: torch.Size([84, 128])
第一个训练批次 batch 'labels' shape: torch.Size([84, 128])
Some weights of BertForTokenClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
/usr/local/lib/python3.10/dist-packages/transformers/optimization.py:591: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  warnings.warn(
模型初始化，使用 19 个标签.
模型使用  cuda 设备.

========= 训练Epoch 1 =========
训练中: 100%|██████████| 457/457 [13:07<00:00,  1.72s/it]
平均训练损失: 6.6670
包含CRF层的模型权重已保存

========= 验证模型 =========
验证准确率: 0.9583
数据中的唯一标签数量: 17

========= 分类报告（使用BERT与CRF混合模型） =========
              precision    recall  f1-score   support

       B-art       0.00      0.00      0.00        94
       B-eve       0.00      0.00      0.00        70
       B-geo       0.84      0.90      0.87      7558
       B-gpe       0.95      0.93      0.94      3142
       B-nat       0.00      0.00      0.00        40
       B-org       0.72      0.67      0.69      4151
       B-per       0.81      0.86      0.83      3400
       B-tim       0.90      0.88      0.89      4077
       I-art       0.00      0.00      0.00       161
       I-eve       0.00      0.00      0.00        77
       I-geo       0.85      0.80      0.82      5816
       I-gpe       0.85      0.74      0.79       310
       I-nat       0.00      0.00      0.00        31
       I-org       0.74      0.65      0.69      6892
       I-per       0.83      0.94      0.88      8236
       I-tim       0.80      0.64      0.71      1662
           O       0.99      0.99      0.99    209170

    accuracy                           0.96    254887
   macro avg       0.55      0.53      0.54    254887
weighted avg       0.96      0.96      0.96    254887


========= 统计BIO规则违例 =========
BIO规则违例数: 883
预测标签总数: 254887
BIO规则违例比例: 0.0035

```

#### 加载预训练模型，测试结果

```
========= Question4 -使用预训练模型验证=========
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
Some weights of BertForTokenClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
<ipython-input-4-5cf8069b957c>:323: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load('/content/joint_model_weights.pth', map_location=device))
模型初始化，使用 19 个标签.
模型使用 cuda 设备.
包含CRF层的模型权重已成功加载

========= 验证模型 =========
验证准确率: 0.9597
数据中的唯一标签数量: 17

========= 分类报告（使用BERT与CRF混合模型） =========
              precision    recall  f1-score   support

       B-art       0.00      0.00      0.00        94
       B-eve       0.61      0.16      0.25        70
       B-geo       0.84      0.90      0.87      7558
       B-gpe       0.96      0.93      0.95      3142
       B-nat       0.00      0.00      0.00        40
       B-org       0.75      0.66      0.70      4151
       B-per       0.83      0.85      0.84      3400
       B-tim       0.91      0.89      0.90      4077
       I-art       0.00      0.00      0.00       161
       I-eve       0.00      0.00      0.00        77
       I-geo       0.83      0.82      0.83      5816
       I-gpe       0.77      0.75      0.76       310
       I-nat       0.00      0.00      0.00        31
       I-org       0.74      0.66      0.70      6892
       I-per       0.83      0.95      0.88      8236
       I-tim       0.78      0.72      0.75      1662
           O       0.99      0.99      0.99    209170

    accuracy                           0.96    254887
   macro avg       0.58      0.55      0.55    254887
weighted avg       0.96      0.96      0.96    254887


========= 统计BIO规则违例 =========
BIO规则违例数: 914
预测标签总数: 254887
BIO规则违例比例: 0.0036

```
