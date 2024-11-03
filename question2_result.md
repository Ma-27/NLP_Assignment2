# Question 2 - BERT模型

#### 训练+测试结果

```
========= Question2 =========
/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_token.py:89: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
tokenizer_config.json: 100%
 48.0/48.0 [00:00<00:00, 3.39kB/s]
vocab.txt: 100%
 232k/232k [00:00<00:00, 1.93MB/s]
tokenizer.json: 100%
 466k/466k [00:00<00:00, 9.04MB/s]
config.json: 100%
 570/570 [00:00<00:00, 38.0kB/s]
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
训练集加载器参数: {'batch_size': 96, 'shuffle': True, 'num_workers': 0, 'collate_fn': <function collate_fn at 0x7b1ff42d4310>}
测试集加载器参数: {'batch_size': 96, 'shuffle': False, 'num_workers': 0, 'collate_fn': <function collate_fn at 0x7b1ff42d4310>}
First training batch keys: dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'labels'])
First training batch 'input_ids' shape: torch.Size([96, 128])
First training batch 'labels' shape: torch.Size([96, 128])
model.safetensors: 100%
 440M/440M [00:01<00:00, 242MB/s]
Some weights of BertForTokenClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Model initialized with 19 labels.
Model is using device: cuda
/usr/local/lib/python3.10/dist-packages/transformers/optimization.py:591: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  warnings.warn(

========= 训练Epoch 1 =========
训练中: 100%|██████████| 400/400 [03:03<00:00,  2.18it/s]
平均训练损失: 0.10026455148123205

========= 训练Epoch 2 =========
训练中: 100%|██████████| 400/400 [03:01<00:00,  2.20it/s]
平均训练损失: 0.03059444472193718

========= 训练Epoch 3 =========
训练中: 100%|██████████| 400/400 [03:01<00:00,  2.20it/s]
平均训练损失: 0.025235946751199664

========= 训练Epoch 4 =========
训练中: 100%|██████████| 400/400 [03:01<00:00,  2.20it/s]
平均训练损失: 0.021192484982311726

========= 训练Epoch 5 =========
训练中: 100%|██████████| 400/400 [03:01<00:00,  2.21it/s]
平均训练损失: 0.018052768725901842
模型权重已保存

========= 验证模型 =========
Validation loss per 100 evaluation steps: 0.035723038017749786
Validation Loss: 0.02707538560964167
Validation Accuracy: 0.9616689692000808
数据中的唯一标签数量: 17

========= 分类报告 =========
              precision    recall  f1-score   support

       B-art       0.29      0.07      0.12        94
       B-eve       0.50      0.34      0.41        70
       B-geo       0.84      0.92      0.88      7558
       B-gpe       0.95      0.95      0.95      3142
       B-nat       0.30      0.25      0.27        40
       B-org       0.78      0.69      0.73      4151
       B-per       0.87      0.83      0.85      3400
       B-tim       0.90      0.91      0.91      4077
       I-art       0.22      0.11      0.14       161
       I-eve       0.34      0.14      0.20        77
       I-geo       0.80      0.88      0.84      5816
       I-gpe       0.87      0.77      0.82       310
       I-nat       0.24      0.16      0.19        31
       I-org       0.71      0.76      0.73      6892
       I-per       0.90      0.87      0.89      8236
       I-tim       0.74      0.79      0.77      1662
           O       0.99      0.99      0.99    209170

    accuracy                           0.96    254887
   macro avg       0.66      0.61      0.63    254887
weighted avg       0.96      0.96      0.96    254887


========= 统计BIO规则违例 =========
BIO规则违例数: 945
预测标签总数: 254887
BIO规则违例比例: 0.0037

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

## 2024.11 版本结果

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
训练集加载器参数: {'batch_size': 192, 'shuffle': True, 'num_workers': 0, 'collate_fn': <function collate_fn at 0x7ac684f14820>}
测试集加载器参数: {'batch_size': 192, 'shuffle': False, 'num_workers': 0, 'collate_fn': <function collate_fn at 0x7ac684f14820>}
First training batch keys: dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'offset_mapping', 'labels'])
First training batch 'input_ids' shape: torch.Size([192, 128])
First training batch 'labels' shape: torch.Size([192, 128])
Some weights of BertForTokenClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
/usr/local/lib/python3.10/dist-packages/transformers/optimization.py:591: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  warnings.warn(
Model initialized with 19 labels.
Model is using device: cuda

========= 训练Epoch 1 =========
训练中: 100%|██████████| 200/200 [02:58<00:00,  1.12it/s]
平均训练损失: 0.17898447411134838

========= 训练Epoch 2 =========
训练中: 100%|██████████| 200/200 [02:58<00:00,  1.12it/s]
平均训练损失: 0.038759582145139575

========= 训练Epoch 3 =========
训练中: 100%|██████████| 200/200 [02:58<00:00,  1.12it/s]
平均训练损失: 0.031207940690219402

========= 训练Epoch 4 =========
训练中: 100%|██████████| 200/200 [02:58<00:00,  1.12it/s]
平均训练损失: 0.027195015642791986

========= 训练Epoch 5 =========
训练中: 100%|██████████| 200/200 [02:58<00:00,  1.12it/s]
平均训练损失: 0.023916419204324482
模型权重已保存

========= 验证模型 =========
Validation loss per 100 evaluation steps: 0.030689455568790436
Validation Loss: 0.027147675193846227
Validation Accuracy: 0.961653408676246
数据中的唯一标签数量: 17

========= 分类报告 =========
              precision    recall  f1-score   support

         art       0.00      0.00      0.00        94
         eve       0.60      0.13      0.21        70
         geo       0.83      0.89      0.86      7558
         gpe       0.95      0.94      0.95      3142
         nat       0.00      0.00      0.00        40
         org       0.63      0.63      0.63      4151
         per       0.72      0.78      0.75      3400
         tim       0.84      0.87      0.85      4077

   micro avg       0.79      0.82      0.81     22532
   macro avg       0.57      0.53      0.53     22532
weighted avg       0.79      0.82      0.80     22532


========= F1 分数 =========
F1 分数: 0.8066

========= 准确率 =========
准确率: 0.9616

========= 统计BIO规则违例 =========
BIO规则违例数: 927
预测标签总数: 254887
BIO规则违例比例: 0.0036

```
