# Question 3 - BERT+激励矩阵

#### 测试结果

```
========= Question3 =========
/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_token.py:89: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
tokenizer_config.json: 100%
 48.0/48.0 [00:00<00:00, 2.92kB/s]
vocab.txt: 100%
 232k/232k [00:00<00:00, 660kB/s]
tokenizer.json: 100%
 466k/466k [00:00<00:00, 908kB/s]
config.json: 100%
 570/570 [00:00<00:00, 43.4kB/s]
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
model.safetensors: 100%
 440M/440M [00:01<00:00, 252MB/s]
Some weights of BertForTokenClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
<ipython-input-2-219d03220756>:335: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  saved_state_dict = torch.load('/content/model_weights.pth', map_location=device)
已成功加载训练好的Bert模型权重
已设置CRF模型的转移矩阵以消除BIO违例

========= 验证模型 =========
每100步的验证损失: 6.11678409576416
验证损失: 4.467497687339783
验证准确率: 0.9641178506208935

========= 统计BIO规则违例 =========
BIO规则违例数: 787
预测标签总数: 254887
BIO规则违例比例: 0.0031

========= 分类报告（使用CRF模型） =========
              precision    recall  f1-score   support

       B-art       0.30      0.17      0.22        94
       B-eve       0.51      0.34      0.41        70
       B-geo       0.87      0.90      0.88      7558
       B-gpe       0.96      0.95      0.95      3142
       B-nat       0.43      0.38      0.40        40
       B-org       0.78      0.72      0.75      4151
       B-per       0.88      0.84      0.86      3400
       B-tim       0.92      0.90      0.91      4077
       I-art       0.27      0.20      0.23       161
       I-eve       0.15      0.05      0.08        77
       I-geo       0.83      0.85      0.84      5816
       I-gpe       0.94      0.78      0.85       310
       I-nat       0.43      0.19      0.27        31
       I-org       0.78      0.71      0.74      6892
       I-per       0.86      0.94      0.90      8236
       I-tim       0.77      0.78      0.77      1662
           O       0.99      0.99      0.99    209170

    accuracy                           0.96    254887
   macro avg       0.69      0.63      0.65    254887
weighted avg       0.96      0.96      0.96    254887

```

## 2024.11 版本结果

当设置

```
    # 获取浮点数的最小值
    min_float = torch.finfo(torch.float32).min
    # 获取浮点数的最大值
    max_float = torch.finfo(torch.float32).max
```

时候，只能得到下面的结果，而BIO违规率为仍然没办法是0。

```
========= Question3 =========
/usr/local/lib/python3.10/dist-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
The sentence in df are:
0        [Thousands, of, demonstrators, have, marched, ...
1        [Iranian, officials, say, they, expect, to, ge...
2        [Helicopter, gunships, Saturday, pounded, mili...
3        [They, left, after, a, tense, hour-long, stand...
4        [U.N., relief, coordinator, Jan, Egeland, said...
                               ...                        
47954    [Opposition, leader, Mir, Hossein, Mousavi, ha...
47955    [On, Thursday, ,, Iranian, state, media, publi...
47956    [Following, Iran, 's, disputed, June, 12, elec...
47957    [Since, then, ,, authorities, have, held, publ...
47958    [The, United, Nations, is, praising, the, use,...
Name: words, Length: 47959, dtype: object
The word_labels in df are:
0        [O, O, O, O, O, O, B-geo, O, O, O, O, O, B-geo...
1        [B-gpe, O, O, O, O, O, O, O, O, O, O, O, O, O,...
2        [O, O, B-tim, O, O, O, O, O, B-geo, O, O, O, O...
3                        [O, O, O, O, O, O, O, O, O, O, O]
4        [B-geo, O, O, B-per, I-per, O, B-tim, O, B-geo...
                               ...                        
47954    [O, O, O, B-per, I-per, O, O, O, O, O, O, O, O...
47955    [O, B-tim, O, B-gpe, O, O, O, O, O, O, O, O, B...
47956    [O, B-geo, O, O, B-tim, I-tim, O, O, O, O, O, ...
47957    [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, ...
47958    [O, B-org, I-org, O, O, O, O, O, O, O, O, O, O...
Name: labels, Length: 47959, dtype: object
测试数据集规模: 9592
Map: 100%
 9592/9592 [00:04<00:00, 2122.97 examples/s]
Some weights of BertForTokenClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
<ipython-input-4-8e4c0dc7f409>:314: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  saved_state_dict = torch.load('/content/model_weights.pth', map_location=device)
已成功加载训练好的Bert模型权重
已设置CRF模型的转移矩阵以消除BIO违例

========= 验证模型 =========

========= 分类报告（使用CRF模型） =========
              precision    recall  f1-score   support

         art       0.17      0.14      0.15        94
         eve       0.38      0.27      0.32        70
         geo       0.84      0.89      0.86      7558
         gpe       0.95      0.95      0.95      3142
         nat       0.42      0.40      0.41        40
         org       0.67      0.66      0.66      4151
         per       0.76      0.79      0.78      3400
         tim       0.88      0.88      0.88      4077

   micro avg       0.81      0.83      0.82     22532
   macro avg       0.63      0.62      0.63     22532
weighted avg       0.81      0.83      0.82     22532


========= F1 分数 =========
F1 分数: 0.8238

========= 准确率 =========
准确率: 0.9640

========= 统计BIO规则违例 =========
BIO规则违例数: 883
预测标签总数: 254887
BIO规则违例比例: 0.0035

```

没办法，只能强制使用维特比算法，将非法矩阵设置一个极低的值，让无论模型输出多么有利，一旦非法，在动态规划中都比不过合法但是模型低分的选项。

```
========= Question3 =========
/usr/local/lib/python3.10/dist-packages/transformers/tokenization_utils_base.py:1601: FutureWarning: `clean_up_tokenization_spaces` was not set. It will be set to `True` by default. This behavior will be depracted in transformers v4.45, and will be then set to `False` by default. For more details check this issue: https://github.com/huggingface/transformers/issues/31884
  warnings.warn(
The sentence in df are:
0        [Thousands, of, demonstrators, have, marched, ...
1        [Iranian, officials, say, they, expect, to, ge...
2        [Helicopter, gunships, Saturday, pounded, mili...
3        [They, left, after, a, tense, hour-long, stand...
4        [U.N., relief, coordinator, Jan, Egeland, said...
                               ...                        
47954    [Opposition, leader, Mir, Hossein, Mousavi, ha...
47955    [On, Thursday, ,, Iranian, state, media, publi...
47956    [Following, Iran, 's, disputed, June, 12, elec...
47957    [Since, then, ,, authorities, have, held, publ...
47958    [The, United, Nations, is, praising, the, use,...
Name: words, Length: 47959, dtype: object
The word_labels in df are:
0        [O, O, O, O, O, O, B-geo, O, O, O, O, O, B-geo...
1        [B-gpe, O, O, O, O, O, O, O, O, O, O, O, O, O,...
2        [O, O, B-tim, O, O, O, O, O, B-geo, O, O, O, O...
3                        [O, O, O, O, O, O, O, O, O, O, O]
4        [B-geo, O, O, B-per, I-per, O, B-tim, O, B-geo...
                               ...                        
47954    [O, O, O, B-per, I-per, O, O, O, O, O, O, O, O...
47955    [O, B-tim, O, B-gpe, O, O, O, O, O, O, O, O, B...
47956    [O, B-geo, O, O, B-tim, I-tim, O, O, O, O, O, ...
47957    [O, O, O, O, O, O, O, O, O, O, O, O, O, O, O, ...
47958    [O, B-org, I-org, O, O, O, O, O, O, O, O, O, O...
Name: labels, Length: 47959, dtype: object
测试数据集规模: 9592
Map: 100%
 9592/9592 [00:04<00:00, 2375.66 examples/s]
Some weights of BertForTokenClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
<ipython-input-15-9c4c06d0c554>:341: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  saved_state_dict = torch.load('/content/model_weights.pth', map_location=device)
已成功加载训练好的Bert模型权重

========= 验证模型 =========
Validation Loss: 0.19225080087780952
Validation Accuracy: 0.9641174324308419

========= 分类报告（使用CRF模型） =========
              precision    recall  f1-score   support

         art       0.28      0.18      0.22        94
         eve       0.43      0.29      0.34        70
         geo       0.86      0.90      0.88      7558
         gpe       0.95      0.95      0.95      3142
         nat       0.43      0.38      0.40        40
         org       0.73      0.67      0.70      4151
         per       0.80      0.80      0.80      3400
         tim       0.89      0.88      0.88      4077

   micro avg       0.84      0.84      0.84     22532
   macro avg       0.67      0.63      0.65     22532
weighted avg       0.84      0.84      0.84     22532


========= F1 分数 =========
F1 分数: 0.8400

========= 准确率 =========
准确率: 0.9641

========= 统计BIO规则违例 =========
BIO规则违例数: 0
预测标签总数: 254887
BIO规则违例比例: 0.0000

```
