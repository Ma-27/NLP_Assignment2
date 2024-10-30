# Question 3 - BERT+激励矩阵

#### 测试结果

```
========= Question3 =========
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
<ipython-input-28-93bdb2a8bb64>:305: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  saved_state_dict = torch.load('/content/model_weights.pth', map_location=device)
已成功加载训练好的Bert模型权重
已设置CRF模型的转移矩阵以消除BIO违例

========= 验证模型 =========
每100步的验证损失: 6.0935516357421875
验证损失: 4.458377907276153
验证准确率: 0.9640695438242186

========= 统计BIO规则违例 =========
BIO规则违例数: 748
预测标签总数: 254887
BIO规则违例比例: 0.0029

========= 分类报告（使用CRF模型） =========
              precision    recall  f1-score   support

       B-art       0.33      0.21      0.26        94
       B-eve       0.50      0.34      0.41        70
       B-geo       0.87      0.90      0.88      7558
       B-gpe       0.95      0.95      0.95      3142
       B-nat       0.45      0.38      0.41        40
       B-org       0.79      0.72      0.75      4151
       B-per       0.87      0.85      0.86      3400
       B-tim       0.92      0.90      0.91      4077
       I-art       0.27      0.20      0.23       161
       I-eve       0.16      0.05      0.08        77
       I-geo       0.82      0.86      0.84      5816
       I-gpe       0.94      0.78      0.85       310
       I-nat       0.50      0.23      0.31        31
       I-org       0.78      0.71      0.74      6892
       I-per       0.86      0.94      0.90      8236
       I-tim       0.77      0.77      0.77      1662
           O       0.99      0.99      0.99    209170

    accuracy                           0.96    254887
   macro avg       0.69      0.63      0.66    254887
weighted avg       0.96      0.96      0.96    254887

```
