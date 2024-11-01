# Question 1 - 分词标注算法

#### 分词标注结果

```
"E:\CScodes\Course\Natural Language Processing\Assignment2\.venv\Scripts\python.exe" "E:\CScodes\Course\Natural Language Processing\Assignment2\question1.py" 
========= Question1 =========
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
完整数据集大小: (47959, 2)
全部数据集: (47959, 2)
训练集大小: (38367, 2)
测试集大小: (9592, 2)

原始句子：
Speaking in Beijing Tuesday , a spokesman for the committee said the earthquake in Sichuan will not affect the relay because the quake-stricken areas are not along the route .

令牌和对应的标签：
[CLS] : O
speaking : O
in : O
beijing : B-geo
tuesday : B-tim
, : O
a : O
spokesman : O
for : O
the : O
committee : O
said : O
the : O
earthquake : O
in : O
sichuan : B-geo
will : O
not : O
affect : O
the : O
relay : O
because : O
the : O
quake : O
- : O
stricken : O
areas : O
are : O
not : O
along : O
the : O
route : O
. : O
[SEP] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O
[PAD] : O

输入ID长度: 128
标签长度: 128

进程已结束，退出代码为 0

```