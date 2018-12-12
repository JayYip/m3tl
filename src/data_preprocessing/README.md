# Predefined Problems

## WeiboNER/WeiboSegment/WeiboFakeCLS

A small dataset that is good for debug and demo.

## CWS

Chinese word segmentation. Data source: <http://sighan.cs.uchicago.edu/bakeoff2005/>

## NER

Chinese Named Entity Recognition. Trained using following dataset:

- [BosonNLP](https://bosonnlp.com/resources/BosonNLP_NER_6C.zip)
- MSRA
- Weibo

## CTBCWS/CTBPOS

Chinese Treebank 8.0. Data source: <https://wakespace.lib.wfu.edu/handle/10339/39379>

## Results

| Problem |  Acc |  Precision | Recall  | F Score  | Acc Per Sequence  |
|---|---|---|---|---|---|
| WeiboNER  |  0.965 | 0.641  |  0.661 |  0.650 |  0.119 |
| WeiboSegment |  0.948 |  - | -  |  0.949 | 0.033  |
| CWS  |  0.970 |  - | -  |  0.970 |  0.721 |
| NER  |  0.990 |  0.930 | 0.937  |  0.933 |  - |
| CTBPOS  |  0.964 |  - | -  |  0.964 |  0.537 |
| CTBCWS  |  0.984 |  - | -  |  0.984 |  0.751 |

## Multitask-Learning Result

| Problem |  Acc |  Precision | Recall  | F Score  | Acc Per Sequence  |
|---|---|---|---|---|---|
| CWS  |  0.965 |  - | -  |  0.954 |  0.665 |
| NER  |  0.992 |  0.936 | 0.946  |  0.942 |  - |
| CTBPOS  |  0.962 |  - | -  |  0.962 |  0.531 |
| CTBCWS  |  0.981 |  - | -  |  0.981 |  0.727 |
