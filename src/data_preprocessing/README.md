# Predefined Problems

## WeiboNER/WeiboSegment/WeiboFakeCLS

A small dataset that is good for debug and demo.

## CWS

Chinese word segmentation. Data source: http://sighan.cs.uchicago.edu/bakeoff2005/

## NER

Chinese Named Entity Recognition. Trained using following dataset:

- [BosonNLP](https://bosonnlp.com/resources/BosonNLP_NER_6C.zip)
- MSRA
- Weibo

## Results

| Problem |  Acc |  Precision | Recall  | F Score  | Acc Per Sequence  |
|---|---|---|---|---|---|
| WeiboNER  |  0.965 | 0.641  |  0.661 |  0.650 |  0.119 |
| WeiboSegment |  0.948 |  - | -  |  0.949 | 0.033  |
| CWS  |  0.970 |  - | -  |  0.970 |  0.721 |
| NER  |  0.990 |  0.930 | 0.937  |  0.933 |  - |