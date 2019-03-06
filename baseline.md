# msra_ner
|             experiment             |msra_ner_Accuracy|msra_ner_F1 Score|msra_ner_Precision|msra_ner_Recall|msra_ner_Accuracy Per Sequence|
|------------------------------------|---------------:|---------------:|----------------:|-------------:|-----------------------------|
|baseline                            |          0.9951|          0.9628|           0.9641|        0.9615|-                            |
|baseline_label_smooth               |          0.9951|          0.9651|           0.9646|        0.9656|-                            |
|multitask_label_transfer_first_train|          0.9961|          0.9684|           0.9697|        0.9670|-                            |
|mix_data_baseline                   |          0.9950|          0.9646|           0.9653|        0.9639|-                            |

# city_cws
|             experiment             |city_cws_Accuracy|city_cws_F1 Score|city_cws_Precision|city_cws_Recall|city_cws_Accuracy Per Sequence|
|------------------------------------|----------------:|----------------:|------------------|---------------|-----------------------------:|
|baseline                            |           0.9818|           0.9819|-                 |-              |                        0.7286|
|baseline_label_smooth               |           0.9813|           0.9816|-                 |-              |                        0.7212|
|multitask_label_transfer_first_train|           0.9578|           0.9579|-                 |-              |                        0.4403|
|mix_data_baseline                   |           0.9496|           0.9497|-                 |-              |                        0.3847|

# CWS
|             experiment             |CWS_Accuracy|CWS_F1 Score|CWS_Precision|CWS_Recall|CWS_Accuracy Per Sequence|
|------------------------------------|------------|------------|-------------|----------|-------------------------|
|baseline                            |-           |-           |-            |-         |-                        |
|baseline_label_smooth               |-           |-           |-            |-         |-                        |
|multitask_label_transfer_first_train|0.9565559   |0.9566846   |-            |-         |0.6468515                |
|mix_data_baseline                   |0.9551066   |0.9551649   |-            |-         |0.65542763               |

# pku_cws
|             experiment             |pku_cws_Accuracy|pku_cws_F1 Score|pku_cws_Precision|pku_cws_Recall|pku_cws_Accuracy Per Sequence|
|------------------------------------|--------------:|--------------:|----------------|-------------|---------------------------:|
|baseline                            |         0.9725|         0.9726|-               |-            |                      0.4875|
|baseline_label_smooth               |         0.9721|         0.9721|-               |-            |                      0.4880|
|multitask_label_transfer_first_train|         0.9543|         0.9545|-               |-            |                      0.3036|
|mix_data_baseline                   |         0.9583|         0.9585|-               |-            |                      0.3313|

# NER
|             experiment             |NER_Accuracy|NER_F1 Score|NER_Precision|NER_Recall|NER_Accuracy Per Sequence|
|------------------------------------|------------|------------|-------------|----------|-------------------------|
|baseline                            |-           |-           |-            |-         |-                        |
|baseline_label_smooth               |-           |-           |-            |-         |-                        |
|multitask_label_transfer_first_train|      0.9895|      0.9333|       0.9275|    0.9392|-                        |
|mix_data_baseline                   |      0.9905|      0.9383|       0.9363|    0.9404|-                        |

# ctb_cws
|             experiment             |ctb_cws_Accuracy|ctb_cws_F1 Score|ctb_cws_Precision|ctb_cws_Recall|ctb_cws_Accuracy Per Sequence|
|------------------------------------|--------------:|--------------:|----------------|-------------|---------------------------:|
|baseline                            |         0.9836|         0.9837|-               |-            |                      0.7445|
|baseline_label_smooth               |         0.9834|         0.9835|-               |-            |                      0.7401|
|multitask_label_transfer_first_train|         0.9772|         0.9773|-               |-            |                      0.6461|
|mix_data_baseline                   |         0.9787|         0.9788|-               |-            |                      0.6683|

# ctb_pos
|             experiment             |ctb_pos_Accuracy|ctb_pos_F1 Score|ctb_pos_Precision|ctb_pos_Recall|ctb_pos_Accuracy Per Sequence|
|------------------------------------|--------------:|--------------:|----------------|-------------|---------------------------:|
|baseline                            |         0.9637|         0.9639|-               |-            |                      0.5395|
|baseline_label_smooth               |         0.9643|         0.9646|-               |-            |                      0.5419|
|multitask_label_transfer_first_train|         0.9634|         0.9637|-               |-            |                      0.5356|
|mix_data_baseline                   |         0.9639|         0.9642|-               |-            |                      0.5332|

# POS
|             experiment             |POS_Accuracy|POS_F1 Score|POS_Precision|POS_Recall|POS_Accuracy Per Sequence|
|------------------------------------|------------|------------|-------------|----------|-------------------------|
|baseline                            |-           |-           |-            |-         |-                        |
|baseline_label_smooth               |-           |-           |-            |-         |-                        |
|multitask_label_transfer_first_train|0.9634436   |0.96374404  |-            |-         |0.53563595               |
|mix_data_baseline                   |0.9638782   |0.9641661   |-            |-         |0.53316885               |

# boson_ner
|             experiment             |boson_ner_Accuracy|boson_ner_F1 Score|boson_ner_Precision|boson_ner_Recall|boson_ner_Accuracy Per Sequence|
|------------------------------------|----------------:|----------------:|-----------------:|--------------:|------------------------------|
|baseline                            |           0.9734|           0.8203|            0.8247|         0.8159|-                             |
|baseline_label_smooth               |           0.9749|           0.8359|            0.8254|         0.8466|-                             |
|multitask_label_transfer_first_train|           0.9657|           0.7985|            0.7698|         0.8294|-                             |
|mix_data_baseline                   |           0.9740|           0.8369|            0.8238|         0.8503|-                             |

# weibo_ner
|             experiment             |weibo_ner_Accuracy|weibo_ner_F1 Score|weibo_ner_Precision|weibo_ner_Recall|weibo_ner_Accuracy Per Sequence|
|------------------------------------|----------------:|----------------:|-----------------:|--------------:|------------------------------|
|baseline                            |           0.9806|           0.6634|            0.6943|         0.6351|-                             |
|baseline_label_smooth               |           0.9807|           0.6877|            0.7030|         0.6730|-                             |
|multitask_label_transfer_first_train|           0.9652|           0.6230|            0.5948|         0.6540|-                             |
|mix_data_baseline                   |           0.9770|           0.6635|            0.6683|         0.6588|-                             |

# as_cws
|             experiment             |as_cws_Accuracy|as_cws_F1 Score|as_cws_Precision|as_cws_Recall|as_cws_Accuracy Per Sequence|
|------------------------------------|-------------:|-------------:|---------------|------------|--------------------------:|
|baseline                            |        0.9750|        0.9751|-              |-           |                     0.8410|
|baseline_label_smooth               |        0.9761|        0.9762|-              |-           |                     0.8447|
|multitask_label_transfer_first_train|        0.9744|        0.9744|-              |-           |                     0.8328|
|mix_data_baseline                   |        0.9753|        0.9753|-              |-           |                     0.8399|

# msr_cws
|             experiment             |msr_cws_Accuracy|msr_cws_F1 Score|msr_cws_Precision|msr_cws_Recall|msr_cws_Accuracy Per Sequence|
|------------------------------------|--------------:|--------------:|----------------|-------------|---------------------------:|
|baseline                            |         0.9873|         0.9873|-               |-            |                      0.7870|
|baseline_label_smooth               |         0.9870|         0.9870|-               |-            |                      0.7847|
|multitask_label_transfer_first_train|         0.9179|         0.9229|-               |-            |                      0.2113|
|mix_data_baseline                   |         0.9095|         0.9101|-               |-            |                      0.2271|

