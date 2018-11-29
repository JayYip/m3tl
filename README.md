# Bert for Multi-task Learning

Using BERT to do multi-task learning

## How to run

The following command will run NER and word segmentation problem on Weibo dataset.

```bash
python main.py --problem "WeiboNER&WeiboSegment" --schedule train
```

## TODO

- Add Pretraining
- Add better evaluation
- Add more ner problem