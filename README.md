# Bert for Multi-task Learning

## What is it

This a project that uses [BERT](https://github.com/google-research/bert) to do multi-task learning with multiple GPU support.

## Why do I need this

In the original BERT code, neither multi-task learning or multiple GPU training is possible. Plus, the original purpose of this project is NER which dose not have a working script in the original BERT code.

To sum up, you can use this project if you:

1. Need multi-task learning
2. Need multiple GPU training
3. Need Sequence labeling

## How to run pre-defined problems

This project is still in very early stage, and the available problems are quite limited. Currently provided pre-defined problems are:

1. WeiboNER (Chinese Named Entity Recognition)
2. WeiboSegment (Chinese Word Segmentation)
3. WeiboFakeCLS (Just a testing classification problem)
4. WeiboPretrain
5. CWS (Chinese Word Segmentation with icwb2 data, download trained checkpoint [here](https://1drv.ms/f/s!An_n1-LB8-2dgetSfhcrMKkjE5VSWA))

The following command will run NER and word segmentation problem on Weibo dataset.

```bash
python main.py --problem "WeiboNER&WeiboSegment" --schedule train
```

## How to add problems

1. Implement data preprocessing function and import it into `src/data_preprocessing/__init__.py`. One example can be found below.


2. Add problem config to `self.problem_type` and `self.num_classes` in `src/params.py`

```python
def WeiboFakeCLS(params, mode):
    """Just a test problem to test multiproblem support

    Arguments:
        params {Params} -- params
        mode {mode} -- mode
    """
    tokenizer = FullTokenizer(vocab_file=params.vocab_file)

    inputs_list = [['科','技','全','方','位','资','讯','智','能','，','快','捷','的','汽','车','生','活','需','要','有','三','屏','一','云','爱','你'],
 ['对', '，', '输', '给', '一', '个', '女', '人', '，', '的', '成', '绩', '。', '失', '望']]
    target_list = [0, 1]

    label_encoder = get_or_make_label_encoder(
        'WeiboFakeCLS', mode, target_list, 0)

    return create_single_problem_generator('WeiboFakeCLS',
                                           inputs_list,
                                           new_target_list,
                                           label_encoder,
                                           params,
                                           tokenizer)
```

## TODO

- ~~Add multiple GPU support AdamWeightDecayOptimizer~~
- ~~Add Pretraining~~
- Add better evaluation
- Add more ner problem
