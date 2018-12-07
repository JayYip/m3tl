# Bert for Multi-task Learning

## What is it

First of all, bad repo name, sorry about that.

This a project that uses [BERT](https://github.com/google-research/bert) to do **multi-task learning** with multiple GPU support.

## Why do I need this

In the original BERT code, neither multi-task learning or multiple GPU training is possible. Plus, the original purpose of this project is NER which dose not have a working script in the original BERT code.

To sum up, you can use this project if you:

1. Need multi-task learning
2. Need multiple GPU training
3. Need Sequence labeling

## How to run pre-defined problems

### How to use pretrained model

Pretrained models for CWS and NER are released. You can use simplified api defined in `src/estimator_wrapper.py` as follow.

```python
    params = Params()
    m = ChineseNER(params, gpu=1)
    print(m.ner(['''正如老K教练所说，勒布朗姆斯的领袖气质开始凸显。面对外界的质疑，勒布朗表
示，“梦十队”一定会在伦敦奥运会上成功卫冕，“我们的球队不只想变得更强，而
且想拿到金牌，”詹姆斯说，“很多人都认为表示没法拿到金牌。他们认为我们在
身高上存在缺陷，说我们没有全身心地投入到国家队当中。但是我们会全身心地投
入，去迎接挑战。我现在已经迫不及待要出场比赛了。”''']))
```

### How to train

This project is still in very early stage, and the available problems are quite limited. Currently provided pre-defined problems are([results](src/data_preprocessing/README.md)):

1. WeiboNER (Chinese Named Entity Recognition)
2. WeiboSegment (Chinese Word Segmentation)
3. WeiboFakeCLS (Just a testing classification problem)
4. WeiboPretrain
5. CWS (download trained checkpoint [here](https://1drv.ms/f/s!An_n1-LB8-2dgetSfhcrMKkjE5VSWA))
6. NER (download trained checkpoint [here](https://1drv.ms/f/s!An_n1-LB8-2dgetZrmW7a2hH2kSluw))

#### Multitrask Training

There are two types of chaining operations can be used to chain problems.

- `&`. If two problems have the same inputs, they can be chained using `&`. Problems chained by `&` will be trained at the same time.
- `|`. If two problems don't have the same inputs, they need to be chained using `|`. Problems chained by `|` will be sampled to train at every instance.

For example, `CWS|NER|WeiboNER&WeiboSegment`, one problem will be sampled at each turn, say `WeiboNER&WeiboSegment`, then `WeiboNER` and `WeiboSegment` will trained for this turn together.

You can train using the following command.

```bash
python main.py --problem "CWS|NER|WeiboNER&WeiboSegment" --schedule train --model_dir "tmp/multitask"
```

For evaluation, you need to separate the problems.

```bash
python main.py --problem "CWS" --schedule eval --model_dir "tmp/multitask"
python main.py --problem "NER" --schedule eval --model_dir "tmp/multitask"
python main.py --problem "WeiboNER&WeiboSegment" --schedule eval --model_dir "tmp/multitask"
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
- ~~Add more ner problem~~
