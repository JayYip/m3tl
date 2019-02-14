# Bert for Multi-task Learning

## What is it

This a project that uses [BERT](https://github.com/google-research/bert) to do **multi-task learning** with multiple GPU support.

## Why do I need this

In the original BERT code, neither multi-task learning or multiple GPU training is possible. Plus, the original purpose of this project is NER which dose not have a working script in the original BERT code.

To sum up, you can use this project if you:

1. Need multi-task learning
2. Need multiple GPU training
3. Need Sequence labeling

## How to run pre-defined problems

### How to train

This project is still in very early stage, and the available problems are quite limited. You can find available problems and baseline [here](baseline.md)

#### Multitask Training

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

### How to use trained model

*It is recommended to use [this](https://github.com/JayYip/bert-as-service) repo to serve model.*

```bash
python main.py --problem "CWS|NER|WeiboNER&WeiboSegment" --schedule train --model_dir "tmp/multitask"
python export_model.py --problem "CWS|NER|WeiboNER&WeiboSegment" --model_dir "tmp/multitask"
```

The above command will train the model and export to the path `tmp/multitask` and create two files: `export_model` and `params.json`.

Then you can start the service with command below. You need to make sure `export_model` and `params.json` and corresponding label encoders are located in the folder you specified below.

```bash
bert-serving-start -num_worker 2 -gpu_memory_fraction 0.95 -device_map 0 1 -problem "CWS|NER|WeiboNER&WeiboSegment" -model_dir tmp/multitask
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
                                           tokenizer,
                                           mode)
```

## TODO

- ~~Add multiple GPU support AdamWeightDecayOptimizer~~
- ~~Add Pretraining~~
- Add better evaluation
- ~~Add more ner problem~~
