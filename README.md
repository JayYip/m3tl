# Bert for Multi-task Learning

*Update:* Download trained checkpoint for Chinese NLP problems. (release later)

## What is it

This a project that uses [BERT](https://github.com/google-research/bert) to do **multi-task learning** with multiple GPU support.

## Why do I need this

In the original BERT code, neither multi-task learning or multiple GPU training is possible. Plus, the original purpose of this project is NER which dose not have a working script in the original BERT code.

To sum up, compared to the original bert repo, this repo has the following features:

1. Multi-task learning(major reason of re-writing the majority of code).
2. Multiple GPU training
3. Support sequence labeling (for example, NER) and Encoder-Decoder Seq2Seq(with transformer decoder).

## How to run pre-defined problems

### How to train

#### Multitask Training

There are two types of chaining operations can be used to chain problems.

- `&`. If two problems have the same inputs, they can be chained using `&`. Problems chained by `&` will be trained at the same time.
- `|`. If two problems don't have the same inputs, they need to be chained using `|`. Problems chained by `|` will be sampled to train at every instance.

For example, `CWS|NER|weibo_ner&weibo_cws`, one problem will be sampled at each turn, say `weibo_ner&weibo_cws`, then `weibo_ner` and `weibo_cws` will trained for this turn together.

You can train using the following command.

```bash
python main.py --problem "CWS|NER|weibo_ner&weibo_cws" --schedule train --model_dir "tmp/multitask"
```

For evaluation, you need to separate the problems.

```bash
python main.py --problem "CWS" --schedule eval --model_dir "tmp/multitask"
python main.py --problem "NER" --schedule eval --model_dir "tmp/multitask"
python main.py --problem "weibo_ner&weibo_cws" --schedule eval --model_dir "tmp/multitask"
```

### How to use trained model

*It is recommended to use [this](https://github.com/JayYip/bert-as-service) repo to serve model.*

```bash
python main.py --problem "CWS|NER|weibo_ner&weibo_cws" --schedule train --model_dir "tmp/multitask"
python export_model.py --problem "CWS|NER|weibo_ner&weibo_cws" --model_dir "tmp/multitask"
```

The above command will train the model and export to the path `tmp/multitask/serve_model`.

Then you can start the service with command below. Please make sure the server is installed.

```bash
bert-serving-start -num_worker 2 -gpu_memory_fraction 0.95 -device_map 0 1 -problem "CWS|NER|weibo_ner&weibo_cws" -model_dir tmp/multitask/serve_model
```

## How to add problems

1. Implement data preprocessing function and import it into `src/data_preprocessing/__init__.py`. One example can be found below.

2. Add problem config to `self.problem_type`.

```python
def weibo_fake_cls(params, mode):
    """Just a test problem to test multiproblem support

    Arguments:
        params {Params} -- params
        mode {mode} -- mode
    """
    tokenizer = FullTokenizer(vocab_file=params.vocab_file)

    # Note: split the inputs to training set and eval set
    if mode == 'train':
        inputs_list = [['科','技','全','方','位','资','讯','智','能','，','快','捷','的','汽','车','生','活','需','要','有','三','屏','一','云','爱','你']]
        target_list = [0]
    else:
        inputs_list = [['对', '，', '输', '给', '一', '个', '女', '人', '，', '的', '成', '绩', '。', '失', '望']]
        target_list = [0]

    label_encoder = get_or_make_label_encoder(
        'weibo_fake_cls', mode, target_list, 0)

    return create_single_problem_generator('weibo_fake_cls',
                                           inputs_list,
                                           new_target_list,
                                           label_encoder,
                                           params,
                                           tokenizer,
                                           mode)
```
