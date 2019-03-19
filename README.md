# Bert for Multi-task Learning

[中文文档](#Bert多任务学习)

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

There are two types of chaining operations can be used to chain problems.

- `&`. If two problems have the same inputs, they can be chained using `&`. Problems chained by `&` will be trained at the same time.
- `|`. If two problems don't have the same inputs, they need to be chained using `|`. Problems chained by `|` will be sampled to train at every instance.

For example, `CWS|NER|weibo_ner&weibo_cws`, one problem will be sampled at each turn, say `weibo_ner&weibo_cws`, then `weibo_ner` and `weibo_cws` will trained for this turn together. Therefore, in a particular batch, some tasks might not be sampled, and their loss could be 0 in this batch.

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

1. Implement data preprocessing function and import it into `src/data_preprocessing/__init__.py`. Please note that the funcion name should the same as problem name. One example can be found below.

2. Add problem config to class `Params`'s attribute `self.problem_type` in `src/params.py`. Supported problem types are `cls`(classification), `seq_tag`(sequence labeling), `seq2seq_tag`(seq2seq tagging, please refer to chunking), `seq2seq_text`(seq2seq text generation).

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

# Bert多任务学习

*更新:* 在多个中文NLP任务上训练好的模型 (稍后放出)

## 这是什么

这是利用[BERT](https://github.com/google-research/bert)进行**多任务学习**并且支持多GPU训练的项目.

## 我为什么需要这个项目

在原始的BERT代码中, 是没有办法直接用多GPU进行多任务学习的. 另外, BERT并没有给出序列标注和Seq2seq的训练代码.

因此, 和原来的BERT相比, 这个项目具有以下特点:

1. 多任务学习
2. 多GPU训练
3. 序列标注以及Encoder-decoder seq2seq的支持(用transformer decoder)

## 如何运行预定义任务

### 目前支持的任务

- 中文命名实体识别
- 中文分词
- 中文词性标注

### 如何训练预定义任务

可以用两种方法来将多个任务连接起来.

- `&`. 如果两个任务有相同的输入, 不同标签的话, 那么他们**可以**用`&`来连接. 被`&`连接起来的任务会被同时训练.
- `|`. 如果两个任务为不同的输入, 那么他们**必须**用`|`来连接. 被`|`连接起来的任务会被随机抽取来训练.

例如, 我们定义任务`CWS|NER|weibo_ner&weibo_cws`, 那么在生成每一条数据时, 一个任务块会被随机抽取出来, 例如在这一次抽样中, `weibo_ner&weibo_cws`被选中. 那么这次`weibo_ner`和`weibo_cws`会被同时训练. 因此, 在一个batch中, 有可能某些任务没有被抽中, loss为0.

接着, 你可以用下面这个命令开始训练.

```bash
python main.py --problem "CWS|NER|weibo_ner&weibo_cws" --schedule train --model_dir "tmp/multitask"
```

你需要单独对每个任务做evaluation.

```bash
python main.py --problem "CWS" --schedule eval --model_dir "tmp/multitask"
python main.py --problem "NER" --schedule eval --model_dir "tmp/multitask"
python main.py --problem "weibo_ner&weibo_cws" --schedule eval --model_dir "tmp/multitask"
```

### 如何使用已经训练完成的模型

*推荐使用[这个](https://github.com/JayYip/bert-as-service)项目来serve模型*

```bash
python main.py --problem "CWS|NER|weibo_ner&weibo_cws" --schedule train --model_dir "tmp/multitask"
python export_model.py --problem "CWS|NER|weibo_ner&weibo_cws" --model_dir "tmp/multitask"
```

上面两行命令会训练模型并输出到目录`tmp/multitask/serve_model`.

然后就可以用下面的命令开启模型服务了. 请确保服务端已经正确安装.

```bash
bert-serving-start -num_worker 2 -gpu_memory_fraction 0.95 -device_map 0 1 -problem "CWS|NER|weibo_ner&weibo_cws" -model_dir tmp/multitask/serve_model
```

## 如何添加自定义的任务

1. 在`src/data_preprocessing/`目录中编写数据预处理函数, 并且import到`src/data_preprocessing/__init__.py`中. 需要注意的是函数名和任务名需要相同. 可以参考下面的例子.

2. 将任务类型添加到`src/params.py`里面的`Params`类的`self.problem_type`中. 目前支持的类型为`cls`(分类), `seq_tag`(序列标注), `seq2seq_tag`(seq2seq标注, 即非定长标注, 参考chunking), `seq2seq_text`(seq2seq文本生成).

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
