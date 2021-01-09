# Bert for Multi-task Learning



[python](https://img.shields.io/badge/python%20-3.6.0-brightgreen.svg) [![tensorflow](https://img.shields.io/badge/tensorflow-1.13.1-green.svg)](https://www.tensorflow.org/) [![PyPI version fury.io](https://badge.fury.io/py/ansicolortags.svg)](https://pypi.python.org/pypi/bert-multitask-learning/) [![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/bert-multitask-learning/)

[中文文档](#Bert多任务学习)

**Note: Since 0.4.0, tf version >= 2.1 is required.**

## Install

```
pip install bert-multitask-learning
```

## What is it

This a project that uses transformers(based on huggingface transformers) to do **multi-modal multi-task learning**.

## Why do I need this

In the original BERT code, neither multi-task learning or multiple GPU training is possible. Plus, the original purpose of this project is NER which dose not have a working script in the original BERT code.

To sum up, compared to the original bert repo, this repo has the following features:

1. Multimodal multi-task learning(major reason of re-writing the majority of code).
2. Multiple GPU training
3. Support sequence labeling (for example, NER) and Encoder-Decoder Seq2Seq(with transformer decoder).

## What type of problems are supported?

- Masked LM and next sentence prediction Pre-train(pretrain)
- Classification(cls)
- Sequence Labeling(seq_tag)
- Multi-Label Classification(multi_cls)
- Multi-modal Mask LM(mask_lm)

## How to run pre-defined problems

There are two types of chaining operations can be used to chain problems.

- `&`. If two problems have the same inputs, they can be chained using `&`. Problems chained by `&` will be trained at the same time.
- `|`. If two problems don't have the same inputs, they need to be chained using `|`. Problems chained by `|` will be sampled to train at every instance.

For example, `cws|NER|weibo_ner&weibo_cws`, one problem will be sampled at each turn, say `weibo_ner&weibo_cws`, then `weibo_ner` and `weibo_cws` will trained for this turn together. Therefore, in a particular batch, some tasks might not be sampled, and their loss could be 0 in this batch.

Please see the examples in [notebooks](notebooks/) for more details about training, evaluation and export models.


# Bert多任务学习

**注意：版本0.4.0后要求tf>=2.1**

## 安装

```
pip install bert-multitask-learning
```

## 这是什么

这是利用transformer(基于huggingface transformers)进行**多模态多任务学习**的项目.

## 我为什么需要这个项目

在原始的BERT代码中, 是没有办法直接用多GPU进行多任务学习的. 另外, BERT并没有给出序列标注和Seq2seq的训练代码.

因此, 和原来的BERT相比, 这个项目具有以下特点:

1. 多任务学习
2. 多GPU训练
3. 序列标注以及Encoder-decoder seq2seq的支持(用transformer decoder)

## 目前支持的任务类型

- Masked LM和next sentence prediction预训练(pretrain)
- 单标签分类(cls)
- 序列标注(seq_tag)
- 多标签分类(multi_cls)
- 多模态Mask LM(mask_lm)

## 如何运行预定义任务

可以用两种方法来将多个任务连接起来.

- `&`. 如果两个任务有相同的输入, 不同标签的话, 那么他们**可以**用`&`来连接. 被`&`连接起来的任务会被同时训练.
- `|`. 如果两个任务为不同的输入, 那么他们**必须**用`|`来连接. 被`|`连接起来的任务会被随机抽取来训练.

例如, 我们定义任务`cws|NER|weibo_ner&weibo_cws`, 那么在生成每一条数据时, 一个任务块会被随机抽取出来, 例如在这一次抽样中, `weibo_ner&weibo_cws`被选中. 那么这次`weibo_ner`和`weibo_cws`会被同时训练. 因此, 在一个batch中, 有可能某些任务没有被抽中, loss为0.

训练, eval和导出模型请见[notebooks](notebooks/)
