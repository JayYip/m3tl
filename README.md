# M3TL



**M**ulti-**M**odal **M**ulti-**T**ask **L**earning

## Install

```
pip install m3tl
```

## What is it

This is a project that uses transformers(based on huggingface transformers) as base model to do **multi-modal multi-task learning**.   

## Why do I need this

Multi-task learning(MTL) is gaining more and more attention, especially in deep learning era. It is widely used in NLP, CV, recommendation, etc. However, MTL usually involves complicated data preprocessing, task managing and task interaction. Other open-source projects, like TencentNLP and PyText, supports MTL but in a naive way and it's not straightforward to implement complicated MTL algorithm. In this project, we try to make writing MTL model as easy as single task learning model and further extend MTL to multi-modal multi-task learning. To do so, we expose following MTL related programable module to user:

- problem sampling strategy
- loss combination strategy
- gradient surgery
- model after base model(transformers)

Apart from programable modules, we also provide various built-in SOTA MTL algorithms.

In a word, you can use this project to:

- implement complicated MTL algorithm
- do SOTA MTL without diving into details
- do multi-modal learning

And since we use transformers as base model, you get all the benefits that you can get from transformers!

## What type of problems are supported?

```
params = Params()
for problem_type in params.list_available_problem_types():
    print('`{problem_type}`: {desc}'.format(
        desc=params.problem_type_desc[problem_type], problem_type=problem_type))

```

    `cls`: Classification
    `multi_cls`: Multi-Label Classification
    `seq_tag`: Sequence Labeling
    `masklm`: Masked Language Model
    `pretrain`: NSP+MLM(Deprecated)
    `regression`: Regression
    `vector_fit`: Vector Fitting
    `premask_mlm`: Pre-masked Masked Language Model
    `contrastive_learning`: Contrastive Learning



## Get Started

Please see tutorials.

