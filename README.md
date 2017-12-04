# Neural Machine Translation (seq2seq) Tutorial

- [Introduction](#introduction)



# Introduction
Sequence-to-sequence (seq2seq) models
([Sutskever et al., 2014](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf),
[Cho et al., 2014](http://emnlp2014.org/papers/pdf/EMNLP2014179.pdf)) have enjoyed great success in a variety of tasks such as machine translation, speech recognition, and text summarization. This tutorial gives readers a full understanding of seq2seq models and shows how to build a competitive seq2seq model from scratch. We focus on the task of Neural Machine Translation (NMT) which was the very first testbed for seq2seq models with wild [success](https://research.googleblog.com/2016/09/a-neural-network-for-machine.html). The included code is lightweight,hig-equality,production-ready,and incorporated with the latest research ideas. We achieve this goal by:

1. Using the recent decoder/attention wrapper [API](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/seq2seq/python/ops), Tensorflow 1.2 data iterator
1. Incorporating our strong expertise in building recurrent and seq2seq models
1. Providing tips and tricks for building the very best NMT models and replicating [Google's NMT(GNMT) System](https://research.google.com/pubs/pub45610.html).

We believe that it is important to provide benchmarks that people can easily replicate. As a rsult, we have provided full experimental results and pretrained on models the folling publicly available datasets:
1. *Small-scale*: english-vietnamese parallel corpus of TED talks (113k Sentence pairs) provided by the [IWSLT Evaluation Campaign](https://sites.google.com/site/iwsltevaluation2015/).
1. *Large-scale*: German-English parallel corpus (4.5M sentence pairs) provided
   by the [WMT Evaluation Campaign](http://www.statmt.org/wmt16/translation-task.html).

We first build up some basic knowledge about seq2seq models for NMT,explaining how to build and train a vanilla NMT . The second part will  go into details of building a competitive NMT model attention mechanism. We then discuss tips and tricks to build the best possible NMT models(both in speed and translation quality) such as Tensorflow best practices (batching, bucketing), bidirectional RNNs, beam search, as well as scaling up to multiple GPUS using GNMT attention.


# Basic

## Background on Neural Machine Translation
Back in the old days, traditional phease-based translation systems performed their task by breaking up source sentences into multiple chunks and then translate them phrase-by-phrase.This led to disfluency in the translation outputs and was no quite like how we, humans,translate. We read the entire source sentence,understand it meaning, and then produce a translation. Neural Machine Translation(NMT) mimics that!

<p align="center">
<img width="80%" src="./nmt/g3doc/img/encdec.jpg" />
<br>
Figure 1. <b>Encoder-decoder architecture</b> – example of a general approach for
NMT. An encoder converts a source sentence into a "meaning" vector which is
passed through a <i>decoder</i> to produce a translation.
</p>