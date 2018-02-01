# Sentence Classification: Customer Reviews (CR), Opinion polarity (MPQA) and Subjectivity (SUBJ)
Dataset official web sites:

* CR: [https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html)
* MPQA: [http://mpqa.cs.pitt.edu](http://mpqa.cs.pitt.edu)
* SUBJ: [https://www.cs.cornell.edu/people/pabo/movie-review-data/](https://www.cs.cornell.edu/people/pabo/movie-review-data/)

## Data Preparation

* Please download the 6B pre-trained model to `./dataset/glove/`. The download address is [http://nlp.stanford.edu/data/glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)
* This repo has contained the dataset. (The processed datasets are used in this project, which is from [https://github.com/harvardnlp/sent-conv-torch/tree/master/data](https://github.com/harvardnlp/sent-conv-torch/tree/master/data))
**Data files checklist for running with default parameters:**
    
    ./dataset/glove/glove.6B.300d.txt
    ./dataset/sentence_classification/custrev.all
    ./dataset/sentence_classification/mpqa.all
    ./dataset/sentence_classification/subj.all

## Training model

For CR dataset, run:

```
python3 sc_main.py --network_type exp_context_fusion --context_fusion_method block --model_dir_suffix training --dataset_type cr --gpu 0
```

For MPQA dataset, change `--dataset_type` to mpqa as:

```
python3 sc_main.py --network_type exp_context_fusion --context_fusion_method block --model_dir_suffix training --dataset_type mpqa --gpu 0
```

For SUBJ dataset, run:

```
python3 sc_main.py --network_type exp_context_fusion --context_fusion_method block --model_dir_suffix training --dataset_type subj --gpu 0
```

Note:

`--network_type`: one of (1) `block` for Bi-BloSA; (2) `lstm` for Bi-LSTM; (3) `gru` for Bi-GRU; (4) `sru` for Bi-SRU; (5) `cnn_kim` for CNN based sequence encoding method; (6) `multi_head` for Multi-head attention; (7) `disa` for Directional Self-attention (DiSA);

The results will appear at the end of training. We list several frequent use parameters in training. (Please refer to the README.md in repo root for more details).

* `--num_steps`: training step;
* `--eval_period`: change the evaluation frequency/period.
* `--save_model`: [default false] if True, save model to the model ckpt dir;
* `--train_batch_size` and `--test_batch_size`: set to smaller value if GPU memory is limited.
* `--dropout` and `--wd`: dropout keep prob and L2 regularization weight decay.
* `--word_embedding_length` and `--glove_corpus`: word embedding Length and glove model.


    
    


