# Stanford Natural Language Inference (SNLI)
Dataset official web site: [https://nlp.stanford.edu/projects/snli/](https://nlp.stanford.edu/projects/snli/)

## Data Preparation

* Please download the 6B pre-trained model to `./dataset/glove/`. The download address is [http://nlp.stanford.edu/data/glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)
* Please download the SNLI dataset files from [https://nlp.stanford.edu/projects/snli/snli_1.0.zip](https://nlp.stanford.edu/projects/snli/snli_1.0.zip).

**Data files checklist for running with default parameters:**
    
    ./dataset/glove/glove.6B.300d.txt
    ./dataset/snli_1.0/snli_1.0_train.jsonl
    ./dataset/snli_1.0/snli_1.0_dev.jsonl
    ./dataset/snli_1.0/snli_1.0_test.jsonl

## Training model

Just simply run the code:

```
python3 snli_main.py --network_type exp_context_fusion --context_fusion_method block --model_dir_suffix training --gpu 0
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


    


