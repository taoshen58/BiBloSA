# The Stanford Question Answering Dataset (SQuAD)
Dataset official web site: [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/)

## Data Preparation

* Please download the 6B pre-trained model to `./dataset/glove/`. The download address is [http://nlp.stanford.edu/data/glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)
* Please download the SQuAD Training dataset [https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json) and dev dataset [https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json](https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json).

**Data files checklist for running with default parameters:**
    
    ./dataset/glove/glove.6B.100d.txt
    ./dataset/SQuAD/train-v1.1.json
    ./dataset/SQuAD/dev-v1.1.json

## Training model

Just simply run the code:

```
python3 squad_sim_main.py --network_type exp_context_fusion --context_fusion_method block --model_dir_suffix training --gpu 0
```

Note:

`--network_type`: one of (1) `block` for Bi-BloSA; (2) `lstm` for Bi-LSTM; (3) `gru` for Bi-GRU; (4) `sru` for Bi-SRU; (5) `cnn` for CNN based context fusion method; (6) `multi_head` for Multi-head attention; (7) `disa` for Directional Self-attention (DiSA);

The results will appear at the end of training. We list several frequent use parameters in training. (Please refer to the README.md in repo root for more details).

* `--num_steps`: training step;
* `--eval_period`: change the evaluation frequency/period.
* `--save_model`: [default false] if True, save model to the model ckpt dir;
* `--train_batch_size` and `--test_batch_size_gain`: set to smaller value if GPU memory is limited.
* `--dropout` and `--wd`: dropout keep prob and L2 regularization weight decay.
* `--word_embedding_length` and `--glove_corpus`: word embedding Length and glove model.


    


