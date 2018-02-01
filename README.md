# Experiments Codes for Bi-directional Block Self-attention
* This repo is the code of paper *title*, <http://>
* The deep learning framework is [Tensorflow/TF](https://www.tensorflow.org) 1.3 (compatible with 1.2 and 1.2.1)
* Please contact author or open an issue for questions and suggestions

## Overall Requirements
* Python3 (verified on 3.5.2, or Anaconda3 4.2.0)
* tensorflow>=1.2
* Numpy

## This repo includes following parts:
* We provide an universal interface for sequence encoding layers, which includes the proposed Bi-directional Block Self-Attention Network (Bi-BloSAN), Directional Self-Attention Network(DiSAN), CNN, Bi-LSTM, Bi-GRU, Bi-SRU and Multi-head attention, as well as the corresponding context fusion models.
* The experiments code on various benchmarks, i.e., Stanford Natural Language Inference, SICK, simplified SQuAD, TREC question-type classification, fine-grained/binary Stanford Sentiment Treebank, Customer Reviews, MPQA and SUBJ.



## Usage of The Universal Interface for Context Fusion and Sentence Encoding
The codes are stored in directory **context_fusion** of this repo, and just import the functions from package **context_fusion** if you want to use them:

    from context_fusion.interface import context_fusion_layers, sentence_encoding_models

These two functions share the similar parameter definitions:

* param **rep_tensor**: [tf Float 3D Tensor] The rank must be 3, i.e., [batch_size, seq_len, feature_dim];
* param **rep_mask**: [tf Bool 2D Tensor] The mask for input with rank 2: [batch_size, seq_len];
* param **method**: [str] different methods supported by context fusion and sentence encoding, which will be introduced below table.
* param **activation_function**: [str] in (relu|elu|slu)
* param **scope**: [str] The name of Tensorflow variable scope
* param **wd**: [float >= 0] if wd>0, add related tensor to tf collectoion "weight_decay" for further l2 decay
* param **is_train**: [tf Bool Scalar] the indicator of training or test.
* param **keep_prob**: [float] dropout keep probability
* param **hn**: [int/None] hidden units num, if None, the hn is same to the input feature dim in some methods.
* **other**: if the **method** is set to block, we can add `block_len=5` to specify the length of block in Bi-BloSA.

| Method Str | Explanation | Context Fusion | Sentence Encoding |
| --- | --- | --- | --- |
| cnn_kim | CNN from Yoon Kim (just for sentence encoding) | F | T |
| no_ct | No context | F | T |
| lstm | Bi-LSTM | T | T |
| gru | Bi-GRU | T | T |
| sru | Bi-SRU (Simple Recurrent Unit) | T | T |
| cnn | CNN with context fusion added | T | T |
| multi_head | multi-head attention with attention dropout and positional encoding | T | T |
| disa | Directional Self-attention(DiSA) | T | T |
| block | Bi-directional Block Self-Attention(Bi-BloSA) | T | T |

## Experiments Codes for Paper

### Project Directories:

1. Directory **exp_SNLI** --- python project for Stanford Natural Language Inference dataset
2. Directory **exp_SICK** --- python project for Sentences Involving Compositional Knowledge dataset
3. Directory **exp_SQuAD_sim** --- python project for simplified Stanford Question Answering dataset
4. Directory **exp_SST** --- python project for fine-grained and binary Stanford Sentiment Treebank dataset
5. Directory **exp_TREC** --- python project for TREC question-type classification dataset
6. Directory **exp_SC** --- python project for three sentence classification bench marks: Customer Review, MPQA and SUBJ.

### Shared Python Parameters to Run the Experiments Codes
Here, we introduce the **shared parameters** which appear in all these benchmark projects:

* `--network_type`: [str] use "exp_context_fusion" for reproduce our experiments
* `--log_period`: [int] step period to save the summary
* `--eval_period`: [int] step period to evaluate the model on dev dataset;
* `--gpu`: [int] GPU index to run the codes;
* `--gpu_mem`: [None or float] if None, it allow soft memory placement, or fixed proportion of the total memory;
* `--save_model`: [Bool] whether save the top-3 eval chechpoints;
* `--mode`: [str] all projects have "train" model to train model, only some projects have "test" model for test, please refer the main script of every project;
* `--load_model`: [Bool] if has "test" mode, set this to True to load checkpoint for test by specify the path-to-ckpt in `--load_path`;
* `--model_dir_suffix`: [str] a name for the model dir (the model dir is a part of programming framework which will be introduced latter);
* `--num_steps`: [int] the training step for mini-batch SGD;
* `--train_batch_size`: [int] training batch size
* `--test_batch_size`:  [int] dev and test batch size. Note that in squad, use `--test_batch_size_gain` instead of it: test_batch_size = test_batch_size_gain * train_batch_size;
* `--word_embedding_length`: [int] word embedding length to load glove pretrained models, which could be 100|200|300 depending on the data provided
* `--glove_corpus`: [str] GloVe corpus name, which could be 6B|42B|840B etc.;
* `--dropout`: [float] dropout keep probability;
* `--wd`: L2 regularization decay factor;
* `--hidden_units_num`: hidden units num. Note that we fix the hidden units num for all context_fusion models on all projects except squad.
* `--optimizer`: [str] mini-batch SGD optimizer, could be adam|adadelta|rmsprop;
* `--learning_rate`: [float] initial learning rate for optimizer;
* `--context_fusion_method`: context fusion method for `network_type` of context_fusion, could be block|lstm|gru|sru|sru_normal|cnn|cnn_kim|multi_head|multi_head_git|disa|no_ct;
* `--block_len`: [None or int] if `context_fusion_method` is set to block, the int block len is the parameter for Bi-BloSA. if the block_len is None, the program will invoke a function to calculate the block len according to the dataset characteristics automatically.


### Programming Framework for all Experiments Codes
We first demonstrate the file directory tree of all these projects:

```
ROOT
--dataset[d]
----glove[d]
----$task_dataset_name$[d]
--src[d]
----model[d]
------template.py[f]
------context_fusion.py[f]
----nn_utils[d]
----utils[d]
------file.py[f]
------nlp.py[f]
------record_log.py[f]
------time_counter.py[f]
----dataset.py[f]
----evaluator.py[f]
----graph_handler.py[f]
----perform_recorder.py[f]
--result[d]
----processed_data[d]
----model[d]
------$model_specific_dir$[d]
--------ckpt[d]
--------log_files[d]
--------summary[d]
--------answer[d]
--configs.py[f]
--$task$_main.py[f]
--$task$_log_analysis.py[f]
```

Note: The result dir will appear after the first running.

We elaborate on the every files[f] and directory[d] as follows:

`./configs.py`: perform the parameters parsing and definitions and declarations of global variables, e.g., parameter definition/default value, name(of train/dev/test_data, model, processed_data, ckpt etc.) definitions, directories(of data, result, `$model_specific_dir$` etc.) definitions and corresponding paths generation. 

`./$task$_main.py`: this is the main entry python script to run the project;

`./$task$_log_analysis.py`: this provides a function to analyze the log file of training process. 

`./dataset/`： this is the directory including datasets for current project.

* `./dataset/glove`: including pre-trained glove file
* `./dataset/$task_dataset_name$/`: This is the dataset dir for current task, we will concretely introduce this in each project dir.

`./src`: dir including python scripts

* `./src/dataset.py`: a class to process raw data from dataset, including data tokenization, token dictionary generation, data digitization, neural network data generation. In addition, there are also some method: `generate_batch_sample_iter` for random mini-batch iteration, `get_statistic` for sentence length statistics and a interface for deleting samples with long sentence in training data.
* `./src/evaluator.py`: a class for model evaluation.
* `./src/graph_handler.py`: a class for handling graph: session initialization, summary saving, model restore etc.
* `./src/perform_recoder.py`: a class to save top-n dev accuracy model checkpoint for future loading.
* `./src/model/`: the dir including tensorflow model file
* `./src/model/template.py`: a abstract python class, including network placeholders, global tensorflow tensor variables, TF loss function, TF accuracy function, EMA for learnable variables and summary, training operation, feed dict generation and training step function. 
* `./src/model/context_fusion.py`: Main TF neural network model, implement the abstract interface `build_network` inheriting from `template.py`.
* `./src/nn_utils/`: a package include various tensorflow layers implemented by this repo author.
* `./src/utils/file.py`: file I/O functions.
* `./src/utils/nlp.py`: natural language processing functions.
* `./src/utils/record_log.py`: a log recorder class, and a corresponding instance for all use in current project.
* `./src/utils/time_counter.py`: time counter class to collect the training time, note this time exclude the process to prepare data but only the time spending in training step.

`./result/`: a dir to place the results.

* `./result/processed_data/`: a dir to place Dataset instance with pickle format. The file name is generated by `get_params_str` in `./config.py` according to the related parameters.
* `./result/model/$model_specific_dir$/`: the name of this dir is generated by `get_params_str` in `./config.py` according to the related parameters, to save the result for a combination of the parameters. In other words, we associate a dir for a combination of the parameters.
* `./result/model/$model_specific_dir$/ckpt/`: a dir to save top-n model checkpoints.
* `./result/model/$model_specific_dir$/log_files/`: a dir to save log file.
* `./result/model/$model_specific_dir$/summary/`: a dir to save tensorboard summary and tensorflow graph meta files.
* `./result/model/$model_specific_dir$/answer/`: a dir to save extra prediction result for a part of these projects.


### Usage of the Experiments Projects

#### Python Package requirements

*  tqdm
*  nltk (please download Models/punkt)

#### Running a Project

    git clone https://github.com/code4review/BiBloSA
    cd $project_dir$

Then, based on the parameters introduction and programming framework, please refer to the `README.md` in the `$project_dir$` for data preparation, data processing and network training. 

**Note** that due to many projects this repo includes, it is inevitable that there are some wrong when I organize the projects into this repo. If you confront some bugs or errors when running the codes, please feel free to report them by opening a issues. I will reply it ASAP.

## Acknowledge






