wikiHowToImprove-v2
===================

Towards Modeling Revision Requirements in wikiHow Instructions.

Dependencies
------------

  - pip install -r requirements.txt
  - install dynet with GPU support: 

    - BACKEND=cuda pip install git+https://github.com/clab/dynet#egg=dynet
  - get wikiHowToImprove corpus:

    - wget https://bitbucket.org/irshadbhat/wikihowtoimprove-corpus/raw/e76ebb974beb5ec859ebb9f5c78037b80c45e42c/wikiHow_revisions_corpus.txt.bz2
    - bunzip2 wikiHow_revisions_corpus.txt.bz2

  - get fastText embeddings:

    - wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
    - gunzip cc.en.300.vec.gz

Version Distinction in wikiHow and Wikipedia
--------------------------------------------

Train models from scratch
^^^^^^^^^^^^^^^^^^^^^^^^^

  - python version_distinction/lstm_binary_clf.py --batch_size 512 --train version_distinction/data/wikiHow/train.tsv --dev version_distinction/data/wikiHow/dev.tsv --pre_word_vec cc.en.300.vec --bin_vec 0 --save path/to/model  --dynet-devices CPU,GPU:0
  - python version_distinction/lstm_pairwise_ranking.py --batch_size 512 --train version_distinction/data/wikiHow/train.tsv --dev version_distinction/data/wikiHow/dev.tsv --pre_word_vec cc.en.300.vec --bin_vec 0 --save path/to/model  --dynet-devices CPU,GPU:0
  - CUDA_VISIBLE_DEVICES=0,1,2,3 python version_distinction/bert_pairwise_ranking.py --model_name_or_path bert-base-cased --data_dir version_distinction/data/wikiHow/ --model_type bert  --do_train --do_eval  --evaluate_during_training --output_dir path/to/model --per_gpu_eval_batch_size 64 --per_gpu_train_batch_size 32


Reproduce results with pre-trained models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  - python version_distinction/lstm_binary_clf.py --batch_size 512 --train version_distinction/data/wikiHow/test.tsv --pre_word_vec cc.en.300.vec --bin_vec 0 --load version_distinction/models/wikiHow_lstm_models/lstm_clf_model  --dynet-devices CPU,GPU:0
  - python version_distinction/lstm_pairwise_ranking.py --batch_size 512 --test version_distinction/data/wikiHow/test.tsv --pre_word_vec cc.en.300.vec --bin_vec 0 --save version_distinction/models/wikiHow_lstm_models/lstm_ranking_model  --dynet-devices CPU,GPU:0
  - CUDA_VISIBLE_DEVICES=0 python version_distinction/bert_pairwise_ranking.py --model_name_or_path bert-base-cased --data_dir version_distinction/data/wikiHow/ --model_type bert  --tokenizer_name bert-base-cased --do_eval --output_dir version_distinction/models/wikiHow_bert_model/ --per_gpu_eval_batch_size 64 

NOTE: For training and testing (with pretrained models) on WikiAtomicEdits, get the training set from https://github.com/google-research-datasets/wiki-atomic-edits. Use version_distinction/data/WikiAtomicEdits/{test.tsv,dev.tsv} as test and dev sets. Make sure you remove test and dev samples from the downloading corpus before training.


Predicting Revision Requirements in wikiHow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Train models from scratch
^^^^^^^^^^^^^^^^^^^^^^^^^

  - python predicting_revision_requirements/lstm_classifier.py --batch_size 512 --rev_norev_file predicting_revision_requirements/data/wikiHow_rev_norev.txt --pre_word_vec cc.en.300.vec --bin_vec 0 --save path/to/model  --dynet-devices CPU,GPU:0
  - python predicting_revision_requirements/bert_classifier.py --model_name_or_path bert-base-cased --data_dir path/to/data  --model_type bert  --do_eval --do_train --evaluate_during_training --output_dir path/to/model --tokenizer_name bert-base-cased  --per_gpu_eval_batch_size 64 --per_gpu_train_batch_size 32  #Note that you need make train.tsv, dev.tsv and test.tsv from predicting_revision_requirements/data/wikiHow_rev_norev.txt (file format- 3 columns: TRAIN/TEST/DEV [TAB] text [TAB] label). Label `0` is `requiring revison` and label `1` is `requiring no revision`. 


Reproduce results with pre-trained models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  - python predicting_revision_requirements/lstm_classifier.py --batch_size 512 --rev_norev_file predicting_revision_requirements/data/wikiHow_rev_norev.txt --pre_word_vec cc.en.300.vec --bin_vec 0 --load predicting_revision_requirements/models/lstm_model/model  --dynet-devices CPU,GPU:3
  - python predicting_revision_requirements/bert_classifier.py --model_name_or_path bert-base-cased --data_dir predicting_revision_requirements/data/wikiHow_rev_norev/  --model_type bert  --do_eval --output_dir predicting_revision_requirements/models/bert_model/  --tokenizer_name bert-base-cased  --per_gpu_eval_batch_size 128
