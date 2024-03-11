#!/bin/bash

# to make script executable, run chmod +x train.sh 
# NOTE TO SELF: MAKE PATHS ABSOLUTE
DATA='/home/jennamansueto/text2gloss/code/data/aslg_pc12/clean/' 
SRC_LANG='en' #Source Language
TGT_LANG='asl' #Target Language
LOG_ROOT_DIR='/home/jennamansueto/text2gloss/code/models/logs'
LOG_DIR="${LOG_ROOT_DIR}/basic_transformer"

TOKENIZER_TYPE='moses' 
BPE_TYPE='subword_nmt' 
BPE_CODES="data/alsg_pc12/bpe_codes.txt" 

EPOCH=30 #Number of epochs
OPTIMIZER='adam' 
SCHEDULER='inverse_sqrt' #Learning rate growth planner
LOSS='label_smoothed_cross_entropy' #Loss metric
ARCHITECTURE='transformer' #Basic transformer
LEARNING_RATE=5e-4 
MAX_TOKENS=2048 #Maximum number of tokens in a sentence
SHARDS=4

SAVE_DIR='/home/jennamansueto/text2gloss/code/models/basic_transformer/checkpoints/'

#--warmup-updates: How many learning rate warmup steps to use
CUDA_VISIBLE_DEVICES=0,1 fairseq-train $DATA \
        --source-lang $SRC_LANG --target-lang $TGT_LANG \
        --bpe $BPE_TYPE --bpe-codes=$BPE_CODES \
        --max-epoch $EPOCH \
        --max-tokens $MAX_TOKENS \
        --optimizer $OPTIMIZER \
        --lr-scheduler $SCHEDULER --warmup-updates 4000 \
        --dropout 0.1 --weight-decay 0.0 \
        --arch $ARCHITECTURE \
        --lr $LEARNING_RATE \
        --criterion $LOSS \
        --share-decoder-input-output-embed \
        --encoder-learned-pos \
        --update-freq 8 \
        --save-interval-updates 10000 \
        --validate-interval-updates 10000 \
        --save-dir $SAVE_DIR \
        --tensorboard-logdir $LOG_DIR \
        --patience 5 \
        --num-shards $SHARDS \