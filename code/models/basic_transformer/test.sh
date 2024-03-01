#!/bin/bash

DATA="/home/jennamansueto/text2gloss/code/data/test_data/clean/"
SRC_LANG='en'
TGT_LANG='asl'

#By default it creates a file called generate-test.txt
RESULTS_PATH='/home/jennamansueto/text2gloss/code/models/basic_transformer/results' 

MODEL_PATH='/home/jennamansueto/text2gloss/code/models/basic_transformer/checkpoints/checkpoint_best.pt'
N_HYPOTHESES=1
#generate sequences of maximum length ax + b, where x is the source length
MAX_LEN_a=1
MAX_LEN_b=10

TOKENIZER_TYPE='moses'
BPE_TYPE='subword_nmt'
BPE_CODES="/home/jennamansueto/text2gloss/code/data/test_data/bpe_codes.txt" #BPE model of the source language


fairseq-generate "$DATA" --source-lang $SRC_LANG --target-lang $TGT_LANG \
        --path $MODEL_PATH \
        --results-path $RESULTS_PATH \
        --bpe $BPE_TYPE --bpe-codes=$BPE_CODES \
        --remove-bpe=$BPE_TYPE \
        --max-len-a $MAX_LEN_a \
        --max-len-b $MAX_LEN_b \
        --beam 5 \
        --sacrebleu \
        --batch-size 1\
        --unkpen 0.5 #We penalise that <unk> appears in translations

# Extract generated translations, de-tokenize/de-binarize if necessary
OUTPUT_FILE="${RESULTS_PATH}/generate-test.txt"
grep ^D- "${OUTPUT_FILE}" | sort -V | cut -f3 > "${RESULTS_PATH}/generated_translations.txt"