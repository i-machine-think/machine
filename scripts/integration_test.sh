#! /bin/sh

TRAIN_PATH=ibm-seq2seq/data/toy_reverse/train/data.txt
DEV_PATH=ibm-seq2seq/data/toy_reverse/dev/data.txt

# Start training
python ibm-seq2seq/scripts/integration_test.py --train_path $TRAIN_PATH --dev_path $DEV_PATH
# Resume training
python ibm-seq2seq/scripts/integration_test.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --resume
# Load checkpoint
python ibm-seq2seq/scripts/integration_test.py --train_path $TRAIN_PATH --dev_path $DEV_PATH \
	--load_checkpoint $(ls -t ibm-seq2seq/experiment/checkpoints/ | head -1)