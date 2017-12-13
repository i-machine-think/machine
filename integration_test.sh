#! /bin/sh

TRAIN_PATH=test/test_data/train_small.txt
DEV_PATH=test/test_data/dev_small.txt
EXPT_DIR=experiment

# Start training
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --attention --print_every 10
# Resume training
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --resume --output_dir $EXPT_DIR --print_every 10 --attention
# Load checkpoint
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 10 \
	--load_checkpoint $(ls -t experiment/checkpoints/ | head -1) --attention
