#! /bin/sh

TRAIN_PATH=tests/test_data/train.txt
DEV_PATH=tests/test_data/dev.txt

# Start training
python scripts/integration_test.py --train_path $TRAIN_PATH --dev_path $DEV_PATH
# Resume training
python scripts/integration_test.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --resume
# Load checkpoint
python scripts/integration_test.py --train_path $TRAIN_PATH --dev_path $DEV_PATH \
	--load_checkpoint $(ls -t experiment/checkpoints/ | head -1)
