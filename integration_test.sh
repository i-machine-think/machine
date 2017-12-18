#! /bin/sh

TRAIN_PATH=test/test_data/train_small.txt
DEV_PATH=test/test_data/dev_small.txt
EXPT_DIR=experiment

# use small parameters for quicker testing
EMB_SIZE=3
H_SIZE=5
CELL='lstm'
EPOCH=3

# check basic training, resuming training and leading checkpoint

# Start training
echo "Test training"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH

# Resume training
echo "\n\nTest resume training"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --resume --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH

echo "\n\nTest load checkpoint"
# Load checkpoint
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --epoch $EPOCH --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --load_checkpoint $(ls -t experiment/checkpoints/ | head -1)

# test with attention
echo "\n\nTest training with attention"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --attention --epoch $EPOCH

# test with gru cell
echo "\n\nTest GRU"
CELL="gru"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 10 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH
 
# test bidirectional
echo "\n\nTest bidirectional model"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --bidirectional --epoch $EPOCH

# test bidirectional with attention
echo "\n\nTest bidirectional model with attention"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --attention --bidirectional --epoch $EPOCH

test input optimizer
echo "\n\nTest command line optimizer"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH --optim adagrad
