#! /bin/sh

TRAIN_PATH=test/test_data/train.txt
DEV_PATH=test/test_data/dev.txt
EXPT_DIR=example

# set values
EMB_SIZE=128
H_SIZE=128
CELL='gru'
EPOCH=6
PRINT_EVERY=10
TF=0.5

# Start training
echo "Train model on example data"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF --bidirectional --attention

echo "Run predictor"
python predict.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) 
