#! /bin/sh

TRAIN_PATH=test/test_data/train.txt
DEV_PATH=test/test_data/dev.txt
EXPT_DIR=example

# set values
EMB_SIZE=16
H_SIZE=64
N_LAYERS=1
CELL='gru'
EPOCH=6
PRINT_EVERY=10
TF=0.5

# Start training
echo "Train model on example data"
python train_model.py --train $TRAIN_PATH --output_dir $EXPT_DIR --print_every $PRINT_EVERY --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF --attention 'pre-rnn' --attention_method 'mlp'

echo "\n\nEvaluate model on test data"
python evaluate.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) --test_data $TRAIN_PATH

echo "\n\nRun in inference mode"
python infer.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) 
