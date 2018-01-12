#! /bin/sh

mkdir test_exp

TRAIN_PATH=test/test_data/train_small.txt
DEV_PATH=test/test_data/dev_small.txt
EXPT_DIR=test_exp

# use small parameters for quicker testing
EMB_SIZE=3
H_SIZE=5
CELL='lstm'
EPOCH=4
CP_EVERY=3

# check basic training, resuming training and leading checkpoint

# Start training
echo "Test training"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH --save_every $CP_EVERY

# Resume training
echo "\n\nTest resume training"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --resume --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH --load_checkpoint $(ls -t test_exp/ | head -1) --save_every $CP_EVERY --optim rmsprop

echo "\n\nTest train from checkpoint"
# Load checkpoint
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --epoch $EPOCH --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --load_checkpoint $(ls -t test_exp/ | head -1) --save_every $CP_EVERY

# test training without dev set
echo "\n\nTest training without dev set"
python train_model.py --train $TRAIN_PATH --output_dir $EXPT_DIR --print_every 10 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH --save_every $CP_EVERY

# Resume training without devset
echo "\n\nTest resume training without dev set"
python train_model.py --train $TRAIN_PATH --resume --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH --save_every $CP_EVERY --load_checkpoint $(ls -t test_exp/ | head -1)

echo "\n\nTest train from checkpoint without dev set"
# Load checkpoint
python train_model.py --train $TRAIN_PATH --output_dir $EXPT_DIR --print_every 50 --epoch $EPOCH --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --load_checkpoint $(ls -t test_exp/ | head -1) --save_every $CP_EVERY --optim sgd
 
# test with attention
echo "\n\nTest training with attention"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --attention --epoch $EPOCH --save_every $CP_EVERY

# test bidirectional
echo "\n\nTest bidirectional model"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --bidirectional --epoch $EPOCH --save_every $CP_EVERY

# test bidirectional with attention
echo "\n\nTest bidirectional model with attention"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --attention --bidirectional --epoch $EPOCH --save_every $CP_EVERY

# test input optimizer
echo "\n\nTest command line optimizer"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH --optim adagrad --save_every $CP_EVERY

# test encoder dropout
echo "\n\nTest encoder dropout"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH --save_every $CP_EVERY --dropout_p_encoder 0.5

# test decoder dropout
echo "\n\nTest decoder dropout"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH --save_every $CP_EVERY --dropout_p_decoder 0.5

# test n_layers
echo "\n\nTest n_layers"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH --save_every $CP_EVERY --n_layers 2

rm -r $EXPT_DIR
