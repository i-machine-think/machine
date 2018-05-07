#! /bin/sh

TRAIN_PATH=test/test_data/train_small.txt
DEV_PATH=test/test_data/dev_small.txt
LOOKUP=test/test_data/lookup_small.txt
EXPT_DIR=test_exp

mkdir $EXPT_DIR

# use small parameters for quicker testing
EMB_SIZE=2
H_SIZE=4
CELL='lstm'
CELL2='gru'
EPOCH=2
CP_EVERY=3

EX=0
ERR=0

# Start training
echo "Test training"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --monitor $DEV_PATH $TRAIN_PATH --output_dir $EXPT_DIR --print_every 30 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH --save_every $CP_EVERY --batch_size 6
ERR=$((ERR+$?)); EX=$((EX+1))

# Resume training
echo "\n\nTest resume training"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --resume --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH --load_checkpoint $(ls -t $EXPT_DIR | head -1) --save_every $CP_EVERY --optim rmsprop --batch_size 6
ERR=$((ERR+$?)); EX=$((EX+1))

echo "\n\nTest train from checkpoint"
# Load checkpoint
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --epoch $EPOCH --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --load_checkpoint $(ls -t $EXPT_DIR/ | head -1) --save_every $CP_EVERY
ERR=$((ERR+$?)); EX=$((EX+1))

# evaluate.py
echo "\n\nTest evaluator"
python evaluate.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) --test_data $DEV_PATH --batch_size 15

# test training without dev set
echo "\n\nTest training without dev set"
python train_model.py --train $TRAIN_PATH --output_dir $EXPT_DIR --print_every 10 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH --save_every $CP_EVERY
ERR=$((ERR+$?)); EX=$((EX+1))

# Resume training without devset
echo "\n\nTest resume training without dev set"
python train_model.py --train $TRAIN_PATH --resume --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH --save_every $CP_EVERY --load_checkpoint $(ls -t test_exp/ | head -1)
EX=$((EX+1))
echo $?
ERR=$((ERR+$?)); EX=$((EX+1))

echo "\n\nTest train from checkpoint without dev set"
# Load checkpoint
python train_model.py --train $TRAIN_PATH --output_dir $EXPT_DIR --print_every 50 --epoch $EPOCH --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --load_checkpoint $(ls -t test_exp/ | head -1) --save_every $CP_EVERY --optim sgd
ERR=$((ERR+$?)); EX=$((EX+1))

echo "\n\nTest ponderer"
python train_model.py --train $LOOKUP --dev $LOOKUP --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --pondering --epoch $EPOCH --save_every $CP_EVERY --batch_size 6
ERR=$((ERR+$?)); EX=$((EX+1))

# test with attention
echo "\n\nTest training with pre_rnn attention and LSTM cell"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --attention 'pre-rnn' --attention_method 'dot' --epoch $EPOCH --save_every $CP_EVERY --teacher_forcing_ratio 1
ERR=$((ERR+$?)); EX=$((EX+1))

echo "\n\nTest training with pre-rnn attention and GRU cell method mlp"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL2 --attention 'pre-rnn' --epoch $EPOCH --save_every $CP_EVERY --attention_method 'dot'
ERR=$((ERR+$?)); EX=$((EX+1))

echo "\n\nTest training with post-rnn attention and LSTM cell"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --attention 'post-rnn' --attention_method 'dot' --epoch $EPOCH --save_every $CP_EVERY --teacher_forcing_ratio 0.5
ERR=$((ERR+$?)); EX=$((EX+1))

# test attention loss
echo "\n\nTest training with attention loss"
python train_model.py --train $LOOKUP --dev $LOOKUP --monitor $LOOKUP --output_dir $EXPT_DIR --print_every 1 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --attention 'pre-rnn' --attention_method 'mlp' --epoch $EPOCH --save_every $CP_EVERY --teacher_forcing_ratio 0.5 --use_attention_loss --batch_size=7
ERR=$((ERR+$?)); EX=$((EX+1))

# test attention loss and pondering
echo "\n\nTest training with attention loss and ponderer"
python train_model.py --train $LOOKUP --dev $LOOKUP --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --attention 'pre-rnn' --attention_method 'mlp' --epoch $EPOCH --save_every $CP_EVERY --teacher_forcing_ratio 0.5 --use_attention_loss --pondering --batch_size=6
ERR=$((ERR+$?)); EX=$((EX+1))

# test full focus
echo "\n\nTest training with full focus"
python train_model.py --train $LOOKUP --dev $LOOKUP --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --attention 'pre-rnn' --attention_method 'mlp' --epoch $EPOCH --save_every $CP_EVERY --teacher_forcing_ratio 0.5 --use_attention_loss --batch_size=7 --full_focus True
ERR=$((ERR+$?)); EX=$((EX+1))

# test bidirectional
echo "\n\nTest bidirectional model"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --bidirectional --epoch $EPOCH --save_every $CP_EVERY
ERR=$((ERR+$?)); EX=$((EX+1))

# test bidirectional with attention at timestep t
echo "\n\nTest bidirectional model with attention at timestep t"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --attention 'post-rnn' --attention_method 'dot' --bidirectional --epoch $EPOCH --save_every $CP_EVERY
ERR=$((ERR+$?)); EX=$((EX+1))
 
# test bidirectional with attention at timestep t-1
echo "\n\nTest bidirectional model with attention at timestep t-1"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --attention 'pre-rnn' --attention_method 'mlp' --bidirectional --epoch $EPOCH --save_every $CP_EVERY
ERR=$((ERR+$?)); EX=$((EX+1))

# test input optimizer
echo "\n\nTest command line optimizer"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH --optim adagrad --save_every $CP_EVERY
ERR=$((ERR+$?)); EX=$((EX+1))

# test encoder dropout
echo "\n\nTest encoder dropout"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH --save_every $CP_EVERY --dropout_p_encoder 0.5
ERR=$((ERR+$?)); EX=$((EX+1))

# test decoder dropout
echo "\n\nTest decoder dropout"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH --save_every $CP_EVERY --dropout_p_decoder 0.5
ERR=$((ERR+$?)); EX=$((EX+1))

# test n_layers
echo "\n\nTest multiple layers"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH --save_every $CP_EVERY --n_layers 2
ERR=$((ERR+$?)); EX=$((EX+1))

# test n_layers
echo "\n\nTest multiple layers with pre-rnn attention"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH --save_every $CP_EVERY --n_layers 3 --attention 'pre-rnn'
ERR=$((ERR+$?)); EX=$((EX+1))

# test n_layers
echo "\n\nTest multiple layers with pre-rnn attention"
python train_model.py --train $TRAIN_PATH --dev $DEV_PATH --output_dir $EXPT_DIR --print_every 50 --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --epoch $EPOCH --save_every $CP_EVERY --n_layers 3 --attention 'post-rnn'
ERR=$((ERR+$?)); EX=$((EX+1))

echo "\n\n\n$EX tests executed, $ERR tests failed\n\n"

rm -r $EXPT_DIR
