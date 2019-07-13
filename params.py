"""Parameters for HAR"""

# Parameter for source data
src_model_trained = True

# Parameter for target data
tgt_model_trained = True

# Parameter for encoder
# opp
lstm_num_layers_opp = 2
lstm_hidden_size_opp = 64
fc1_size_opp = 64
fc2_size_opp = 64
# pcmap2
lstm_num_layers_pamap2 = 2
lstm_hidden_size_pamap2 = 50
fc1_size_pamap2 = 50
fc2_size_pamap2 = 25

## Parameter for discriminator
#discriminator_input_size = lstm_hidden_size
#discriminator_hidden_size = 50
#discriminator_output_size = 2

# Learning rates
src_ec_learning_rate = 1e-3
tgt_ec_learning_rate = 1e-3
#ec_learning_rate = 5e-3
#d_learning_rate = 1e-3

# Optimizer parameters
beta1 = 0.9
beta2 = 0.999

# Number of epochs
src_num_epochs_pre = 500
tgt_num_epochs_pre = 200
num_epochs = 120
src_num_epochs = 10

# Number of evaluation steps
src_eval_step_pre = 10
tgt_eval_step_pre = 10
src_save_step = src_num_epochs

# Loss function parameters
alpha = 0

# l21-norm balance parameter
lambda_train = 1e-2

# Parameter for model saving
model_weight_dir = "./model_weight/"

# Parameters for Laplacian matrix
num_nn = 5
sigma = 1
normed = False