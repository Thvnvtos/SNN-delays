from spikingjelly.activation_based import surrogate

class Config:
    
    ################################################
    #            General configuration             #
    ################################################
    debug = False
    datasets_path = '../Datasets'

    seed = 0

    model_type = 'snn_delays'          # 'ann', 'snn', 'snn_delays'
    dataset = 'shd'             # 'shd', 'ssc'

    epochs = 10
    batch_size = 128
    time_step = 50

    
    n_bins = 70
    n_inputs = 700//n_bins

    n_hidden_layers = 2
    n_hidden_neurons = 128
    
    n_outputs = 20 if dataset == 'shd' else 35
    
    use_batchnorm = True
    dropout_p = 0
    bias = False

    lr_w = 1e-3
    lr_pos = 0

    optimizer_w = 'adam'
    optimizer_pos = 'adam'

    loss = 'mean'
    loss_fn = 'CEloss'
    
    init_w_method = 'kaiming_uniform'

    #############################
    #           SNN             #
    #############################

    spiking_neuron_type = 'lif'
    init_tau = 2.0
    v_threshold = 1.0
    
    alpha = 3.0
    surrogate_function = surrogate.ATan(alpha = alpha)#FastSigmoid(alpha)
    detach_reset = True
    kernel_count = 1
    DCLSversion = 'gauss'

    max_delay = 100//time_step
    max_delay = max_delay if max_delay%2==1 else max_delay+1 # to make kernel_size an odd number

    left_padding = max_delay-1
    right_padding = 0

    output_v_threshold = 1e9 # use 1e9 for loss = 'mean' or 'max'

    #############################
    #           Wandb           #
    #############################

    use_wandb = False

    wandb_project_name = 'project'
    wandb_run_name = ''


