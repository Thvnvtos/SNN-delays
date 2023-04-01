from spikingjelly.activation_based import surrogate

class Config:
    
    ################################################
    #            General configuration             #
    ################################################
    debug = False
    datasets_path = '../Datasets'

    seed = 0

    model_type = 'snn_delays_lr0'          # 'ann', 'snn', 'snn_delays' 'snn_delays_lr0'
    dataset = 'shd'             # 'shd', 'ssc'

    time_step = 20
    n_bins = 10

    epochs = 50
    batch_size = 256

    ################################################
    #               Model Achitecture              #
    ################################################
    spiking_neuron_type = 'lif'
    init_tau = 1.5

    n_inputs = 700//n_bins
    n_hidden_layers = 2
    n_hidden_neurons = 64
    n_outputs = 20 if dataset == 'shd' else 35
    
    dropout_p = 0
    use_batchnorm = True
    bias = False
    detach_reset = True

    loss = 'mean'
    loss_fn = 'CEloss'
    output_v_threshold = 1e9 #use 1e9 for loss = 'mean' or 'max'

    v_threshold = 1.0
    alpha = 3.0
    surrogate_function = surrogate.ATan(alpha = alpha)#FastSigmoid(alpha)

    init_w_method = 'kaiming_uniform'

    ################################################
    #                Optimization                  #
    ################################################
    optimizer_w = 'adam'
    optimizer_pos = 'adam'

    lr_w = 1e-3
    lr_pos = 0
    
    scheduler_w = 'one_cycle'    # 'one_cycle', 'cosine_a'
    scheduler_pos = 'one_cycle'

    # for one cycle
    max_lr_w = 4
    max_lr_pos = 2

    # for cosine annealing
    t_max_w = epochs
    t_max_pos = epochs

    ################################################
    #                    Delays                    #
    ################################################
    DCLSversion = 'gauss'
    kernel_count = 1

    sigInit = 0.5
    max_delay = 500//time_step
    max_delay = max_delay if max_delay%2==1 else max_delay+1 # to make kernel_size an odd number

    left_padding = max_delay-1
    right_padding = (max_delay-1) // 2

    init_pos_method = 'uniform'
    init_pos_a = -max_delay//2 
    init_pos_b = max_delay//2

    #############################
    #           Wandb           #
    #############################
    use_wandb = False

    wandb_project_name = 'project'
    wandb_run_name = ''


