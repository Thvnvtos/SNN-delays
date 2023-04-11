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
    batch_size = 64

    ################################################
    #               Model Achitecture              #
    ################################################
    spiking_neuron_type = 'lif'         # plif, lif
    init_tau = 2.0

    n_inputs = 700//n_bins
    n_hidden_layers = 2
    n_hidden_neurons = 128
    n_outputs = 20 if dataset == 'shd' else 35
    
    dropout_p = 0.1
    use_batchnorm = True
    bias = False
    detach_reset = False

    loss = 'sum'           # 'mean', 'max', 'spike_count', 'sum
    loss_fn = 'CEloss'
    output_v_threshold = 2.0 if loss == 'spike_count' else 1e9  #use 1e9 for loss = 'mean' or 'max'

    v_threshold = 1.0
    alpha = 5.0
    surrogate_function = surrogate.ATan(alpha = alpha)#FastSigmoid(alpha)

    init_w_method = 'kaiming_uniform'

    ################################################
    #                Optimization                  #
    ################################################
    optimizer_w = 'adam'
    optimizer_pos = 'adam'

    lr_w = 1e-2
    lr_pos = 250*lr_w
    
    scheduler_w = 'one_cycle'    # 'one_cycle', 'cosine_a'
    scheduler_pos = 'cosine_a'

    # for one cycle
    max_lr_w = 1.5 * lr_w
    max_lr_pos = 5 * lr_pos

    # for cosine annealing
    t_max_w = epochs
    t_max_pos = epochs

    ################################################
    #                    Delays                    #
    ################################################
    DCLSversion = 'gauss'
    decrease_sig_method = 'exp'
    kernel_count = 1

    max_delay = 200//time_step
    max_delay = max_delay if max_delay%2==1 else max_delay+1 # to make kernel_size an odd number
    
    sigInit = 0.5#max_delay // 3
    final_epoch = 0#(4*epochs)//5


    left_padding = max_delay-1
    right_padding = (max_delay-1) // 2

    init_pos_method = 'uniform'
    init_pos_a = -max_delay//2
    init_pos_b = max_delay//2

    #############################
    #           Wandb           #
    #############################
    use_wandb = True

    wandb_project_name = 'SHD-BestACC'
    wandb_run_name = f'SOTA||{dataset}||{model_type}||{loss}||seed={seed}||MaxDelay={max_delay}'

    wandb_group_name = f"Tests {model_type}"


