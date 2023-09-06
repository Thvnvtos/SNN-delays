from spikingjelly.activation_based import surrogate

class Config:
    
    ################################################
    #            General configuration             #
    ################################################
    debug = False

    # dataset could be set to either 'shd', 'ssc' or 'gsc', change datasets_path accordingly.
    dataset = 'shd'                    
    datasets_path = 'Datasets/SHD'

    seed = 0

    # model type could be set to : 'snn_delays' |  'snn_delays_lr0' |  'snn'
    model_type = 'snn_delays'          
    

    time_step = 10
    n_bins = 5

    epochs = 150
    batch_size = 256

    ################################################
    #               Model Achitecture              #
    ################################################
    spiking_neuron_type = 'lif'         # plif, lif
    init_tau = 10.05                    # in ms, can't be < time_step

    stateful_synapse_tau = 10.0        # in ms, can't be < time_step
    stateful_synapse = False
    stateful_synapse_learnable = False

    n_inputs = 700//n_bins
    n_hidden_layers = 2
    n_hidden_neurons = 256 
    n_outputs = 20 if dataset == 'shd' else 35

    sparsity_p = 0

    dropout_p = 0.4
    use_batchnorm = True
    bias = False
    detach_reset = True

    loss = 'sum'           # 'mean', 'max', 'spike_count', 'sum
    loss_fn = 'CEloss'
    output_v_threshold = 2.0 if loss == 'spike_count' else 1e9  #use 1e9 for loss = 'mean' or 'max'

    v_threshold = 1.0
    alpha = 5.0
    surrogate_function = surrogate.ATan(alpha = alpha)#FastSigmoid(alpha)

    init_w_method = 'kaiming_uniform'

    init_tau = (init_tau  +  1e-9) / time_step
    stateful_synapse_tau = (stateful_synapse_tau  +  1e-9) / time_step
    ################################################
    #                Optimization                  #
    ################################################
    optimizer_w = 'adam'
    optimizer_pos = 'adam'

    weight_decay = 1e-5

    lr_w = 1e-3
    lr_pos = 100*lr_w   if model_type =='snn_delays' else 0
    
    # 'one_cycle', 'cosine_a', 'none'
    scheduler_w = 'one_cycle'    
    scheduler_pos = 'cosine_a'   if model_type =='snn_delays' else 'none'


    # for one cycle
    max_lr_w = 5 * lr_w
    max_lr_pos = 5 * lr_pos


    # for cosine annealing
    t_max_w = epochs
    t_max_pos = epochs

    ################################################
    #                    Delays                    #
    ################################################
    DCLSversion = 'gauss' if model_type =='snn_delays' else 'max'
    decrease_sig_method = 'exp'
    kernel_count = 1

    max_delay = 250//time_step
    max_delay = max_delay if max_delay%2==1 else max_delay+1 # to make kernel_size an odd number
    
    # For constant sigma without the decreasing policy, set model_type == 'snn_delays' and sigInit = 0.23 and final_epoch = 0
    sigInit = max_delay // 2        if model_type == 'snn_delays' else 0
    final_epoch = (1*epochs)//4     if model_type == 'snn_delays' else 0


    left_padding = max_delay-1
    right_padding = (max_delay-1) // 2

    init_pos_method = 'uniform'
    init_pos_a = -max_delay//2
    init_pos_b = max_delay//2

    ################################################
    #                 Fine-tuning                  #
    ################################################
    # BELOW IS NOT USED, NEED TO UPDATE

    lr_w_finetuning = 1e-4
    max_lr_w_finetuning = 1.2 * lr_w_finetuning

    dropout_p_finetuning = 0
    stateful_synapse_learnable_finetuning = False
    spiking_neuron_type_finetuning = 'lif'
    epochs_finetuning = 30


    ################################################
    #               Data-Augmentation              #
    ################################################

    augment = False

    rnoise_sig = 0

    TN_mask_aug_proba = 0.65
    time_mask_size = max_delay//3
    neuron_mask_size = n_inputs//5

    cutmix_aug_proba = 0.5


    #############################################
    #                      Wandb                #
    #############################################
    # If use_wand is set to True, specify your wandb api token in wandb_token and the project and run names. 

    use_wandb = False
    wandb_token = 'your_wandb_token'
    wandb_project_name = 'Wandb Project Name'

    run_name = 'Wandb Run Name'


    run_info = f'||{model_type}||{dataset}||{time_step}ms||bins={n_bins}'

    wandb_run_name = run_name + f'||seed={seed}' + run_info
    wandb_group_name = run_name + run_info

    # REPL is going to be replaced with best_acc or best_loss for best model according to validation accuracy or loss
    save_model_path = f'{wandb_run_name}_REPL.pt'


    wandb_run_name_finetuning = wandb_run_name.replace('(Pre-train)', 
                                       f'(Fine-tune_lr={lr_w_finetuning:.1e}->{max_lr_w_finetuning:.1e}_dropout={dropout_p_finetuning}_{spiking_neuron_type_finetuning}_SS={stateful_synapse_learnable_finetuning})')
    wandb_group_name_finetuning = wandb_group_name.replace('(Pre-train)', '(Fine-tune)')

    save_model_path_finetuning = f'{wandb_run_name_finetuning}.pt'