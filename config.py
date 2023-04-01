
class Config:
    
    ################################################
    #            General configuration             #
    ################################################
    debug = False
    datasets_path = '../Datasets'

    seed = 0

    model_type = 'ann'          # 'ann', 'snn', 'snn_delays'
    dataset = 'shd'             # 'shd', 'ssc'

    epochs = 10
    batch_size = 128
    time_step = 50

    
    n_bins = 70
    n_inputs = 700//n_bins

    n_hidden_layers = 1
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
    #           Wandb           #
    #############################

    use_wandb = True

    wandb_project_name = 'project'
    wandb_run_name = ''


