
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
    n_hidden_neurons = 128
    n_hidden_layers = 2
    n_outputs = 20
    
    use_batchnorm = True
    dropout_p = 0

    lr = 1e-3
    
    #############################
    #           Wandb           #
    #############################

    use_wandb = True

    wandb_project_name = 'project'
    wandb_run_name = ''


