import deepSI
import numpy as np
from datetime import datetime
from torch import nn
from deepSI import fit_systems
import os
import time
from f1tenth_augmentation.utils import plot_losses, load_system_data, add_noise_to_output


if __name__ == "__main__":
    # NOISE SETTING
    noise_level = 25  # [dB]


    # Parameters for the encoder:
    T = 15  # Truncation time
    epochs = 1000
    batch_size = 256
    n_nodes_per_layer_e = 64
    n_hidden_layers_e = 2
    n_nodes_per_layer_din = 64  # for h-net and f-net
    n_hidden_layers_din = 2  # for h-net and f-net
    activation = nn.Tanh  # for all networks
    nf = 12  # minimum criteria: (n > n_x - 1)  recommended: n_x*2
    na = nf  # no. inputs to the e-net to estim. x
    nb = nf  # no. outputs for the e-net to estim. x
    nu = 2  # no. inputs of the system
    ny = 3  # no. outputs of the system
    nx = 6  # number of states in the subnet structure
    tau = 50  # normalization factor for continuous identification [s]

    learning_rate = 1e-3

    # Data concatenation:
    cwd = os.path.dirname(__file__)
    save_name = f'../results/bb_models/SUBNET_' + (datetime.now()).strftime("%Y%m%d-T%H%M%S")
    saveFolder = os.path.join(cwd, save_name)
    os.mkdir(saveFolder)

    # load data
    data_file_path = os.path.join(cwd, "..", "data")
    train_data = load_system_data(os.path.join(data_file_path, "f1tenth_sim_train.npz"))
    valid_data = load_system_data(os.path.join(data_file_path, "f1tenth_sim_valid.npz"))
    test_data = load_system_data(os.path.join(data_file_path, "f1tenth_sim_test.npz"))

    # adding noise to train + valid. sets
    train_data, valid_data = add_noise_to_output(train_data, valid_data, noise_level)

    # Initialization:
    e_net = fit_systems.encoders.default_encoder_net
    f_net = fit_systems.encoders.default_state_net
    h_net = fit_systems.encoders.default_output_net
    sys_ss_encoder = deepSI.fit_systems.SS_encoder_general(nx=nx, na=na, nb=nb, e_net=e_net, f_net=f_net, h_net=h_net,
                                                            e_net_kwargs=dict(n_nodes_per_layer=n_nodes_per_layer_e, n_hidden_layers=n_hidden_layers_e, activation=activation),
                                                            f_net_kwargs=dict(n_nodes_per_layer=n_nodes_per_layer_din, n_hidden_layers=n_hidden_layers_din, activation=activation),
                                                            h_net_kwargs=dict(n_nodes_per_layer=n_nodes_per_layer_din, n_hidden_layers=n_hidden_layers_din, activation=activation))

    # Training:
    print(f"Training DT SUBNET.")
    startTime = time.time()
    sys_ss_encoder.fit(train_sys_data=train_data, val_sys_data=valid_data, epochs=epochs, batch_size=batch_size, loss_kwargs=dict(nf=T), auto_fit_norm=True, optimizer_kwargs=dict(lr=learning_rate))#, validation_measure='20-step-NRMS')
    trainTime = time.time() - startTime

    # Testing:
    test_sim_encoder = sys_ss_encoder.apply_experiment(test_data, save_state=False)

    RMS_test = test_sim_encoder.RMS(test_data)
    NRMS_test = test_sim_encoder.NRMS(test_data)
    print(f'RMS Test simulation SS encoder {RMS_test:.2}')
    print(f'NRMS Test simulation SS encoder {NRMS_test:.2%}')

    print(f'Training finished in {trainTime:.2} seconds')

    # Plots:
    plot_losses(sys_ss_encoder, block=True)

    sys_ss_encoder.checkpoint_load_system(name='_best')
    sys_ss_encoder.save_system(saveFolder + '/best.pt')
    sys_ss_encoder.checkpoint_load_system(name='_last')
    sys_ss_encoder.save_system(saveFolder + '/last.pt')
