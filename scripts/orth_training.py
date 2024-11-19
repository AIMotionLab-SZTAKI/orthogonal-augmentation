from datetime import datetime
import os
from torch import nn
import torch
from matplotlib import pyplot as plt
from f1tenth_augmentation.utils import plot_losses, load_system_data, add_noise_to_output
from f1tenth_augmentation.car_models import nonlinearCar
from model_augmentation.torch_nets import simple_res_net
from model_augmentation.augmentation_structures import get_augmented_fitsys, SSE_AdditiveAugmentation
from model_augmentation.utils import calculate_orthogonalisation
import numpy as np


# USER DEFINITIONS
bParmTune = True
beta = 1e-7
noise_level = 0  # [dB] (noise = 0 for no noise)
results_folder = 'results/orthogonalization'


cwd = os.path.dirname(__file__)
if bParmTune:
    save_name = f'../{results_folder}/SNR{noise_level}_param_tuning_beta{beta}_' + (datetime.now()).strftime("%Y%m%d-T%H%M%S")
else:
    save_name = f'../{results_folder}/SNR{noise_level}_nominal_params_' + (datetime.now()).strftime("%Y%m%d-T%H%M%S")
saveFolder = os.path.join(cwd, save_name)
os.mkdir(saveFolder)

Ts = 0.025  # sampling time
T = 15  # truncation time
epoch = 1500
batch_size = 256
lr = 1e-3


# hyperparameters of the augmentation net
n_in = 5  # states + inputs
n_out = 3  # states
n_hidden_layers = 2  # no. of hidden layers
n_nodes_per_layer = 64  # no. nodes per layer
n_act = nn.Tanh  # activation function

# load data
data_file_path = os.path.join(cwd, "..", "data")
train_data = load_system_data(os.path.join(data_file_path, "f1tenth_sim_train.npz"))
valid_data = load_system_data(os.path.join(data_file_path, "f1tenth_sim_valid.npz"))
test_data = load_system_data(os.path.join(data_file_path, "f1tenth_sim_test.npz"))

# adding noise to train + valid. sets
train_data, valid_data = add_noise_to_output(train_data, valid_data, noise_level)

# neural net for augmentation
ffw_net = simple_res_net(n_in=n_in, n_out=n_out, n_hidden_layers=n_hidden_layers, n_nodes_per_layer=n_nodes_per_layer,
                         activation=n_act)

# Dynamical models of the car
bb_model = nonlinearCar(nx=n_out, ny=n_out, nu=2, ts=Ts, parmTune=bParmTune)

# Initialization
fit_sys = get_augmented_fitsys(augmentation_structure=SSE_AdditiveAugmentation, known_system=bb_model, neur_net=ffw_net,
                               e_net=None, regLambda=0, orthLambda=beta, norm_data=train_data, norm_x0_meas=True)
fit_sys.init_model(test_data, auto_fit_norm=False)

if beta != 0 and bParmTune:
    U_orth, X, U = calculate_orthogonalisation(sys=bb_model, train_data=train_data, x_meas=True)
    loss_kwargs = dict(nf=T, online_construct=False, U1_orth=U_orth, X=X, U=U)
else:
    loss_kwargs = dict(nf=T, online_construct=False)

# Training
print('Training the augmented model...')
fit_sys.fit(train_sys_data=train_data, epochs=epoch, val_sys_data=valid_data, batch_size=batch_size,
            loss_kwargs=loss_kwargs, optimizer_kwargs=dict(lr=lr))

# Training losses
plot_losses(fit_sys, saveFolder)

# Testing
print('Applying an experiment...')
fit_sys.checkpoint_load_system(name='_best')
test_augmented_model = fit_sys.apply_experiment(test_data)

if bb_model.parm_corr_enab:
    P = fit_sys.hfn.sys.P.detach().numpy()
    parm_corr_data = {
        "beta (orthogonalization)": beta,
        "m": P[0],
        "Jz": P[1],
        "lr": P[2],
        "lf": P[3],
        "Cm1": P[4],
        "Cm2": P[5],
        "Cm3": P[6],
        "Cr": P[7],
        "Cf": P[8]
    }
    with torch.no_grad():
        bb_model.P.data = bb_model.P_orig

test_fp_model = bb_model.apply_experiment(test_data, x0_meas=True)

fig1, ax1 = plt.subplots(1, 3, layout="tight")
ax1[0].plot(test_data.y[:, 0], "k")
ax1[0].plot(test_data.y[:, 0] - test_fp_model.y[:, 0], '-.b')
ax1[0].plot(test_data.y[:, 0] - test_augmented_model.y[:, 0], "--r")
ax1[0].legend(["MuJoCo sim.", "FP model error", "Augmented model error"])
ax1[0].set_xlabel("Sim index")
ax1[0].set_ylabel("Longitud. vel. [m/s]")
ax1[0].set_xlim([0, test_data.y.shape[0]])

ax1[1].plot(test_data.y[:, 1], "k")
ax1[1].plot(test_data.y[:, 1] - test_fp_model.y[:, 1], '-.b')
ax1[1].plot(test_data.y[:, 1] - test_augmented_model.y[:, 1], "--r")
ax1[1].set_xlabel("Sim index")
ax1[1].set_ylabel("Lateral vel. [m/s]")
ax1[1].set_xlim([0, test_data.y.shape[0]])

ax1[2].plot(test_data.y[:, 2], "k")
ax1[2].plot(test_data.y[:, 2] - test_fp_model.y[:, 2], '-.b')
ax1[2].plot(test_data.y[:, 2] - test_augmented_model.y[:, 2], "--r")
ax1[2].set_xlabel("Sim index")
ax1[2].set_ylabel("Ang. vel. [rad/s]")
ax1[2].set_xlim([0, test_data.y.shape[0]])
plt.savefig(saveFolder + '/errors.png')
plt.show(block=False)

print(f"RMS simulation FP model: {test_fp_model.RMS(test_data):.2}")
print(f"RMS simulation augmented model: {test_augmented_model.RMS(test_data):.2}")
print(f"NRMS simulation FP model: {test_fp_model.NRMS(test_data):.2%}")
print(f"NRMS simulation augmented model: {test_augmented_model.NRMS(test_data):.2%}")

fig2, ax2 = plt.subplots(2, 3, layout="tight")
ax2[0, 0].plot(test_data.y[:, 0], "k")
ax2[0, 0].plot(test_fp_model.y[:, 0], "-.b")
ax2[0, 0].set_xlabel("Sim index")
ax2[0, 0].set_ylabel("Longitud. vel. [m/s]")
ax2[0, 0].legend(["MuJoCo", "FP model"])
ax2[0, 0].set_xlim(0, test_data.y.shape[0])

ax2[0, 1].plot(test_data.y[:, 1], "k")
ax2[0, 1].plot(test_fp_model.y[:, 1], "-.b")
ax2[0, 1].set_xlabel("Sim index")
ax2[0, 1].set_ylabel("Lateral vel. [m/s]")
ax2[0, 1].set_xlim(0, test_data.y.shape[0])

ax2[0, 2].plot(test_data.y[:, 2], "k")
ax2[0, 2].plot(test_fp_model.y[:, 2], "-.b")
ax2[0, 2].set_xlabel("Sim index")
ax2[0, 2].set_ylabel("Ang. vel. [rad/s]")
ax2[0, 2].set_xlim(0, test_data.y.shape[0])

ax2[1, 0].plot(test_data.y[:, 0], "k")
ax2[1, 0].plot(test_augmented_model.y[:, 0], "--r")
ax2[1, 0].set_xlabel("Sim index")
ax2[1, 0].set_ylabel("Longitud. vel. [m/s]")
ax2[1, 0].legend(["MuJoCo", "Augmented model"])
ax2[1, 0].set_xlim(0, test_data.y.shape[0])

ax2[1, 1].plot(test_data.y[:, 1], "k")
ax2[1, 1].plot(test_augmented_model.y[:, 1], "--r")
ax2[1, 1].set_xlabel("Sim index")
ax2[1, 1].set_ylabel("Lateral vel. [m/s]")
ax2[1, 1].set_xlim(0, test_data.y.shape[0])

ax2[1, 2].plot(test_data.y[:, 2], "k")
ax2[1, 2].plot(test_augmented_model.y[:, 2], "--r")
ax2[1, 2].set_xlabel("Sim index")
ax2[1, 2].set_ylabel("Ang. vel. [rad/s]")
ax2[1, 2].set_xlim(0, test_data.y.shape[0])
plt.savefig(saveFolder + '/outputs.png')
plt.show(block=True)

augmentation_data = {
    "Augmentation": "STATIC ADDITIVE",
    "truncation time": T,
    "epochs": epoch,
    "batch size": batch_size,
    "nodes per layer": n_nodes_per_layer,
    "no. layers": n_hidden_layers,
    "activation function": str(n_act),
    "ANN inputs": n_in,
    "ANN outputs": n_out,
    "Test RMS": test_augmented_model.RMS(test_data),
    "Test NRMS": test_augmented_model.NRMS(test_data),
    "Correction enabled": bb_model.parm_corr_enab
}

text = 'Training properties: \n'
with open(saveFolder + '/info.txt', 'w') as f:
    f.write(text)
    for key, value in augmentation_data.items():
        f.write('%s: %s\n' % (key, value))
    if bb_model.parm_corr_enab:
        for key, value in parm_corr_data.items():
            f.write('%s: %s\n' % (key, value))

# save system
fit_sys.checkpoint_load_system(name='_best')
fit_sys.save_system(saveFolder + '/best.pt')
fit_sys.checkpoint_load_system(name='_last')
fit_sys.save_system(saveFolder + '/last.pt')
