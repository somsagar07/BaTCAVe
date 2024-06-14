import json
import h5py
import numpy as np
import os
import robomimic
import robomimic.utils.file_utils as FileUtils
from captum.concept import TCAV, BTCAV
# the dataset registry can be found at robomimic/__init__.py
from robomimic import DATASET_REGISTRY

import numpy as np

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.test_utils as TestUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.train_utils as TrainUtils
from robomimic.utils.dataset import SequenceDataset

from robomimic.config import config_factory
from robomimic.algo import algo_factory
import robomimic.utils.env_utils as EnvUtils

from robomimic.algo import RolloutPolicy
from robomimic.utils.train_utils import run_rollout
import imageio

import urllib.request


# set download folder and make it
WS_DIR = "robomimic_data"
download_folder = WS_DIR + "/robomimic_data/"
os.makedirs(download_folder, exist_ok=True)

# download the dataset
task = "can" # lift : Lift Cube, square : Nut Assembly, can : Pick Place Can
dataset_type = "ph"
hdf5_type = "low_dim"
FileUtils.download_url(
    url=DATASET_REGISTRY[task][dataset_type][hdf5_type]["url"],
    download_dir=download_folder,
)

# enforce that the dataset exists
dataset_path = os.path.join(download_folder, "low_dim_v141.hdf5")
assert os.path.exists(dataset_path)


def get_example_model(dataset_path, device):
    """
    Use a default config to construct a BC model.
    """

    # default BC config
    config = config_factory(algo_name="bc")

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # read dataset to get some metadata for constructing model
    # all_obs_keys determines what observations we will feed to the policy
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path,
        all_obs_keys=sorted((
            "robot0_eef_pos",  # robot end effector position
            "robot0_eef_quat",   # robot end effector rotation (in quaternion)
            "robot0_gripper_qpos",   # parallel gripper joint position
            "object",  # object information
        )),
    )

    # make BC model
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    return model

device = TorchUtils.get_torch_device(try_to_use_cuda=True)
model = get_example_model(dataset_path, device=device)

print(model)

dataset = SequenceDataset(
    hdf5_path=dataset_path,
    obs_keys=(                      # observations we want to appear in batches
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "object",
    ),
    dataset_keys=(                  # can optionally specify more keys here if they should appear in batches
        "actions",
        "rewards",
        "dones",
    ),
    load_next_obs=True,
    frame_stack=1,
    seq_length=10,                  # length-10 temporal sequences
    pad_frame_stack=True,
    pad_seq_length=True,            # pad last obs per trajectory to ensure all sequences are sampled
    get_pad_mask=False,
    goal_mode=None,
    hdf5_cache_mode="all",          # cache dataset in memory to avoid repeated file i/o
    hdf5_use_swmr=False,
    hdf5_normalize_obs=False,
    filter_by_attribute=None,       # can optionally provide a filter key here
)
print("\n============= Created Dataset =============")
print(dataset)
print("")

"""
WARNING: This code snippet is only for instructive purposes, and is missing several useful
         components used during training such as logging and rollout evaluation.
"""
def get_data_loader(dataset_path):
    """
    Get a data loader to sample batches of data.
    Args:
        dataset_path (str): path to the dataset hdf5
    """
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=(                      # observations we want to appear in batches
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        ),
        dataset_keys=(                  # can optionally specify more keys here if they should appear in batches
            "actions",
            "rewards",
            "dones",
        ),
        load_next_obs=True,
        frame_stack=1,
        seq_length=10,                  # length-10 temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,            # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode="all",          # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        filter_by_attribute=None,       # can optionally provide a filter key here
    )
    print("\n============= Created Dataset =============")
    print(dataset)
    print("")

    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,       # no custom sampling logic (uniform sampling)
        batch_size=100,     # batches of size 100
        shuffle=True,
        num_workers=0,
        drop_last=True      # don't provide last batch in dataset pass if it's less than 100 in size
    )
    return data_loader


def run_train_loop(model, data_loader, num_epochs=50, gradient_steps_per_epoch=100):
    """
    Note: this is a stripped down version of @TrainUtils.run_epoch and the train loop
    in the train function in train.py. Logging and evaluation rollouts were removed.
    Args:
        model (Algo instance): instance of Algo class to use for training
        data_loader (torch.utils.data.DataLoader instance): torch DataLoader for
            sampling batches
    """
    # ensure model is in train mode
    model.set_train()

    epoch_losses = []

    for epoch in range(1, num_epochs + 1): # epoch numbers start at 1

        # iterator for data_loader - it yields batches
        data_loader_iter = iter(data_loader)

        # record losses
        losses = []

        for _ in range(gradient_steps_per_epoch):

            # load next batch from data loader
            try:
                batch = next(data_loader_iter)
            except StopIteration:
                # data loader ran out of batches - reset and yield first batch
                data_loader_iter = iter(data_loader)
                batch = next(data_loader_iter)

            # process batch for training
            input_batch = model.process_batch_for_training(batch)

            # forward and backward pass
            info = model.train_on_batch(batch=input_batch, epoch=epoch, validate=False)

            # record loss
            step_log = model.log_info(info)
            losses.append(step_log["Loss"])

        # do anything model needs to after finishing epoch
        model.on_epoch_end(epoch)

        epoch_loss = np.mean(losses)
        epoch_losses.append(epoch_loss)

        print("Train Epoch {}: Loss {}".format(epoch, np.mean(losses)))

        epoch_loss = np.mean(losses)
        epoch_losses.append(epoch_loss)

        print("Train Epoch {}: Loss {}".format(epoch, np.mean(losses)))

    return epoch_losses
        

data_loader = get_data_loader(dataset_path=dataset_path)

# run training loop
epoch_losses = run_train_loop(model=model, data_loader=data_loader, num_epochs=500, gradient_steps_per_epoch=100)




# create simulation environment
env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)

env = EnvUtils.create_env_from_metadata(
    env_meta=env_meta,
    env_name=env_meta["env_name"],
    render=False,
    render_offscreen=True,
    use_image_obs=False,
)

# create a thin wrapper around the model to interact with the environment
policy = RolloutPolicy(model)

# create a video writer
video_path = "rollout.mp4"
video_writer = imageio.get_writer(video_path, fps=10)

# run rollout
rollout_log = run_rollout(
    policy=policy,
    env=env,
    horizon=1000,
    video_writer=video_writer,
    render=False,
    terminate_on_success=True
)

video_writer.close()
# print rollout results
print(rollout_log)


##########################################################
# Explainability 

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import IterableDataset
from typing import Iterator

from captum.concept._utils.data_iterator import dataset_to_dataloader
from captum.concept import TCAV
from captum.concept import Concept
from captum.attr import LayerGradientXActivation, LayerIntegratedGradients
from torchvision import transforms
import matplotlib.pyplot as plt
from captum.concept._utils.common import concepts_to_str
import pandas as pd


df = pd.read_json('actions.json', orient='records', lines=True) # replace with exp action
df2 = pd.read_json('observations.json', orient='records', lines=True) # replace with obs action

# Define a function to adjust the keys
def adjust_key_names(loaded_dict):
    new_dict = {}
    for key, value in loaded_dict.items():
        new_key = key
        # Adjust the keys by removing prefix and adjusting to new model's layer names
        new_key = new_key.replace('nets.mlp._model.', 'mlp.mlp.')  # Adjust MLP keys
        new_key = new_key.replace('nets.decoder.nets.action.', 'decoder.decoder.')  # Adjust Decoder keys

        new_dict[new_key] = value
    return new_dict

class ObservationEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ObservationEncoder, self).__init__()
        # Example: If the encoder is a pass-through, we may not need any parameters.
        # This is a placeholder - the actual logic will depend on the original encoder's function.

    def forward(self, x):
        # Pass through without transformation if that's what the original does
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, layer_dims):
        super(MLP, self).__init__()
        layers = []
        current_dim = input_dim
        for dim in layer_dims:
            layers.append(nn.Linear(current_dim, dim))
            layers.append(nn.ReLU())
            current_dim = dim
        layers.append(nn.Linear(current_dim, output_dim))
        layers.append(nn.ReLU())  # Assuming there's always a ReLU after the last layer
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class ObservationDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ObservationDecoder, self).__init__()
        self.decoder = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.decoder(x)

class ActorNetwork(nn.Module):
    def __init__(self, obs_input_dims, action_dim, encoder_output_dim=23, mlp_layer_dims=(1024,)):
        super(ActorNetwork, self).__init__()
        self.encoder = ObservationEncoder(sum(obs_input_dims), encoder_output_dim)
        self.mlp = MLP(encoder_output_dim, 1024, mlp_layer_dims)
        self.decoder = ObservationDecoder(1024, action_dim)

    def forward(self, obs):
        encoded_obs = self.encoder(obs)
        mlp_output = self.mlp(encoded_obs)
        actions = self.decoder(mlp_output)
        return actions


tcav_model = ActorNetwork(obs_input_dims=[14, 3, 4, 2], action_dim=7)
loaded_weights = model.nets["policy"].state_dict()

adjusted_weights = adjust_key_names(loaded_weights)
tcav_model.load_state_dict(adjusted_weights, strict=True)

# Save the model weights
torch.save(tcav_model.state_dict(), 'tcav_model_weights.pth')

def transform(x):
    # Convert a NumPy array to a PyTorch tensor
    return torch.tensor(x, dtype=torch.float32)

# Custom class to handle image dataset
class CustomEnvDataset(IterableDataset):
    def __init__(self, env_list, transform):
        self.env_list = env_list
        self.transform = transform

    def __iter__(self) -> Iterator[Tensor]:
        for env_data in self.env_list:
            yield self.transform(env_data)

def assemble_concept_env(concept_name, id, envs, transform):
    dataset = CustomEnvDataset(envs, transform)
    concept_iter = dataset_to_dataloader(dataset)
    return Concept(id=id, name=concept_name, data_iter=concept_iter)



from sklearn import linear_model
from sklearn.model_selection import train_test_split
from captum.concept._utils.classifier import Classifier
import shutil
from batcave.BayesianLogistic import VBLogisticRegression

class CustomClassifier(Classifier):
    def __init__(self):
        # self.lm = linear_model.LogisticRegression(max_iter=1000)
        self.lm = VBLogisticRegression(fit_intercept=False)  # We artificially add an intercept below
        self.test_size = 0.33
        self.evaluate_test = False
        self.metrics = None

    def train_and_eval(self, dataloader: DataLoader, **kwargs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs, labels = [], []
        for X, y in dataloader:
            X = torch.cat((torch.ones((X.shape[0], 1), device=device), X.to(device)), dim=1)  # Add the intercept term. This is required only for the cav classifier.
            inputs.append(X)
            labels.append(y.to(device))
        
        # Move tensors to CPU before converting to NumPy
        inputs = torch.cat(inputs).detach().cpu().numpy()
        labels = torch.cat(labels).detach().cpu().numpy()
        
        if self.evaluate_test:
            X_train, X_test, y_train, y_test = train_test_split(inputs, labels, test_size=self.test_size)
        else:
            X_train, y_train = inputs, labels
        
        self.lm.fit(X_train, y_train)

        if self.evaluate_test:
            self.metrics = {'accs': self.lm.score(X_test, y_test)}
            return self.metrics
        self.metrics = {'accs': self.lm.score(X_train, y_train)}
        print(self.metrics)
        return self.metrics

    def weights(self):
        if len(self.lm.coef_) == 1:
            # if there are two concepts, there is only one label.
            # We split it in two.
            return torch.tensor(np.array([-1 * self.lm.coef_[0], self.lm.coef_[0]])).to('cuda')
        else:
            return torch.tensor(self.lm.coef_).to('cuda')

    def classes(self):
        return self.lm.classes_

    def get_metrics(self):
        return self.metrics

classifier = CustomClassifier()

def remove_cav_folder():
    if os.path.exists('cav'):
        shutil.rmtree('cav')

def format_float(f):
    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))

def extract_btcav_scores(experimental_sets, tcav_scores, layers):

    ex_sets, cons, vals = [], [], []
    for idx_es, concepts in enumerate(experimental_sets):

        concepts = experimental_sets[idx_es]
        concepts_key = concepts_to_str(concepts)
        
        for i in range(len(concepts)):
            vals_j = []
            for j in range(len(tcav_scores)):
                val = [format_float(scores['sign_count'][i]) for layer, scores in tcav_scores[j][concepts_key].items()]
                vals_j.append(val)
            ex_sets.append(idx_es)
            cons.append(concepts[i].name)
            vals.append(vals_j)

    return ex_sets, cons, vals


def get_BTCAV(concept,random,input,layer,model,target):
    remove_cav_folder()
    model.to(device)
    mytcav = BTCAV(model=model,
                    layers=layer,
                    classifier=CustomClassifier(),
                    n_samples=500,
                    layer_attr_method = None
                )

    experimental_set_rand = [[concept, random]]
   

    tcav_score = mytcav.interpret(inputs=input,
                                experimental_sets=experimental_set_rand,
                                target=target)
    
    ex_sets, cons, vals = extract_btcav_scores(experimental_set_rand, tcav_score, layer)
    print(vals)
    vals_list = []
    vals_list.append(vals)
    vals_list = np.array(vals_list)[0,:,:,0]

    mean = list(np.mean(vals_list, axis=1))[0]
    std = list(np.std(vals_list, axis=1))[0]
    lb = list(np.percentile(vals_list, 25, axis=1))[0]
    ub = list(np.percentile(vals_list, 75, axis=1))[0]
    
    return mean, std, lb, ub


n_episode = 100
random_concept_data = np.concatenate(
    (np.random.rand(n_episode, 14), np.random.rand(n_episode, 3), np.random.rand(n_episode, 4), np.random.rand(n_episode, 2)), # change 14 to 10 if task is lift
    axis=1
)

def get_score_t(df2, target):
    # for index in range(2):
    index = 2
    object_list = []
    robot0_eef_pos_list = []
    robot0_eef_quat_list = []
    robot0_gripper_qpos_list = []
    res = []

    for i in range(len(df2[f'Observation {index}'])):
        object_list.append(df2[f'Observation {index}'][i]['object'])  # Use .get() to avoid KeyError if the key is missing
        robot0_eef_pos_list.append(df2[f'Observation {index}'][i]['robot0_eef_pos'])
        robot0_eef_quat_list.append(df2[f'Observation {index}'][i]['robot0_eef_quat'])
        robot0_gripper_qpos_list.append(df2[f'Observation {index}'][i]['robot0_gripper_qpos'])

    data_object = torch.tensor(object_list, dtype=torch.float32)
    data_robot0_eef_pos = torch.tensor(robot0_eef_pos_list, dtype=torch.float32)
    data_robot0_eef_quat = torch.tensor(robot0_eef_quat_list, dtype=torch.float32)
    data_robot0_gripper_qpos = torch.tensor(robot0_gripper_qpos_list, dtype=torch.float32)

    object_concept_data = np.concatenate(
            (data_object.cpu().detach().numpy(), np.random.rand(n_episode, 3), np.random.rand(n_episode, 4), np.random.rand(n_episode, 2)),
            axis=1
        )

    robot0_eef_pos_concept_data = np.concatenate(
        (np.random.rand(n_episode, 14), data_robot0_eef_pos.cpu().detach().numpy(), np.random.rand(n_episode, 4), np.random.rand(n_episode, 2)),
        axis=1
    )

    robot0_eef_quat_concept_data = np.concatenate(
        (np.random.rand(n_episode, 14), np.random.rand(n_episode, 3), data_robot0_eef_quat.cpu().detach().numpy(), np.random.rand(n_episode, 2)),
        axis=1
    )

    robot0_gripper_qpos_concept_data = np.concatenate(
        (np.random.rand(n_episode, 14), np.random.rand(n_episode, 3), np.random.rand(n_episode, 4), data_robot0_gripper_qpos.cpu().detach().numpy()),
        axis=1
    )


    object_concept = assemble_concept_env("object", 0, object_concept_data, transform)
    robot0_eef_pos_concept = assemble_concept_env("robot0_eef_pos", 1, robot0_eef_pos_concept_data, transform)
    robot0_eef_quat_concept = assemble_concept_env("robot0_eef_quat", 2, robot0_eef_quat_concept_data, transform)
    robot0_gripper_qpos_concept = assemble_concept_env("robot0_gripper_qpos", 3, robot0_gripper_qpos_concept_data, transform)
    
    random_0_concept = assemble_concept_env("random_0", 4, random_concept_data, transform)


    layers=['decoder']
    
    
    for k in range(50):

        concatenated_data = np.concatenate(
            (torch.stack([data_object[k], data_object[k]]).cpu().detach().numpy(), torch.stack([data_robot0_eef_pos[k],data_robot0_eef_pos[k]]).cpu().detach().numpy(), torch.stack([data_robot0_eef_quat[0],data_robot0_eef_quat[0]]).cpu().detach().numpy(), torch.stack([data_robot0_gripper_qpos[0],data_robot0_gripper_qpos[0]]).cpu().detach().numpy()),
            axis=1
        )
    

        
        obj = get_BTCAV(object_concept,random_0_concept,torch.from_numpy(concatenated_data),layers,tcav_model,target)
        pos = get_BTCAV(robot0_eef_pos_concept,random_0_concept,torch.from_numpy(concatenated_data),layers,tcav_model,target)
        qpos = get_BTCAV(robot0_eef_quat_concept,random_0_concept,torch.from_numpy(concatenated_data),layers,tcav_model,target)
        grip = get_BTCAV(robot0_gripper_qpos_concept,random_0_concept,torch.from_numpy(concatenated_data),layers,tcav_model,target)
        res.append([obj,pos,qpos,grip])
    return res


indx = 1
target = get_score_t(df2,indx)

action1 = []
action2 = []
action3 = []
action4 = []
action5 = []
action6 = []
action7 = []
for i in range(len(df.iloc[0])):
    action1.append(df.iloc[0][i][0])
    action2.append(df.iloc[0][i][1])
    action3.append(df.iloc[2][i][2])
    action4.append(df.iloc[0][i][3])
    action5.append(df.iloc[0][i][4])
    action6.append(df.iloc[0][i][5])
    action7.append(df.iloc[0][i][6])


def get_mean_stds_ofconcept(target,concept):
    means = []
    stds = []
    for i in range(len(target)):
        means.append(target[i][concept][0])
        stds.append(target[i][concept][1])
    return means, stds

def plot_data(heatmap_data, bar_data1, bar_data2, bar_data3, bar_data4, std_data1, std_data2, std_data3, std_data4, name, TCAV_PN=True):
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(5, 2, width_ratios=[0.95, 0.05], height_ratios=[0.5, 0.5, 0.5, 0.5, 0.5])

    # Plot the heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    cax = ax1.matshow(heatmap_data, aspect='auto', cmap='viridis')
    ax1.set_title(name, loc='center', pad=20)
    ax1.set_yticks([])

    # Add a colorbar to the right
    cbar_ax = fig.add_subplot(gs[0, 1])
    colorbar = fig.colorbar(cax, cax=cbar_ax)

    if TCAV_PN:

                # Calculate the 3.4th value of the colorbar
        cbar_min, cbar_max = cax.get_clim()
        threshold_value = cbar_min + 3.4 * (cbar_max - cbar_min) / 4

        # Initialize mask for bars
        mask = np.ones_like(heatmap_data, dtype=bool)

        # Plot '*' on the heatmap where values exceed the threshold
        for (i, j), val in np.ndenumerate(heatmap_data):
            if val >= threshold_value:
                ax1.scatter(j, i, color='black', marker='*',s=10)
                mask[0, j] = False  # Do not mask this column
            else:
                mask[0, j] = True  # Mask this column

    else:
            # Calculate the 15% threshold value of the colorbar
        cbar_min, cbar_max = cax.get_clim()
        # Setting threshold for bottom 15%
        threshold_value = cbar_min + 0.15 * (cbar_max - cbar_min)

        # Initialize mask for bars
        mask = np.ones_like(heatmap_data, dtype=bool)

        # Plot '*' on the heatmap where values are below the threshold (bottom 15%)
        for (i, j), val in np.ndenumerate(heatmap_data):
            if val <= threshold_value:
                ax1.scatter(j, i, color='black', marker='*', s=10)
                mask[i, j] = False  # Set True for bottom 15% values
            else:
                mask[i, j] = True  # Set False for other values


    # Mask bar data
    bar_data1_masked = np.where(mask[0], 0, bar_data1)
    bar_data2_masked = np.where(mask[0], 0, bar_data2)
    bar_data3_masked = np.where(mask[0], 0, bar_data3)
    bar_data4_masked = np.where(mask[0], 0, bar_data4)

    std_data1_masked = np.where(mask[0], 0, std_data1)
    std_data2_masked = np.where(mask[0], 0, std_data2)
    std_data3_masked = np.where(mask[0], 0, std_data3)
    std_data4_masked = np.where(mask[0], 0, std_data4)

    fig.text(0.90, 0.65, 'object', ha='center', va='center', fontsize=20)
    fig.text(0.90, 0.5, 'eef_pos', ha='center', va='center', fontsize=20)
    fig.text(0.90, 0.35, 'eef_quat', ha='center', va='center', fontsize=20)
    fig.text(0.90, 0.2, 'gripper', ha='center', va='center', fontsize=20)

    # Create x-axis labels to match the heatmap columns
    x_labels = range(heatmap_data.shape[1])

    # Plot the bar graphs
    bar_width = 1.0  # Full width for each bar to eliminate gaps
    yticks = [0.5]

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.bar(x_labels, bar_data1_masked, width=bar_width, yerr=std_data1_masked, capsize=2, error_kw={'elinewidth': 0.5}, align='center')
    with open(f'{name}_mean2.txt', 'a') as f:
        f.write('object : ')
        f.write(f'{list(bar_data1_masked)}\n')
    with open(f'{name}_std2.txt', 'a') as f:
        f.write('object : ')
        f.write(f'{list(std_data1_masked)}\n')

    ax2.set_ylim(0, 1)
    ax2.set_yticks(yticks)

    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.bar(x_labels, bar_data2_masked, width=bar_width, yerr=std_data2_masked, capsize=2, error_kw={'elinewidth': 0.5}, align='center')
    with open(f'{name}_mean2.txt', 'a') as f:
        f.write('eef_pos : ')
        f.write(f'{list(bar_data2_masked)}\n')
    with open(f'{name}_std2.txt', 'a') as f:
        f.write('eef_pos : ')
        f.write(f'{list(std_data2_masked)}\n')

    ax3.set_ylim(0, 1)
    ax3.set_yticks(yticks)

    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    ax4.bar(x_labels, bar_data3_masked, width=bar_width, yerr=std_data3_masked, capsize=2, error_kw={'elinewidth': 0.5}, align='center')
    with open(f'{name}_mean2.txt', 'a') as f:
        f.write('eef_quat : ')
        f.write(f'{list(bar_data3_masked)}\n')
    with open(f'{name}_std2.txt', 'a') as f:
        f.write('eef_quat : ')
        f.write(f'{list(std_data3_masked)}\n')

    ax4.set_ylim(0, 1)
    ax4.set_yticks(yticks)

    ax5 = fig.add_subplot(gs[4, 0], sharex=ax1)
    ax5.bar(x_labels, bar_data4_masked, width=bar_width, yerr=std_data4_masked, capsize=2, error_kw={'elinewidth': 0.5}, align='center')
    with open(f'{name}_mean2.txt', 'a') as f:
        f.write('gripper : ')
        f.write(f'{list(bar_data4_masked)}\n')
    with open(f'{name}_std2.txt', 'a') as f:
        f.write('gripper : ')
        f.write(f'{list(std_data4_masked)}\n')

    ax5.set_ylim(0, 1)
    ax5.set_yticks(yticks)

    # Adjust layout to ensure shared x-axis is visible
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)

    plt.subplots_adjust(hspace=0.1)  # Remove gaps between rows
    plt.savefig(f"{name}pickplacecan_loss.pdf", format='pdf', dpi=1200)
    plt.show()


plt.rcParams.update({'font.size': 14})
conceptx1_mean, conceptx1_std = get_mean_stds_ofconcept(target,0)
conceptx2_mean, conceptx2_std = get_mean_stds_ofconcept(target,1)
conceptx3_mean, conceptx3_std = get_mean_stds_ofconcept(target,2)
conceptx4_mean, conceptx4_std = get_mean_stds_ofconcept(target,3)

# change action depending on indx if indx = 1 choose action1 if indx =5 choose action5 
plot_data(np.array(action1).reshape(1,50),conceptx1_mean,conceptx2_mean,conceptx3_mean,conceptx4_mean,conceptx1_std,conceptx2_std,conceptx3_std,conceptx4_std, 'Label') 