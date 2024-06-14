import pandas as pd
import numpy as np
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
import json



# Load json
df = pd.read_json('observations.json', orient='records', lines=True)


class ObservationEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ObservationEncoder, self).__init__()
    def forward(self, x):
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
    def __init__(self, obs_input_dims, action_dim, encoder_output_dim=19, mlp_layer_dims=(1024,)):
        super(ActorNetwork, self).__init__()
        self.encoder = ObservationEncoder(sum(obs_input_dims), encoder_output_dim)
        self.mlp = MLP(encoder_output_dim, 1024, mlp_layer_dims)
        self.decoder = ObservationDecoder(1024, action_dim)

    def forward(self, obs):
        encoded_obs = self.encoder(obs)
        mlp_output = self.mlp(encoded_obs)
        actions = self.decoder(mlp_output)
        return actions

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




# loading TCAV model with policy weights
tcav_model = ActorNetwork(obs_input_dims=[10, 3, 4, 2], action_dim=7)

tcav_model_weights = torch.load('tcav_model_weights.pth')
tcav_model.load_state_dict(tcav_model_weights)
print("Weights loaded successfully into the model.")



# Generating random concepts
n_episode = 100

random1_concept_data = np.concatenate(
    (np.random.rand(n_episode, 10), np.random.rand(n_episode, 3), np.random.rand(n_episode, 4), np.random.rand(n_episode, 2)),
    axis=1
)
random2_concept_data = np.concatenate(
    (np.random.rand(n_episode, 10), np.random.rand(n_episode, 3), np.random.rand(n_episode, 4), np.random.rand(n_episode, 2)),
    axis=1
)



# Loading concept data
object_list = []
robot0_eef_pos_list = []
robot0_eef_quat_list = []
robot0_gripper_qpos_list = []


# Change column to try different timestep
for i in range(len(df['Initial Observations'])): # At t = 0
    observation = df['Initial Observations'][i]  
    object_list.append(observation.get('object', None))  
    robot0_eef_pos_list.append(observation.get('robot0_eef_pos', None))
    robot0_eef_quat_list.append(observation.get('robot0_eef_quat', None))
    robot0_gripper_qpos_list.append(observation.get('robot0_gripper_qpos', None))


data_object = torch.tensor(object_list, dtype=torch.float32)
data_robot0_eef_pos = torch.tensor(robot0_eef_pos_list, dtype=torch.float32)
data_robot0_eef_quat = torch.tensor(robot0_eef_quat_list, dtype=torch.float32)
data_robot0_gripper_qpos = torch.tensor(robot0_gripper_qpos_list, dtype=torch.float32)



# Environment Data
concatenated_data = np.concatenate(
    (data_object.cpu().detach().numpy(), data_robot0_eef_pos.cpu().detach().numpy(), data_robot0_eef_quat.cpu().detach().numpy(), data_robot0_gripper_qpos.cpu().detach().numpy()),
    axis=1
)



# Generating concepts
object_concept_data = np.concatenate(
    (data_object.cpu().detach().numpy(), np.random.rand(n_episode, 3), np.random.rand(n_episode, 4), np.random.rand(n_episode, 2)),
    axis=1
)
robot0_eef_pos_concept_data = np.concatenate(
    (np.random.rand(n_episode, 10), data_robot0_eef_pos.cpu().detach().numpy(), np.random.rand(n_episode, 4), np.random.rand(n_episode, 2)),
    axis=1
)
robot0_eef_quat_concept_data = np.concatenate(
    (np.random.rand(n_episode, 10), np.random.rand(n_episode, 3), data_robot0_eef_quat.cpu().detach().numpy(), np.random.rand(n_episode, 2)),
    axis=1
)
robot0_gripper_qpos_concept_data = np.concatenate(
    (np.random.rand(n_episode, 10), np.random.rand(n_episode, 3), np.random.rand(n_episode, 4), data_robot0_gripper_qpos.cpu().detach().numpy()),
    axis=1
)




# Concepts
object_concept = assemble_concept_env("object", 0, object_concept_data, transform)
robot0_eef_pos_concept = assemble_concept_env("robot0_eef_pos", 1, robot0_eef_pos_concept_data, transform)
robot0_eef_quat_concept = assemble_concept_env("robot0_eef_quat", 2, robot0_eef_quat_concept_data, transform)
robot0_gripper_qpos_concept = assemble_concept_env("robot0_gripper_qpos", 3, robot0_gripper_qpos_concept_data, transform)


# Random concepts
random_0_concept = assemble_concept_env("random_0", 4, random1_concept_data, transform)
random_1_concept = assemble_concept_env("random_1", 5, random2_concept_data, transform)

# Final layer
layers=['decoder']

mytcav = TCAV(model=tcav_model,
              layers=layers,
              # classifier=classifier,
              layer_attr_method = LayerIntegratedGradients(
                tcav_model, None, multiply_by_inputs=False))

experimental_set_rand = [[object_concept, random_0_concept], [robot0_eef_pos_concept, random_0_concept], [robot0_eef_quat_concept, random_0_concept], [robot0_gripper_qpos_concept, random_0_concept]]
input_tensor = torch.from_numpy(concatenated_data)


z_ind = 0
tcav_scores_w_random = mytcav.interpret(inputs=input_tensor,
                                        experimental_sets=experimental_set_rand,
                                        target=z_ind,
                                        n_steps=5,
                                       )
print(tcav_scores_w_random)

def format_float(f):
    return float('{:.3f}'.format(f) if abs(f) >= 0.0005 else '{:.3e}'.format(f))

def plot_tcav_scores(experimental_sets, tcav_scores):
    fig, ax = plt.subplots(1, len(experimental_sets), figsize = (25, 7))

    barWidth = 1 / (len(experimental_sets[0]) + 1)

    for idx_es, concepts in enumerate(experimental_sets):

        concepts = experimental_sets[idx_es]
        concepts_key = concepts_to_str(concepts)

        pos = [np.arange(len(layers))]
        for i in range(1, len(concepts)):
            pos.append([(x + barWidth) for x in pos[i-1]])
        _ax = (ax[idx_es] if len(experimental_sets) > 1 else ax)
        for i in range(len(concepts)):
            val = [format_float(scores['sign_count'][i]) for layer, scores in tcav_scores[concepts_key].items()]
            _ax.bar(pos[i], val, width=barWidth, edgecolor='white', label=concepts[i].name)

        # Add xticks on the middle of the group bars
        _ax.set_xlabel('Set {}'.format(str(idx_es)), fontweight='bold', fontsize=16)
        _ax.set_xticks([r + 0.3 * barWidth for r in range(len(layers))])
        _ax.set_xticklabels(layers, fontsize=16)

        # Create legend & Show graphic
        _ax.legend(fontsize=16)

    plt.show()
plot_tcav_scores(experimental_set_rand, tcav_scores_w_random)
