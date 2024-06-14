import json
import pandas as pd
from PIL import Image
import torch
import cv2
import numpy as np
import clip
import matplotlib.pyplot as plt
import time
from transforms3d import euler
import torch.nn as nn
import robosuite as suite
from robosuite.utils.input_utils import *
from control_utils import *
from transforms3d.euler import quat2euler, euler2mat, mat2euler
from recorder import SuccessRecorderLift
import os
from collect_data_lift_panda import init_suite as init_suite_lift
from utils.camera_utils import xyz_to_xy
from captum.concept._utils.data_iterator import dataset_to_dataloader
from captum.concept import TCAV, Concept
from captum.attr import LayerIntegratedGradients
from torchvision import transforms
from torch import Tensor
from torch.utils.data import IterableDataset
from typing import Iterator
import glob
from captum.concept._utils.classifier import Classifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from torch.utils.data import IterableDataset, DataLoader

from captum.concept._utils.common import concepts_to_str
from btcav.BayesianLogistic import VBLogisticRegression
from captum.concept import TCAV, BTCAV
from matplotlib.gridspec import GridSpec
import os, shutil

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Model initialization
from models.film_model_dense_waypoints import Backbone as ImageBC
ckpt = '40000.pth'
model = ImageBC(img_size=224, embedding_size=256, num_weight_points=36, input_nc=3)
if ckpt is not None:
    model.load_state_dict(torch.load(ckpt, map_location=device)['model'], strict=True)
model = model.to(device)

TASK = 'lift'
RENDER = True
GRIPPER_STATUS = None
METHOD = 'image_bc'
ROLLING_GIPPER = []


TASK = 'lift'
RENDER = True
GRIPPER_STATUS = None
METHOD = 'image_bc'
ROLLING_GIPPER = []



def get_input(lang='pick up the can', env=None):
    obs = env._get_observations()

    # grab image
    img = obs['agentview_image'][::-1]

    # grab proprioception
    joints = list(obs['robot0_joint_pos_cos']) + list(obs['robot0_joint_pos_sin']) + list(obs['robot0_gripper_qpos'][0:1])
    # joints = torch.tensor(joints, dtype=torch.float32).unsqueeze(0).to(device)
    
    inputs = {
        'image': img,
        'joints': joints,
        'lang': lang,
    }
    return inputs

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

def make_image_tensor(path):
    img = Image.open(path)
    img_tensor = np.array(img)[:, :, :3] / 255.0
    img_tensor = img_tensor - imagenet_mean
    img_tensor = img_tensor / imagenet_std
    img_tensor = torch.tensor(img_tensor, dtype=torch.float32).unsqueeze(0)

    return img_tensor.to(device)

def make_sent_tensor(sent):
    return clip.tokenize([sent]).to(device)


def get_tensor_from_filename(filename):
    img = Image.open(filename).convert("RGB")
    return img


def load_image_tensors(class_name, root_path='Concept', image_index=None):
    path = os.path.join(root_path, class_name)
    filenames = glob.glob(path + '/*.png')
    filenames.sort()  # Ensure consistent ordering

    if image_index is not None:
        if image_index < 0 or image_index >= len(filenames):
            raise IndexError("image_index is out of range")
        filenames = [filenames[image_index]] * 2

    tensors = []
    sentence = "pick up"
    for filename in filenames:
        inp = torch.cat((make_image_tensor(filename).flatten(1), make_sent_tensor(sentence).flatten(1)), dim=1).to(device)
        tensors.append(inp)
    
    if image_index is not None:
        return torch.stack(tensors, dim=1).squeeze(0)
    return torch.stack(tensors, dim=1).squeeze()


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

def read_json_lines(file_path):
    data = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from line: {line}")
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
    return data

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
    
class WrapperModel(nn.Module):
    def __init__(self, original_model, target_index):
        super(WrapperModel, self).__init__()
        self.original_model = original_model
        self.target_index = target_index

    def forward(self, combined_input):
        # Assuming combined_input is of shape [batch_size, img_size + sentence_size]
        # Split the combined input back into img_tensor and sentence_tensor
        img_size = 224 * 224 * 3
        sentence_size = 77
        
        # Separate the image part and the sentence part
        img_flattened = combined_input[:, :-sentence_size]
        sentence_tensor = combined_input[:, -sentence_size:]

        # Reshape the image part
        img_tensor = img_flattened.reshape(-1, 224, 224, 3)

        # Call the original model with the reshaped inputs
        output = self.original_model(img_tensor, sentence_tensor.to(torch.int32))
        
        # Return only the target index output to ensure single value
        return output[:, self.target_index]
    
# Custom class to handle image dataset
class CustomEnvDataset(IterableDataset):
    def __init__(self, env_list):
        self.env_list = env_list


    def __iter__(self) -> Iterator[Tensor]:
        for env_data in self.env_list:
            yield env_data

def assemble_concept_env(concept_name, id, envs):
    dataset = CustomEnvDataset(envs)
    concept_iter = dataset_to_dataloader(dataset)
    return Concept(id=id, name=concept_name, data_iter=concept_iter)


def get_BTCAV(concept,random,input,layer,model,target):
    remove_cav_folder()
    model.to(device)
    mytcav = BTCAV(model=model,
                    layers=layer,
                    classifier=CustomClassifier(),
                    n_samples=1000,
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

def get_allscore(color_concept, random_concept, input_tensor):
    layers1 = ['original_model.controller_xyz']
    layers2 = ['original_model.controller_rpy']
    layers3 = ['original_model.controller_grip']
    scores0 = []
    scores1 = []
    scores2 = []
    scores3 = []
    scores4 = []
    scores5 = []
    scores6 = []
    scores7 = []
    scores8 = []
    scores9 = []

    all_means = [[] for _ in range(10)]
    all_stds = [[] for _ in range(10)]
    all_lbs = [[] for _ in range(10)]
    all_ubs = [[] for _ in range(10)]

    for i in range(36):
        remove_cav_folder()
        scores0.append(get_BTCAV(color_concept,random_concept,input_tensor,layers1,WrapperModel(model, 0).to(device),i))
        remove_cav_folder()
        scores1.append(get_BTCAV(color_concept,random_concept,input_tensor,layers1,WrapperModel(model, 1).to(device),i))
        remove_cav_folder()
        scores2.append(get_BTCAV(color_concept,random_concept,input_tensor,layers1,WrapperModel(model, 2).to(device),i))
        remove_cav_folder()
        scores3.append(get_BTCAV(color_concept,random_concept,input_tensor,layers2,WrapperModel(model, 3).to(device),i))
        remove_cav_folder()
        scores4.append(get_BTCAV(color_concept,random_concept,input_tensor,layers2,WrapperModel(model, 4).to(device),i))
        remove_cav_folder()
        scores5.append(get_BTCAV(color_concept,random_concept,input_tensor,layers2,WrapperModel(model, 5).to(device),i))
        remove_cav_folder()
        scores6.append(get_BTCAV(color_concept,random_concept,input_tensor,layers2,WrapperModel(model, 6).to(device),i))
        remove_cav_folder()
        scores7.append(get_BTCAV(color_concept,random_concept,input_tensor,layers2,WrapperModel(model, 7).to(device),i))
        remove_cav_folder()
        scores8.append(get_BTCAV(color_concept,random_concept,input_tensor,layers2,WrapperModel(model, 8).to(device),i))
        remove_cav_folder()
        scores9.append(get_BTCAV(color_concept,random_concept,input_tensor,layers3,WrapperModel(model, 9).to(device),i))
    all_scores = [scores0, scores1, scores2, scores3, scores4, scores5, scores6, scores7, scores8, scores9]
    
    # Extract means, stds, lbs, ubs
    for idx, scores in enumerate(all_scores):
        for score in scores:
            mean, std, lb, ub = score
            all_means[idx].append(mean)
            all_stds[idx].append(std)
            all_lbs[idx].append(lb)
            all_ubs[idx].append(ub)

    result = {
        'means': all_means,
        'stds': all_stds,
        'lbs': all_lbs,
        'ubs': all_ubs
    }

    return result


def get_action_tcavplot(action_data, tcav_data, ubs, lbs, concept1_name, concept2_name):
    names = ['x', 'y', 'z', 'rcos', 'rsin', 'pcos', 'psin', 'ycos', 'ysin', 'gripper']

    # Create the figure and GridSpec for subplots
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(15, 4, height_ratios=[1]*15, width_ratios=[20, 1, 20, 1], hspace=0, wspace=0.5)

    three_fourth_values = []

    for i in range(10):
        row = (i // 2) * 3
        col = (i % 2) * 2
        
        # Heatmap axes
        ax_heatmap = fig.add_subplot(gs[row, col])
        im = ax_heatmap.imshow(action_data[i].reshape(1, -1), aspect='auto', cmap='jet')
        ax_heatmap.set_xticks([])
        ax_heatmap.set_yticks([])

        # Place the names to the left of the heatmap
        fig.text(ax_heatmap.get_position().x0 - 0.01, ax_heatmap.get_position().y0 + 0.5 * ax_heatmap.get_position().height, names[i], ha='right', va='center', fontsize=12,fontweight='bold')

        # Calculate the three-fourth value (threshold)
        im_limits = im.get_clim()
        three_fourth_value = (3/4) * (im_limits[1] - im_limits[0]) + im_limits[0]
        three_fourth_values.append(three_fourth_value)

        avg_tcav = []
        # Add points based on the threshold
        for j in range(len(action_data[i])):
            if action_data[i][j] > three_fourth_value:
                ax_heatmap.scatter(j, 0, color='black', s=20, marker='*')
                avg_tcav.append(tcav_data[i][j])
            else:
                tcav_data[i][j] = 0
                ubs[i][j] = 0
                lbs[i][j] = 0

        # Add individual colorbars for each heatmap
        cax = fig.add_axes([ax_heatmap.get_position().x1 + 0.02, ax_heatmap.get_position().y0, 0.02, ax_heatmap.get_position().height])
        fig.colorbar(im, cax=cax, orientation='vertical')
        
        # Bar plot axes
        ax_bar = fig.add_subplot(gs[row + 1, col])
        bar_width = 1.0
        ax_bar.bar(range(len(tcav_data[i])), tcav_data[i], color='skyblue', yerr=[tcav_data[i] - np.array(lbs[i]), np.array(ubs[i]) - tcav_data[i]],capsize=1 ,error_kw={'elinewidth': 0.5},width=bar_width, align='edge')
        ax_bar.set_xlim(0, len(tcav_data[i]))
        ax_bar.set_xlabel('Timestep', fontsize=10)
        ax_bar.set_ylabel('TCAV', fontsize=10)

        fig.text(ax_bar.get_position().x0, ax_bar.get_position().y0 - 0.5 * ax_bar.get_position().height, f'Avg TCAV({names[i]}) : {np.mean(avg_tcav):.2f}', ha='right', va='center', fontsize=10, fontweight='bold')
    
    fig.suptitle(f'Local Explaination : {concept1_name} / {concept2_name}', fontsize=16)
        
    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.show()



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

    fig.text(0.90, 0.65, 'object', ha='right', va='center', fontsize=14)
    fig.text(0.90, 0.5, 'eef_pos', ha='right', va='center', fontsize=14)
    fig.text(0.90, 0.35, 'eef_quat', ha='right', va='center', fontsize=14)
    fig.text(0.90, 0.2, 'gripper', ha='right', va='center', fontsize=14)

    # Create x-axis labels to match the heatmap columns
    x_labels = range(heatmap_data.shape[1])

    # Plot the bar graphs
    bar_width = 1.0  # Full width for each bar to eliminate gaps
    yticks = [0.5]

    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2.bar(x_labels, bar_data1_masked, width=bar_width, yerr=std_data1_masked, capsize=2, error_kw={'elinewidth': 0.5}, align='center')

    ax2.set_ylim(0, 1)
    ax2.set_yticks(yticks)

    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax3.bar(x_labels, bar_data2_masked, width=bar_width, yerr=std_data2_masked, capsize=2, error_kw={'elinewidth': 0.5}, align='center')

    ax3.set_ylim(0, 1)
    ax3.set_yticks(yticks)

    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    ax4.bar(x_labels, bar_data3_masked, width=bar_width, yerr=std_data3_masked, capsize=2, error_kw={'elinewidth': 0.5}, align='center')

    ax4.set_ylim(0, 1)
    ax4.set_yticks(yticks)

    ax5 = fig.add_subplot(gs[4, 0], sharex=ax1)
    ax5.bar(x_labels, bar_data4_masked, width=bar_width, yerr=std_data4_masked, capsize=2, error_kw={'elinewidth': 0.5}, align='center')

    ax5.set_ylim(0, 1)
    ax5.set_yticks(yticks)

    # Adjust layout to ensure shared x-axis is visible
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax4.get_xticklabels(), visible=False)

    plt.subplots_adjust(hspace=0.1)  # Remove gaps between rows
    plt.savefig(f"{name}_Lift.pdf", format='pdf', dpi=1200)
    plt.show()


ROLL_OUT_NUM = 0
output_data = read_json_lines('output_list_1.json') 
output_t1 = output_data[ROLL_OUT_NUM][0]['traj']
output_t2 = output_data[ROLL_OUT_NUM][1]['traj']
concatenated_list = [list(a + b) for a, b in zip(output_t1, output_t2)]
action_data=np.array(concatenated_list)



# choosing concepts

color_concept = assemble_concept_env('red', 0, load_image_tensors('blur_red'))
random_concept = assemble_concept_env('rand', 1, load_image_tensors('random_l'))
input_tensor = load_image_tensors('Inputs/input1', image_index=0).to(device)
input_tensor2 = load_image_tensors('Inputs/input2', image_index=0).to(device)
red_tcavs = get_allscore(color_concept,random_concept,input_tensor)
red_tcavs2 = get_allscore(color_concept,random_concept,input_tensor2)
red_tcav_data = {
        key: [a + b for a, b in zip(red_tcavs[key], red_tcavs2[key])]
        for key in red_tcavs
    }

# change to desired concept here its shown for red
get_action_tcavplot(action_data.reshape(1,72), red_tcav_data['means'], red_tcav_data['ubs'], red_tcav_data['lbs'], 'Red', 'Random')