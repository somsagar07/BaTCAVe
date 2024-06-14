import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind
from PIL import Image
import cv2
import torch
import torchvision
from torch.utils.data import IterableDataset, DataLoader
from torchvision import transforms
from captum.attr import LayerGradientXActivation, LayerIntegratedGradients
from captum.concept import TCAV, BTCAV
from captum.concept import Concept
from captum.concept._utils.data_iterator import dataset_to_dataloader, CustomIterableDataset
from captum.concept._utils.common import concepts_to_str
from captum.concept._utils.classifier import Classifier
from sklearn.linear_model import SGDClassifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from BayesianLogistic import VBLogisticRegression
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
# import gym
# import gym_donkeycar
from model import PolicyModel, image_transform, get_tensor_from_filename, load_image_tensors, SKLearnSGDCustom, get_tensor_from_numpy

class CustomClassifier(Classifier):
    def __init__(self):
        # self.lm = linear_model.LogisticRegression(max_iter=1000)
        self.lm = VBLogisticRegression(fit_intercept=False) # We artifically add an intercept below
        self.test_size = 0.33
        self.evaluate_test = False
        self.metrics = None

    def train_and_eval(self, dataloader: DataLoader, **kwargs):
        inputs, labels = [], []
        for X, y in dataloader:
            X = torch.cat((torch.ones((X.shape[0], 1)), X), dim=1) # Add the intercept term. This is required only for the cav classifier. 
            inputs.append(X)
            labels.append(y)
        inputs, labels = torch.cat(inputs).detach().cpu().numpy(), torch.cat(labels).detach().cpu().numpy()
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
            return torch.tensor(np.array([-1 * self.lm.coef_[0], self.lm.coef_[0]]))
        else:
            return torch.tensor(self.lm.coef_)

    def classes(self):
        return self.lm.classes_

    def get_metrics(self):
        return self.metrics

def assemble_concept(name, id, concepts_path="data/tcav/image/concepts/"):
    concept_path = os.path.join(concepts_path, name) + "/"
    dataset = CustomIterableDataset(get_tensor_from_filename, concept_path)
    concept_iter = dataset_to_dataloader(dataset)

    return Concept(id=id, name=name, data_iter=concept_iter)

def delete_files_in_folder(folder_path):
    # Get list of files in the folder
    files = os.listdir(folder_path)

    # Iterate over each file and delete it
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
                print(f"Deleted directory: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

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

def save_btcavs_images(i, image, preds, mean, std, img_path):
    # print("mags", i, magnitudes, preds[0])
    plt.imshow(np.transpose(image, axes=[1, 2 ,0]))
    plt.axis('off')
    plt.title("{} | str={:.2f}, throt={:.2f}, S+={:.2f}±{:.2f}, S-={:.2f}±{:.2f}".format(i, preds[0][0], preds[0][1], mean[0], std[1], mean[1], std[1]), y=-0.1)
    plt.savefig(img_path + '/{}.png'.format(i), bbox_inches='tight')
    # plt.show()

def create_vid(image_folder):
    # Define the path to your folder with images
    output_video = image_folder + '/output_video.mp4'

    # Get all the images in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".png") or img.endswith(".jpg")]

    # Sort the images numerically by converting the filenames to integers
    images.sort(key=lambda x: int(os.path.splitext(x)[0]))

    # Sort the images by name
    # images.sort()

    # Read the first image to get the dimensions
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for .mp4
    frame_rate = 5
    video = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))

    # Add images to the video
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    # Release the video writer object
    video.release()
    cv2.destroyAllWindows()

    print("Video created successfully!")


path = '/Users/...'

# Model
variables = torch.load(path + '/models/ppo_donkeycar_115000_steps/policy.pth', map_location=torch.device('cpu'))
model = PolicyModel(model_weights=variables, post_process=False, return_mean_actions=True)
print(model)

# Concepts
concepts_path = path + "./dataset/data/tcav/image/concepts/"
road_black = assemble_concept('road_black', 1, concepts_path=concepts_path)
orange = assemble_concept('orange', 2, concepts_path=concepts_path)

# BaTCAVe model
delete_files_in_folder("/Users/.../default_model_id") 
layers = ['linear.0',]
classifier = CustomClassifier() 
# classifier = SKLearnSGDCustom()
mytcav = BTCAV(model=model, layers=layers, classifier=classifier, n_samples=2000,
                layer_attr_method = None)

# Infer
action_dict = {'steer': 0, 'throttle': 1}
experimental_sets = [[road_black,orange],] 

concepts_to_index = dict()
for i, exp_set in enumerate(experimental_sets):
    for j, concept in enumerate(exp_set):
        concepts_to_index[concept.name] = (i, j)

experimental_sets_ids = ['-'.join([f'{i.id}' for i in j]) for j in experimental_sets]
magnitudes = [{layers[0]: [], } for _ in range(len(experimental_sets))]
target_action = 'steer' 

rollout_imgs_tensors = get_tensor_from_numpy(filename=path + '/roll-out/ppo_donkeycar_115000_steps/R_2/2.npy')  
inference_length = len(rollout_imgs_tensors)

# Individual BaTCAVes
path_results = path + 'results/results0'
vals_list = []
tt = 0
for i, (experimental_sets_id, experimental_set) in enumerate(zip(experimental_sets_ids, experimental_sets)):
    print("Running BaTCAVe for: ", i, (experimental_sets_id, experimental_set))
    preds = []
    summary_batcaves = []
    for image in rollout_imgs_tensors.unbind():
        tt += 1
        image = image.unsqueeze(dim=0)
        with torch.no_grad():
            model_output = model(image).detach().cpu().numpy()
            preds.append(model_output)
        tcav_score = mytcav.interpret(inputs=image, 
                                    experimental_sets=[experimental_set],
                                    target=action_dict[target_action])
        ex_sets, cons, vals = extract_btcav_scores([experimental_set], tcav_score, layers)
        vals = np.array(vals)[:,:,0]
        vals_list.append(vals)
        val_mean, val_std = np.mean(vals, axis=1), np.std(vals, axis=1)
        summary_batcaves.append([tt-1, model_output[0][0], model_output[0][1], val_mean[0], val_std[0], val_mean[1], val_std[1]])

        data = np.array(summary_batcaves)
        filter = data[:,1] > 2 # Define action conepts
        means = np.mean(data[filter,:], axis=0)[[3,5]]

        # Save summary
        print("Test img={} \n BaTCAVe avg={} \n BaTCAVe std={}\n--------".format(tt, val_mean, val_std))
        save_btcavs_images(len(preds)-1,
                    image.detach().cpu().numpy()[0,:],
                    model(image).detach().cpu().numpy(),
                    val_mean, val_std,
                    path_results)
    np.savetxt(path + 'results/results0/steer_r2_2-road_black-orange_video_linear0.csv', np.array(summary_batcaves), delimiter=",") #steer, throttle, batcavs for the two concepts

# Generate a video
create_vid(path_results)
