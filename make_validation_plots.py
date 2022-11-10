import matplotlib.pyplot as plt
import pickle
import utils
import numpy as np
from tqdm import tqdm

naip_nlcd_soft_model = "./checkpoints/FCN_softlabeled_model_11_10.h5"
naip_nlcd_hard_model = "./checkpoints/FCN_ohe_model_11_10.h5"

naip_nlcd_soft_history = "./history/FCN_softlabeled_model_11_10_history.pkl"
naip_nlcd_hard_history = "./history/FCN_ohe_model_11_10_history.pkl"

if __name__ == "__main__":
    with open(naip_nlcd_soft_history, "rb") as f:
        soft_history = pickle.load(f)
        
    with open(naip_nlcd_hard_history, "rb") as f:
        hard_history = pickle.load(f)
    
    for k in tqdm(soft_history.history.keys()):
        plt.figure() # start a new figure
        plt.plot(np.arange(1, utils.train_epochs+1), soft_history.history[k], label="soft labeled")
        plt.plot(np.arange(1, utils.train_epochs+1), hard_history.history[k], label="hard labeled")
        plt.title(f"Model {k.capitalize()} Comparison")
        plt.legend()
        plt.savefig(f"./plots/model_{k}_comparison_11_10.png", bbox_inches="tight", dpi=200)