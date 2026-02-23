import argparse

import numpy as np
import yaml
import torch
import os


import logging
from torch.utils.tensorboard import SummaryWriter
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from datetime import datetime
from models import CFM
from models import Classifier
from omnifold_dataset import Omnifold
from plots import plot_naive_unfold, plot_reweighted_distribution, plot_prior_unfold,calculate_triangle_distance

import sys
def setup_logging(run_dir):
    log_file = os.path.join(run_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),  # Log to a file
            logging.StreamHandler()  # Log to the console
        ]
    )
    def log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
        logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = log_uncaught_exceptions
    logging.getLogger("lightning_fabric.plugins.environments.slurm").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    train  = True

    args = parser.parse_args()


    # read in the parameters
    with open(args.path, 'r') as f:
        params = yaml.safe_load(f)

    if train:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + params["run_name"]
        run_dir = os.path.join(dir_path, "results", run_name)
        os.makedirs(run_dir)
    else:
        run_dir = os.path.dirname(os.path.abspath(args.path))
        run_name = os.path.basename(run_dir)

    with open(os.path.join(run_dir, "params.yaml"), 'w') as f:
        yaml.dump(params, f)

    plot_path = os.path.join(run_dir, f"run_{params['runs']}", "plots")
    os.makedirs(plot_path, exist_ok=True)
    setup_logging(run_dir)
    logging.info("Starting run: %s", run_name)

    log_dir = os.path.join(run_dir, "logs")
    logger = SummaryWriter(log_dir)

    logging.info(f"train_model: Logging to log_dir {log_dir}")
    params["classifier_params"]["run_dir"] = run_dir
    dim = params["dim"]
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    train_data = Omnifold({"path": params["data_path"], "channels": params["channels"]})
    test_data = Omnifold({"path": params["data_path"], "channels": params["channels"]}, dataset_type="test")

    logging.info(f"Loaded mc data with shape {train_data.mc_rec.shape}, pseudo data with shape {train_data.data_rec.shape} and background train_data"
          f"with shape {train_data.mc_bkg.shape}")
    logging.info(
        f"Applied mc cuts rec survived {train_data.mc_rec_mask.sum()}, gen survived {train_data.mc_gen_mask.sum()}, bkg survived "
        f"{train_data.mc_bkg_mask.sum()}")
    logging.info(
        f"Applied data cuts rec survived {train_data.data_rec_mask.sum()}.")
    logging.info(f"Using device:{train_data.device}")


    reweighter_params = params["classifier_params"]
    classifier = Classifier(dim, reweighter_params, logger, f"reweighter").to(
        train_data.device)

    classifier_true = train_data.data_gen
    classifier_false = train_data.mc_gen
    classifier.train_classifier_with_validation(classifier_true,classifier_false)
    train_mc_weights = classifier.evaluate(classifier_false)
    test_mc_weights = classifier.evaluate(test_data.mc_gen)

    np.save(f"{run_dir}/train_weights.npy", train_mc_weights.cpu())
    np.save(f"{run_dir}/test_weights.npy", test_mc_weights.cpu())

    test_data.mc_gen, _, _ ,_ , _ = test_data.apply_preprocessing(test_data.mc_gen, parameters=[test_data.mean,
                                                                                       test_data.std,
                                                                                       test_data.shift,
                                                                                       test_data.factor], reverse=True)
    test_data.mc_rec, _, _, _, _ = test_data.apply_preprocessing(test_data.mc_rec, parameters=[test_data.mean,
                                                                                               test_data.std,
                                                                                               test_data.shift,
                                                                                               test_data.factor],
                                                                 reverse=True)
    test_data.data_rec, _, _, _, _ = test_data.apply_preprocessing(test_data.data_rec, parameters=[test_data.mean,
                                                                                               test_data.std,
                                                                                               test_data.shift,
                                                                                               test_data.factor],
                                                                 reverse=True)


    with PdfPages(f"{plot_path}/reweighted_pythia.pdf") as out:
        for channel in params["channels"]:
            plot_reweighted_distribution(out, test_data.data_gen[:, channel],
                                         test_data.mc_gen[:, channel],
                                         test_data.mc_gen[:, channel],
                                         reweighted_weights=test_mc_weights.cpu(),
                                         fake_weights=torch.ones_like(test_mc_weights).cpu(),
                                         true_weights= torch.ones_like(test_data.data_gen[:, channel]),
                                         bins=test_data.observables[channel]["bins"],
                                         labels=[r"$p_{d,s}(y)$", "reweighted",r"$p_{MC,s}(y)$"],
                                         name=test_data.observables[channel]["tex_label"],
                                         leg_pos=test_data.observables[channel]["leg_pos"],
                                         yscale=test_data.observables[channel]["yscale"], density=True)

            plot_reweighted_distribution(out, test_data.data_rec[:, channel],
                                         test_data.mc_rec[:, channel],
                                         test_data.mc_rec[:, channel],
                                         reweighted_weights=test_mc_weights.cpu(),
                                         fake_weights=torch.ones_like(test_mc_weights).cpu(),
                                         true_weights=torch.ones_like(test_data.data_rec[:, channel]),
                                         bins=test_data.observables[channel]["bins"],
                                         labels=[r"$p_{d,s}(x)$", "reweighted",r"$p_{MC,s}(x)$"],
                                         name=test_data.observables[channel]["tex_label"],
                                         yscale=test_data.observables[channel]["yscale"],
                                         leg_pos=test_data.observables[channel]["leg_pos"],
                                         density=True)

    logging.info("Finished.")
if __name__ == '__main__':
    main()