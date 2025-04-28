import json
import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import binned_statistic

from models import CFM, Flow
from models import Classifier
from omnifold_dataset import Omnifold
import os
import logging
from torch.utils.tensorboard import SummaryWriter
from plots import plot_naive_unfold, SetStyle

SetStyle()


def setup_logging(run_dir):
    log_file = os.path.join(run_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",  # Clean format without timestamp
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def calculate_triangle_distance(feed_dict, weights, binning, alternative_name,reference_name='True Data', ntrials=100):
    w = np.abs(binning[1] - binning[0])
    x, _ = np.histogram(feed_dict[reference_name], weights=weights[reference_name], bins=binning)
    x2,_ = np.histogram(feed_dict[reference_name], weights=weights[reference_name]**2, bins=binning)
    
    x_norm = np.sum(x)*w
    y, _ = np.histogram(feed_dict[alternative_name], weights=weights[alternative_name], bins=binning)
    y2, _ = np.histogram(feed_dict[alternative_name], weights=weights[alternative_name]**2, bins=binning)
    y_norm = np.sum(y)*w

    dist = sum(0.5 * w*(x[ib] / x_norm - y[ib] / y_norm) ** 2 / (x[ib] / x_norm + y[ib] / y_norm)
               if x[ib]+ y[ib] > 0 else 0.0 for ib in range(len(x)))
    
    return dist * 1e3

def generate_reco_signal(signal_generator, num_data_reco, data, empty_val,plot=True,plot_folder='plots'):
    generated_signal = signal_generator.evaluate(
        num_evts=num_data_reco,
        device=data.device,
        dtype=data.data_rec.dtype,
    )

    # Combine generated signal with empty events to account for MC acceptance
    num_data_empty = (
        num_data_reco * data.epsilon / (1.0 - data.epsilon)
    )  # N*(1-delta)*epsilon/(1-epsilon)
    
    generated_signal = torch.cat(
        [
            generated_signal,
            empty_val * torch.ones_like(generated_signal[: int(num_data_empty)]),
        ]
    )
    
    signal_mask = generated_signal[:,0] != args.empty_val

    
    if plot:
        logging.info("Generating Signal subtracted Events")
        for i, observable in enumerate(data.observables):
            plt.figure()
            y_true, bins, _ = plt.hist(
                data.revert(data.data_signal_rec[data.data_sig_mask.bool()])[:, i],
                bins=observable["bins"],
                label="True signal",
                histtype="step",
                # density=True,
            )
            y_gen = plt.hist(
                data.revert(generated_signal[signal_mask.bool()])[:, i],
                bins=bins,
                label="Generated signal",
                histtype="step",
                # density = True,
            )
            y_mc = plt.hist(
                data.revert(data.mc_rec[data.mc_rec_mask.bool()])[:, i],
                bins=bins,
                label="Initial MC",
                histtype="step",
                # density = True,
            )
            plt.legend()
            plt.xlabel(observable["tex_label"])
            plt.ylabel("Counts")
            plt.title(f"Signal Feature {i}")
            plt.savefig(f"{plot_folder}/Signal_feat{i}.pdf")
            plt.close()
    return generated_signal, signal_mask



def main(args=None):

    # Create folders to store results
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.plot_path):
        os.makedirs(args.plot_path)

    setup_logging(args.model_path)
    logger = SummaryWriter(args.model_path)

    data = Omnifold(args.path, args.nmc, args.ndata, args.nbkg, args.empty_val,val=True)
    logging.info(
        f"Loaded mc data with shape {data.mc_rec.shape}, pseudo data with shape {data.data_rec.shape} and background data"
        f"with shape {data.mc_bkg.shape}"
    )
    logging.info(
        f"Applied mc cuts rec survived {data.mc_rec_mask.sum()}, gen survived {data.mc_gen_mask.sum()}, bkg survived "
        f"{data.mc_bkg_mask.sum()}"
    )
    logging.info(
        f"Applied data cuts rec survived {data.data_rec_mask.sum()}, gen survived {data.data_gen_mask.sum()}."
    )
    logging.info(f"Epsilon: {data.epsilon}, Delta: {data.delta}")
    logging.info(f"Using device:{data.device}")

    generator_params = {
        "hidden_layers": 4,
        "internal_size": 128,
        "lr": 5e-5,
        "n_epochs": 300,
        "batch_size": 256,
        "batch_size_sample": 2000,
    }
    
    
    signal_generator = CFM(data.data_rec.shape[-1], 0, generator_params, logger).to(
        data.device
    )
    state_dict = torch.load(f"{args.model_path}/signal_generator.pt")
    signal_generator.network.load_state_dict(state_dict)

    classifier_params = {
        "hidden_layers": 4,
        "internal_size": 64,
        "lr": 5e-5,
        "n_epochs": 50,
        "batch_size": 128,
        "batch_size_sample": 2000,
    }

    acceptance_classifier = Classifier(
        dims_in=data.data_rec.shape[1],
        params=classifier_params,
        logger=logger,
        model_name="acceptance classifier",
    )

    state_dict = torch.load(f"{args.model_path}/acceptance_classifier.pt")
    acceptance_classifier.network.load_state_dict(state_dict)
    
    unfold_generator = unfold_generator = CFM(
        data.data_gen.shape[-1], data.data_gen.shape[-1], generator_params, logger
    ).to(data.device)

    state_dict = torch.load(f"{args.model_path}/unfold_generator_{args.niter}.pt")
    unfold_generator.network.load_state_dict(state_dict)


    num_data_reco = int((1.0 - args.bkg_frac) * data.data_rec_mask.sum())

    #Generate background subtracted signal with empty entries
    generated_signal, signal_mask = generate_reco_signal(
        signal_generator, num_data_reco, data,args.empty_val, plot=True,plot_folder = args.plot_path,
    )
        
    #Acceptance calculated based on data
    #Since empty-empty events don't exist, we assign acceptance of 1 to empty reco events
    acceptance = torch.cat([
        acceptance_classifier.evaluate(generated_signal[signal_mask.bool()], return_weights=False),
        torch.ones_like(generated_signal[~signal_mask.bool()][:,0])],-1)
    
    #Produce unfolded estimate based on reco data
    unfolded = unfold_generator.evaluate(data_c=generated_signal)

    #Some reco events are outside the fiducial region
    acceptance_mask = torch.bernoulli(acceptance)
                
    with PdfPages(f"{args.plot_path}/final_unfolding_{args.niter}.pdf") as out:
        for j, observable in enumerate(data.observables):
            print(observable["tex_label"])
            
            plot_naive_unfold(out,
                              data.revert(data.data_gen[data.data_gen_mask.bool()])[:, j],
                              data.revert(data.data_rec[(data.data_rec_mask.bool())])[:, j],
                              data.revert(unfolded[acceptance_mask.bool()])[:, j],
                              bins=observable["bins"],
                              name=observable["tex_label"],
                              yscale=observable["yscale"])




            feed_dict = {
                'Pythia': data.revert(data.mc_gen[data.mc_gen_mask.bool()])[:, j],
                'Data Unfolded':data.revert(unfolded[acceptance_mask.bool()])[:, j],
                'True Data': data.revert(data.data_gen[data.data_gen_mask.bool()])[:, j],
            }
            weights = {
                'Pythia': np.ones_like(feed_dict['Pythia']),
                'Data Unfolded':np.ones_like(feed_dict['Data Unfolded']),
                'True Data': np.ones_like(feed_dict['True Data']),
            }
            
            d = calculate_triangle_distance(feed_dict,weights,observable["bins"],
                                            alternative_name='Data Unfolded')
            print("OmniGen feat {}: {}".format(observable["tex_label"],d))

            d = calculate_triangle_distance(feed_dict,weights,observable["bins"],
                                            alternative_name='Pythia')
            print("Pythia feat {}: {}".format(observable["tex_label"],d))
            
            

if __name__ == "__main__":
    parser = ArgumentParser()

    # General Options
    parser.add_argument(
        "-p", "--path", default="./", help="Directory containing the input data"
    )
    parser.add_argument(
        "-m", "--model_path", default="models", help="Directory to save the models"
    )
    parser.add_argument("--plot_path", default="plots", help="Directory to save plots")
    parser.add_argument(
        "--nmc", default=1000_000, type=int, help="Number of MC events to load"
    )
    parser.add_argument(
        "--ndata", default=1000_000, type=int, help="Number of data events to load"
    )
    parser.add_argument(
        "--nbkg", default=300_000, type=int, help="Number of background events to load"
    )
    parser.add_argument(
        "--niter", default=4, type=int, help="Iteration to load"
    )
    parser.add_argument(
        "--empty_val", default=-1.0, type=float, help="Default values for empty entries"
    )
    parser.add_argument(
        "--bkg_frac", default=0.0814, type=float, help="Default values for empty entries"
    )

    args = parser.parse_args()
    main(args)
