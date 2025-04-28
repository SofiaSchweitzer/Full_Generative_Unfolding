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
from plots import plot_naive_unfold

def setup_logging(run_dir):
    log_file = os.path.join(run_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",  # Clean format without timestamp
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def sample_reco(data,efficiency_classifier, detector_model, gen_model,empty_val,plot_folder='plots'):
    """Generates h(reco|gen) samples by sampling from: h(reco|gen) = c(gen) + (1-c(gen))*p(reco|gen)*p(gen)"""
    num_gen = data.mc_gen_mask.sum()    
    gen_events = gen_model.evaluate(num_evts=num_gen,
                                    device=data.device,
                                    dtype=data.data_rec.dtype)

    gen_mask = data.apply_cuts_preprocessed(gen_events)
    logging.info(f"Accepting {gen_mask.sum()/gen_mask.shape[0]}")    
    gen_events = gen_events[gen_mask.bool()]
    num_gen = gen_events.shape[0]
        
    efficiency = efficiency_classifier.evaluate(gen_events, return_weights=False)
    sample_efficiency = torch.bernoulli(efficiency)

    #Create empty events for gen
    num_data_empty = (
        num_gen * data.delta / (1.0 - data.delta)
    )  # N*(1-delta)*epsilon/(1-epsilon)
    
    gen_events = torch.cat(
        [
            gen_events,
            empty_val * torch.ones_like(gen_events[: int(num_data_empty)]),
        ]
    )
    gen_mask = gen_events[:,0] != empty_val
    
    sample_efficiency = torch.cat(
        [sample_efficiency,
         torch.ones_like(sample_efficiency[:int(num_data_empty)])]
        )
    
    logging.info(
        f"Number of expected signal events in the data {gen_events.size(0)}, true number is {data.data_gen.shape[0]}, ratio is {1.0*gen_events.size(0)/data.data_gen.shape[0]}"
    )

    reco_events = detector_model.evaluate(data_c=gen_events)
    reco_events[~sample_efficiency.bool()] = empty_val * torch.ones_like(gen_events[~sample_efficiency.bool()])

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
            data.revert(reco_events[sample_efficiency.bool()])[:, i],
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
        plt.savefig(f"{plot_folder}/Signal_forward_feat{i}.pdf")
        plt.close()

    
    return reco_events, gen_events, gen_mask, sample_efficiency


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


def get_acceptance_model(args, data, classifier_params,logger):
    acceptance_classifier = Classifier(
        dims_in=data.data_rec.shape[1],
        params=classifier_params,
        logger=logger,
        model_name="acceptance classifier",
    )
    if args.train_acc:
        acceptance_true = data.mc_rec[
            (data.mc_rec_mask.bool()) & (data.mc_gen_mask.bool())
        ]
        acceptance_false = data.mc_rec[
            (data.mc_rec_mask.bool()) & ~(data.mc_gen_mask.bool())
        ]
        acceptance_classifier.train_classifier(
            acceptance_true, acceptance_false, balanced=False
        )
        torch.save(
            acceptance_classifier.network.state_dict(),
            f"{args.model_path}/acceptance_classifier.pt",
        )
    else:
        state_dict = torch.load(f"{args.model_path}/acceptance_classifier.pt")
        acceptance_classifier.network.load_state_dict(state_dict)

    return acceptance_classifier


def get_efficiency_model(args, data,  classifier_params,logger):
    efficiency_classifier = Classifier(
        dims_in=data.data_rec.shape[1],
        params=classifier_params,
        logger=logger,
        model_name="efficiency classifier",
    )
    if args.train_eff:
        efficiency_true = data.mc_gen[
            (data.mc_rec_mask.bool()) & (data.mc_gen_mask.bool())
        ]
        efficiency_false = data.mc_gen[
            ~(data.mc_rec_mask.bool()) & (data.mc_gen_mask.bool())
        ]
        efficiency_classifier.train_classifier(
            efficiency_true, efficiency_false, balanced=False
        )
        torch.save(
            efficiency_classifier.network.state_dict(),
            f"{args.model_path}/efficiency_classifier.pt",
        )
    else:
        state_dict = torch.load(f"{args.model_path}/efficiency_classifier.pt")
        efficiency_classifier.network.load_state_dict(state_dict)

    return efficiency_classifier


def get_gen_model(args, data, generator_params, logger, plot=True):
    gen_generator = CFM(data.data_gen.shape[-1], 0, generator_params, logger).to(data.device)


    if args.train_gen:
        mc_gen = data.mc_gen[data.mc_gen_mask.bool()][:]
        weights = torch.ones_like(mc_gen[:, 0])
        gen_generator.train(mc_gen, weights=weights)
        torch.save(
            gen_generator.network.state_dict(), f"{args.model_path}/gen_generator.pt"
        )
    else:
        state_dict = torch.load(f"{args.model_path}/gen_generator.pt")
        gen_generator.network.load_state_dict(state_dict)

    if plot:
        logging.info("Generating Gen-level Events")

        gen_level_signal = gen_generator.evaluate(
            # num_evts = 100,
            num_evts=data.mc_gen_mask.sum().cpu().detach().numpy(),
            device=data.device,
            dtype=data.data_rec.dtype,
        )

        for i, observable in enumerate(data.observables):
            plt.figure()
            y_true, bins, _ = plt.hist(
                data.revert(data.mc_gen[data.mc_gen_mask.bool()])[:, i],
                bins=observable["bins"],
                label="True signal",
                histtype="step",
                # density=True,
            )
            y_gen = plt.hist(
                data.revert(gen_level_signal)[:, i],
                bins=bins,
                label="Generated signal",
                histtype="step",
                # density = True,
            )
            plt.legend()
            plt.xlabel(observable["tex_label"])
            plt.ylabel("Counts")
            plt.title(f"Signal Feature gen {i}")
            plt.savefig(f"{args.plot_path}/Signal_feat_gen{i}.pdf")
            plt.close()

    return gen_generator


def get_detector_model(args, data, generator_params, logger, plot=True):
    det_generator = CFM(
        data.data_gen.shape[-1], data.data_gen.shape[-1], generator_params, logger
    ).to(data.device)

    if args.train_det:
        mc_rec = data.mc_rec[data.mc_rec_mask.bool()][:]
        mc_gen = data.mc_gen[data.mc_rec_mask.bool()][:]
        weights = torch.ones_like(mc_rec[:, 0])
        det_generator.train(mc_rec, weights=weights, data_c=mc_gen)
        torch.save(
            det_generator.network.state_dict(), f"{args.model_path}/det_generator.pt"
        )
    else:
        state_dict = torch.load(f"{args.model_path}/det_generator.pt")
        det_generator.network.load_state_dict(state_dict)

    if plot:
        logging.info("Generating Det-level Events")

        det_level_signal = det_generator.evaluate(
            data_c=data.mc_gen[data.mc_rec_mask.bool()][:],
            # num_evts= data.mc_rec_mask.sum().cpu().detach().numpy(),
        )

        for i, observable in enumerate(data.observables):
            plt.figure()
            y_true, bins, _ = plt.hist(
                data.revert(data.mc_rec[data.mc_rec_mask.bool()])[:, i],
                bins=observable["bins"],
                label="True signal",
                histtype="step",
                # density=True,
            )
            y_gen = plt.hist(
                data.revert(det_level_signal)[:, i],
                bins=bins,
                label="Generated signal",
                histtype="step",
                # density = True,
            )
            plt.legend()
            plt.xlabel(observable["tex_label"])
            plt.ylabel("Counts")
            plt.title(f"Signal Feature gen {i}")
            plt.savefig(f"{args.plot_path}/Signal_feat_det{i}.pdf")
            plt.close()

    return det_generator


def get_unfold_model(args, data, generator_params, logger, plot=True):
    unfold_generator = CFM(
        data.data_gen.shape[-1], data.data_gen.shape[-1], generator_params, logger
    ).to(data.device)

    if args.train_inverse:
        mc_rec = data.mc_rec[data.mc_gen_mask.bool()][:]
        mc_gen = data.mc_gen[data.mc_gen_mask.bool()][:]
        weights = torch.ones_like(mc_gen[:, 0])
        unfold_generator.train(mc_gen, weights=weights, data_c=mc_rec)
        torch.save(
            unfold_generator.network.state_dict(), f"{args.model_path}/unfold_generator.pt"
        )
    else:
        state_dict = torch.load(f"{args.model_path}/unfold_generator.pt")
        unfold_generator.network.load_state_dict(state_dict)

    if plot:
        logging.info("Generating Unfolded-level Events")

        unfold_level_signal = unfold_generator.evaluate(
            data_c=data.mc_rec[data.mc_gen_mask.bool()][:],
            # num_evts= data.mc_rec_mask.sum().cpu().detach().numpy(),
        )

        for i, observable in enumerate(data.observables):
            plt.figure()
            y_true, bins, _ = plt.hist(
                data.revert(data.mc_gen[data.mc_gen_mask.bool()])[:, i],
                bins=observable["bins"],
                label="True signal",
                histtype="step",
                # density=True,
            )
            y_gen = plt.hist(
                data.revert(unfold_level_signal)[:, i],
                bins=bins,
                label="Generated signal",
                histtype="step",
                # density = True,
            )
            plt.legend()
            plt.xlabel(observable["tex_label"])
            plt.ylabel("Counts")
            plt.title(f"Signal Feature gen {i}")
            plt.savefig(f"{args.plot_path}/Signal_feat_unfold{i}.pdf")
            plt.close()

    return unfold_generator




def get_bkg_model(args, data, generator_params,logger):
    signal_generator = CFM(data.data_rec.shape[-1], 0, generator_params, logger).to(
        data.device
    )


    bkg_frac = 1.0 * data.data_bkg_mask.sum() / data.data_rec_mask.sum()
    logging.info(f"Background fraction is {bkg_frac}.")

    if args.train_bkg:
        bkg_mc = data.mc_bkg[data.mc_bkg_mask.bool()][:]
        data_rec = data.data_rec[data.data_rec_mask.bool()][:]
        mc_bkg_frac = 1.0 * data.mc_bkg_mask.sum() / data.data_rec_mask.sum()
        weights = torch.cat(
            [
                torch.ones_like(data_rec[:, 0]),
                -bkg_frac / mc_bkg_frac * torch.ones_like(bkg_mc[:, 0]),
            ],
            0,
        )
        data_rec = torch.cat([data_rec, bkg_mc], 0)

        signal_generator.train(data_rec, weights=weights)
        torch.save(
            signal_generator.network.state_dict(),
            f"{args.model_path}/signal_generator.pt",
        )
    else:
        state_dict = torch.load(f"{args.model_path}/signal_generator.pt")
        signal_generator.network.load_state_dict(state_dict)

    return signal_generator


def main(args=None):

    # Create folders to store results
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.plot_path):
        os.makedirs(args.plot_path)

    setup_logging(args.model_path)
    logger = SummaryWriter(args.model_path)

    data = Omnifold(args.path, args.nmc, args.ndata, args.nbkg, args.empty_val)
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
        "n_epochs": 200,
        "batch_size": 256,
        "batch_size_sample": 2000,
    }
    
    signal_generator = get_bkg_model(args, data, generator_params, logger)
    bkg_frac = 1.0 * data.data_bkg_mask.sum() / data.data_rec_mask.sum()

    classifier_params = {
        "hidden_layers": 4,
        "internal_size": 64,
        "lr": 1e-4,
        "n_epochs": 50,
        "batch_size": 128,
        "batch_size_sample": 2000,
    }

    acceptance_classifier = get_acceptance_model(args, data, classifier_params, logger)
    efficiency_classifier = get_efficiency_model(args, data, classifier_params, logger)

    # Now we need the detector model and pgen models
    gen_generator = get_gen_model(args, data, generator_params, logger)
    det_generator = get_detector_model(args, data, generator_params, logger)
    unfold_generator = get_unfold_model(args, data, generator_params, logger)

    # Now let's unfold!

    logging.info("Generating Signal Events")
    num_data_reco = int((1.0 - bkg_frac) * data.data_rec_mask.sum())
    logging.info(
        f"Number of expected signal events in the data passing reco cuts {num_data_reco}, true number is {data.data_sig_mask.sum()}"
    )

    generated_signal, signal_mask = generate_reco_signal(
        signal_generator, num_data_reco, data,args.empty_val, plot=True,plot_folder = args.plot_path,
    )
    
    logging.info(
        f"Number of expected signal events in the data {generated_signal.size(0)}, true number is {data.data_signal_rec.shape[0]}, ratio is {1.0*generated_signal.size(0)/data.data_signal_rec.shape[0]}"
    )

    
    #Acceptance calculated based on data
    #Since empty-empty events don't exist, we assign acceptance of 1 to empty reco events
    acceptance = torch.cat([
        acceptance_classifier.evaluate(generated_signal[signal_mask.bool()], return_weights=False),
        torch.ones_like(generated_signal[~signal_mask.bool()][:,0])],-1)
    
    if args.start > 0:
        unfold_generator.network.load_state_dict(
            torch.load(f"{args.model_path}/unfold_generator_{args.start-1}.pt"))
        gen_generator.network.load_state_dict(
            torch.load(f"{args.model_path}/gen_generator_{args.start-1}.pt"))

    #Reduce number of training epochs since we start from the previous model
    gen_generator.params['n_epochs'] = 100
    unfold_generator.params['n_epochs'] = 100
    acceptance_classifier.params['n_epochs'] = 10
    for i in range(args.start,args.niters):
        logging.info(f"Running iteration {i}")
        #Generate forward simulation events
        reco_train, gen_train, gen_mask, reco_mask = sample_reco(
            data,
            efficiency_classifier,
            det_generator,
            gen_generator,
            args.empty_val,
            args.plot_path,
        )

        if i > 0:
            unfold_generator.train(
                gen_train[gen_mask.bool()],
                weights=torch.ones_like(gen_train[gen_mask.bool()][:, 0]),
                data_c=reco_train[gen_mask.bool()],
            )
            acceptance_classifier.train_classifier(
                reco_train[(reco_mask.bool()) & (gen_mask.bool())],
                reco_train[(reco_mask.bool()) & ~(gen_mask.bool())], balanced=False
            )

            acceptance = torch.cat([
                acceptance_classifier.evaluate(generated_signal[signal_mask.bool()], return_weights=False),
                torch.ones_like(generated_signal[~signal_mask.bool()][:,0])],-1)
            
        #Produce unfolded estimate based on reco data
        unfolded = unfold_generator.evaluate(data_c=generated_signal)

        #Some reco events are outside the fiducial region
        acceptance_mask = torch.bernoulli(acceptance)
        
        gen_generator.train(unfolded[acceptance_mask.bool()], weights=torch.ones_like(unfolded[:, 0][acceptance_mask.bool()]))
        
        torch.save(
            unfold_generator.network.state_dict(),
            f"{args.model_path}/unfold_generator_{i}.pt",
        )

        torch.save(
            gen_generator.network.state_dict(),
            f"{args.model_path}/gen_generator_{i}.pt",
        )

        
        with PdfPages(f"{args.plot_path}/final_unfolding_{i}.pdf") as out:
            for j, observable in enumerate(data.observables):
                plot_naive_unfold(out,
                                  data.revert(data.data_gen[data.data_gen_mask.bool()])[:, j],
                                  data.revert(data.data_rec[(data.data_rec_mask.bool())])[:, j],
                                  data.revert(unfolded[acceptance_mask.bool()])[:, j],
                                  bins=observable["bins"],
                                  name=observable["tex_label"],
                                  yscale=observable["yscale"])


        for j, observable in enumerate(data.observables):
            plt.figure()
            y_true, bins, _ = plt.hist(
                data.revert(data.data_gen[data.data_gen_mask.bool()])[:, j],
                bins=observable["bins"],
                label="True signal",
                histtype="step",
                # density=True,
            )
            y_gen = plt.hist(
                data.revert(unfolded[acceptance_mask.bool()])[:, j],
                bins=bins,
                label="Unfolded signal",
                histtype="step",
                # density = True,
            )
            
            y_mc = plt.hist(
                data.revert(data.mc_gen[data.mc_gen_mask.bool()])[:, j],
                bins=bins,
                label="Initial MC prediction",
                histtype="step",
                # density = True,
            )
            
            plt.legend()
            plt.xlabel(observable["tex_label"])
            plt.ylabel("Counts")
            plt.title(f"Signal Feature gen {j}")
            plt.savefig(f"{args.plot_path}/Signal_feat_unf_{j}_iter{i}.pdf")
            plt.close()



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
        "--niters", default=10, type=int, help="Number of iterations to run"
    )
    parser.add_argument(
        "--start", default=0, type=int, help="Iteration to start from"
    )
    parser.add_argument(
        "--empty_val", default=-1.0, type=float, help="Default values for empty entries"
    )
    parser.add_argument(
        "--train_bkg",
        action="store_true",
        default=False,
        help="Train model to subtract background events using geenrative models",
    )
    parser.add_argument(
        "--train_acc",
        action="store_true",
        default=False,
        help="Train acceptance classifier model",
    )
    parser.add_argument(
        "--train_eff",
        action="store_true",
        default=False,
        help="Train efficiency classifier model",
    )
    parser.add_argument(
        "--train_gen",
        action="store_true",
        default=False,
        help="Train MC gen generator",
    )
    parser.add_argument(
        "--train_inverse",
        action="store_true",
        default=False,
        help="Train initial inverse model",
    )

    parser.add_argument(
        "--train_det",
        action="store_true",
        default=False,
        help="Train forward model",
    )

    args = parser.parse_args()
    main(args)
