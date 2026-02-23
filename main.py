import argparse

import yaml
import torch
import os
import numpy as np

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
    if train:
        logging.info("Training set to true")
        if params["include_bkg"]:
            #background classifier
            bkg_true = torch.cat([train_data.data_rec[train_data.data_rec_mask.bool()], train_data.mc_bkg[train_data.mc_bkg_mask.bool()]])
            weights_true = torch.cat([train_data.data_rec_weights[train_data.data_rec_mask.bool()],
                                      torch.ones_like(train_data.mc_bkg[train_data.mc_bkg_mask.bool()][:,0]) * -1])
            bkg_false = train_data.data_rec[train_data.data_rec_mask.bool()]
            weights_false = train_data.data_rec_weights[train_data.data_rec_mask.bool()]

            background_classifier_params = params["background_classifier_params"]

            background_classifier = Classifier(dim, background_classifier_params, logger, "background").to(train_data.device)
            logging.info("background classifier has", sum(p.numel() for p in background_classifier.parameters()), "trainable parameters.")

            background_classifier.train_classifier(bkg_true, bkg_false, weights_true, weights_false)

            background_path = os.path.join(run_dir, "models", f"background.pt")
            torch.save(background_classifier.state_dict(), background_path)

            test_background_weights = background_classifier.evaluate(test_data.data_rec[test_data.data_rec_mask.bool()])
            test_background_weights = test_background_weights * (
            test_data.data_rec_weights[test_data.data_rec_mask.bool()].sum() - len(test_data.mc_bkg[test_data.mc_bkg_mask]))/ test_background_weights.sum()

            train_background_weights = background_classifier.evaluate(train_data.data_rec[train_data.data_rec_mask.bool()])
            train_background_weights = train_background_weights * (train_data.data_rec_weights[train_data.data_rec_mask.bool()].sum()
                                                                   - len(train_data.mc_bkg[train_data.mc_bkg_mask]))/ train_background_weights.sum()
        else:
            train_data.data_rec = train_data.data_rec[:len(train_data.data_signal_rec)]
            train_data.data_rec_mask = train_data.data_rec_mask[:len(train_data.data_rec)]
            train_data.data_rec_weights = train_data.data_rec_weights[:len(train_data.data_rec)]
            train_background_weights = torch.ones_like(train_data.data_rec[train_data.data_rec_mask.bool()][:,0])

            test_data.data_rec = test_data.data_rec[:len(test_data.data_signal_rec)]
            test_data.data_rec_mask = test_data.data_rec_mask[:len(test_data.data_rec)]
            test_data.data_rec_weights = test_data.data_rec_weights[:len(test_data.data_rec)]
            test_background_weights = torch.ones_like(test_data.data_rec[test_data.data_rec_mask.bool()][:, 0])

        if params["include_acceptance"]:
            acceptance_true = train_data.mc_rec[(train_data.mc_rec_mask.bool()) & (train_data.mc_gen_mask.bool())]
            acceptance_false = train_data.mc_rec[(train_data.mc_rec_mask.bool()) & ~(train_data.mc_gen_mask.bool())]
            acceptance_classifier_params = params["acceptance_classifier_params"]
            acceptance_classifier = Classifier(dim, acceptance_classifier_params, logger, "acceptance").to(
                train_data.device)
            acceptance_classifier.train_classifier(acceptance_true, acceptance_false, balanced=False)
            acceptance_train = acceptance_classifier.evaluate(train_data.data_rec[train_data.data_rec_mask.bool()],
                                                              return_weights=False)
            acceptance_test = acceptance_classifier.evaluate(test_data.data_rec[test_data.data_rec_mask.bool()],
                                                             return_weights=False)

            acceptance_path = os.path.join(run_dir, "models", f"acceptance.pt")
            torch.save(acceptance_classifier.state_dict(), acceptance_path)
        else:
            acceptance_train = torch.ones_like(train_data.data_rec[:, 0][train_data.data_rec_mask.bool()])
            acceptance_test = torch.ones_like(test_data.data_rec[:, 0][test_data.data_rec_mask.bool()])

        #Iterative unfolder
        iterative_unfolding_params = {"iterations": params["iterations"],
                                      "generator": params["unfolder_params"],
                                      "classifier": params["iterative_classifier_params"]}

        for j in range(iterative_unfolding_params["iterations"]):
            logging.info(f"Starting with the {j}.iteration.")

            if j == 0:
                logging.info("Initalize unfolder")
                unfolder = CFM(dim, dim, iterative_unfolding_params["generator"], logger).to(train_data.device)
                logging.info("unfolder has", sum(p.numel() for p in unfolder.parameters()),
                      "trainable parameters.")
                mc_gen = train_data.mc_gen[(train_data.mc_rec_mask.bool()) &(train_data.mc_gen_mask.bool()) ]
                mc_rec = train_data.mc_rec[(train_data.mc_rec_mask.bool()) &(train_data.mc_gen_mask.bool())]
                mc_weights = torch.ones_like(mc_rec[:, 0])
                test_mc_weights = torch.ones_like(test_data.mc_rec[(test_data.mc_rec_mask.bool()) &(test_data.mc_gen_mask.bool())][:, 0])

                train_data_weights = train_background_weights * acceptance_train * train_data.data_rec_weights[train_data.data_rec_mask.bool()]
                test_data_weights = test_background_weights * acceptance_test * test_data.data_rec_weights[test_data.data_rec_mask.bool()]

            if j > 0:
                unfolder.params["n_epochs"] = 40
                iterative_classifier = Classifier(dim, iterative_unfolding_params["classifier"],logger, f"iterative_{j}").to(train_data.device)
                logging.info("iterator has", sum(p.numel() for p in iterative_classifier.parameters()),
                      "trainable parameters.")
                iterative_classifier.train_classifier_with_validation(train_data_unfold, mc_gen, train_data_weights, mc_weights)
                mc_weights *= iterative_classifier.evaluate(mc_gen)
                test_mc_weights *= iterative_classifier.evaluate(test_data.mc_gen[test_data.mc_rec_mask.bool() &(test_data.mc_gen_mask.bool())])
                np.save(f"{run_dir}/{j}th_mc_weight.npy", test_mc_weights.cpu())
                #plot histograms
                plt.figure()
                plt.hist(mc_weights.cpu(), bins=50, label='data_rec')
                plt.xlabel('weight')
                plt.ylabel('Frequency')
                plt.legend()
                plt.yscale('log')
                plt.savefig(os.path.join(plot_path, f"classifier_histograms_{j}.pdf"))
                plt.close()
                with PdfPages(f"{plot_path}/prior_dependence_reweighting_{j}.pdf") as out:
                    for channel in params["channels"]:
                        plot_reweighted_distribution(out,  data_unfold_inter[:, channel], mc_gen_inter[:, channel][(test_data.mc_rec_mask.bool()) & (test_data.mc_gen_mask.bool())],
                                          mc_gen_inter[:, channel][(test_data.mc_rec_mask.bool())& (test_data.mc_gen_mask.bool())],
                                          reweighted_weights=test_mc_weights.cpu(),
                                          true_weights= test_data_weights.cpu(),
                                          fake_weights=torch.ones_like(test_mc_weights).cpu(),
                                          bins=test_data.observables[channel]["bins"], name=test_data.observables[channel]["tex_label"],
                                          yscale=test_data.observables[channel]["yscale"],
                                          density=True,
                                          labels=[r"unfolded", r"$w \cdot p_{MC,s}(y)_r$", r"$p_{MC,s}(y)_r$"])
            unfolder.train_unfolder(mc_gen, mc_rec, mc_weights)
            logging.info("unfold data")
            train_data_unfold = unfolder.evaluate(train_data.data_rec[(train_data.data_rec_mask.bool())])
            test_data_unfold = unfolder.evaluate(test_data.data_rec[(test_data.data_rec_mask.bool()) ])
            test_mc_unfold = unfolder.evaluate(test_data.mc_rec[(test_data.mc_rec_mask.bool())  &(test_data.mc_gen_mask.bool())])

            with PdfPages(f"{plot_path}/prior_dependence_{j}.pdf") as out:
                data_unfold_inter, _, _, _,_ = test_data.apply_preprocessing(test_data_unfold.clone(),
                                                                             parameters=[test_data.mean,
                                                                                         test_data.std,
                                                                                         test_data.shift,
                                                                                         test_data.factor], reverse=True)
                np.save(f"{run_dir}/{j}th_unfolded.npy", data_unfold_inter)
                mc_rec_inter, _, _, _, _ = test_data.apply_preprocessing(test_data.mc_rec.clone(),
                                                                         parameters=[test_data.mean, test_data.std, test_data.shift,
                                                                                     test_data.factor], reverse=True)
                mc_gen_inter, _, _, _, _ = test_data.apply_preprocessing(test_data.mc_gen.clone(),
                                                                         parameters=[test_data.mean, test_data.std,
                                                                                     test_data.shift,
                                                                                     test_data.factor], reverse=True)

                mc_unfold, _, _, _, _ = test_data.apply_preprocessing(test_mc_unfold.clone(),
                                                                         parameters=[test_data.mean, test_data.std, test_data.shift,
                                                                                     test_data.factor], reverse=True)
                for channel in params["channels"]:
                    plot_prior_unfold(out,
                                      test_data.data_gen[:, channel][(test_data.data_rec_mask[:len(test_data.data_signal_rec)].bool())
                                      &(test_data.data_gen_mask.bool())],
                                      mc_gen_inter[:, channel][(test_data.mc_rec_mask.bool()) &(test_data.mc_gen_mask.bool())], data_unfold_inter[:, channel],
                                      gen_weights= test_data.data_gen_weights[(test_data.data_rec_mask[:len(test_data.data_signal_rec)].bool())  &(test_data.data_gen_mask.bool())].cpu(),
                                      unfolded_weights=test_data_weights.cpu(), prior_weights=test_mc_weights.cpu(),
                                      bins=test_data.observables[channel]["bins"], name=test_data.observables[channel]["tex_label"],
                                      yscale=test_data.observables[channel]["yscale"],
                                      density=True)
                    plot_naive_unfold(out, mc_gen_inter[:, channel][(test_data.mc_rec_mask.bool())  &(test_data.mc_gen_mask.bool())],
                                      mc_rec_inter[:, channel][(test_data.mc_rec_mask.bool())], mc_unfold[:, channel],
                                      rec_weights=torch.ones_like(mc_rec_inter[:, channel][(test_data.mc_rec_mask.bool())]).cpu(),
                                      gen_weights=test_mc_weights.cpu(), unfolded_weights=test_mc_weights.cpu(),
                                      bins=test_data.observables[channel]["bins"], name=test_data.observables[channel]["tex_label"],
                                      yscale=test_data.observables[channel]["yscale"],
                                      density=True, includes=False)
        #
        iterator_path = os.path.join(run_dir, "models", f"iterator.pt")
        torch.save(iterative_classifier.state_dict(), iterator_path)
        unfolder_path = os.path.join(run_dir, "models", f"unfolder.pt")
        torch.save(unfolder.state_dict(), unfolder_path)

        if params["include_efficiency"]:
            efficiency_classifier_params = params["classifier_params"]

            efficiency_true = train_data.mc_gen[(train_data.mc_gen_mask.bool()) & (train_data.mc_rec_mask.bool())]
            efficiency_false = train_data.mc_gen[~(train_data.mc_rec_mask.bool()) & (train_data.mc_gen_mask.bool())]

            efficiency_classifier = Classifier(dim, efficiency_classifier_params, logger, "efficiency").to(train_data.device)
            logging.info("efficiency_classifier has", sum(p.numel() for p in efficiency_classifier.parameters()),
                  "trainable parameters.")

            efficiency_classifier.train_classifier(efficiency_true, efficiency_false, balanced=False)

            efficiency_path = os.path.join(run_dir, "models", f"efficiency.pt")
            torch.save(efficiency_classifier.state_dict(), efficiency_path)

            efficiency = efficiency_classifier.evaluate(test_data_unfold, return_weights=False)

        else:
            efficiency = torch.ones_like(test_data_unfold)

        data_weights = test_data.data_rec_weights[test_data.data_rec_mask.bool()]* \
                       test_background_weights * \
                       acceptance_test / efficiency
        data_weights = data_weights.clip(0, 10)

    else:
        logging.info("Training set to false")
        background_classifier_params = params["background_classifier_params"]
        background_classifier = Classifier(dim, background_classifier_params, logger, "background").to(
            train_data.device)
        logging.info("Loading bkg classifier.")
        # logging.info("background classifier has", sum(p.numel() for p in background_classifier.parameters()),
        #              "trainable parameters.")
        background_path = os.path.join(run_dir, "models", f"background.pt")
        state_dict = torch.load(background_path, weights_only=True)
        background_classifier.load_state_dict(state_dict)

        test_background_weights = background_classifier.evaluate(test_data.data_rec[test_data.data_rec_mask.bool()])
        test_background_weights = test_background_weights * (len(
            test_data.data_rec[test_data.data_rec_mask.bool()]) - len(
            test_data.mc_bkg[test_data.mc_bkg_mask])) / test_background_weights.sum()
        logging.info("Loading acceptance classifier.")
        acceptance_path = os.path.join(run_dir, "models", f"acceptance.pt")
        state_dict = torch.load(acceptance_path, weights_only=True)

        acceptance_classifier_params = params["acceptance_classifier_params"]
        acceptance_classifier = Classifier(dim, acceptance_classifier_params, logger, "acceptance").to(
                train_data.device)
        acceptance_classifier.load_state_dict(state_dict)
        acceptance_test = acceptance_classifier.evaluate(test_data.data_rec[test_data.data_rec_mask.bool()],
                                                         return_weights=False)
        logging.info("Load unfolder")
        unfolder_params = params["unfolder_params"]

        unfolder_path = os.path.join(run_dir, "models", f"unfolder.pt")
        unfolder = CFM(dim, dim, unfolder_params, logger).to(train_data.device)
        # logging.info("unfolder has", sum(p.numel() for p in unfolder.parameters()),
        #              "trainable parameters.")
        state_dict = torch.load(unfolder_path, weights_only=True)

        unfolder.load_state_dict(state_dict)
        test_data_unfold = unfolder.evaluate(test_data.data_rec[test_data.data_rec_mask.bool()])
        data_unfold_inter, _, _, _, _ = test_data.apply_preprocessing(test_data_unfold.clone(),
                                                                      parameters=[test_data.mean,
                                                                                  test_data.std,
                                                                                  test_data.shift,
                                                                                  test_data.factor], reverse=True)

        logging.info("Loading efficiency classfier")
        efficiency_classifier_params = params["classifier_params"]
        efficiency_classifier = Classifier(dim, efficiency_classifier_params, logger, "efficiency").to(
            train_data.device)
        # efficiency_true = train_data.mc_gen[(train_data.mc_gen_mask.bool()) & (train_data.mc_rec_mask.bool())]
        # efficiency_false = train_data.mc_gen[~(train_data.mc_rec_mask.bool()) & (train_data.mc_gen_mask.bool())]
        #
        # efficiency_classifier.train_classifier(efficiency_true, efficiency_false, balanced=False)
        #
        # efficiency_path = os.path.join(run_dir, "models", f"efficiency.pt")
        # torch.save(efficiency_classifier.state_dict(), efficiency_path)
        # logging.info("efficiency_classifier has", sum(p.numel() for p in efficiency_classifier.parameters()),
        #              "trainable parameters.")

        efficiency_path = os.path.join(run_dir, "models", f"efficiency.pt")
        state_dict = torch.load(efficiency_path, weights_only=True)
        efficiency_classifier.load_state_dict(state_dict)

        efficiency = efficiency_classifier.evaluate(test_data_unfold, return_weights=False)
        data_weights = test_data.data_rec_weights[test_data.data_rec_mask.bool()] * \
                       test_background_weights * \
                       acceptance_test / efficiency
        data_weights = data_weights.clip(0, 10)

    # unfolded_mask = ((data_unfold_inter[:,6] > 150).squeeze())

    test_data.data_rec, _, _ ,_ , _ = test_data.apply_preprocessing(test_data.data_rec, parameters=[test_data.mean,
                                                                                       test_data.std,
                                                                                       test_data.shift,
                                                                                       test_data.factor], reverse=True)
    test_data.mc_gen, _, _ ,_ , _ = test_data.apply_preprocessing(test_data.mc_gen, parameters=[test_data.mean,
                                                                                       test_data.std,
                                                                                       test_data.shift,
                                                                                       test_data.factor], reverse=True)

    # np.save(f"{run_dir}/unfolded.npy", data_unfold_inter[unfolded_mask])
    # np.save(f"{run_dir}/weights.npy", data_weights.cpu()[unfolded_mask])

    np.save(f"{run_dir}/final_unfolded.npy", data_unfold_inter)
    np.save(f"{run_dir}/weights.npy", data_weights.cpu())

    # %%
    with PdfPages(f"{plot_path}/background_suppression.pdf") as out:
        for channel in params["channels"]:
            plot_reweighted_distribution(out, test_data.data_signal_rec[:, channel][
                test_data.data_rec_mask[:len(test_data.data_signal_rec)].bool()],
                                         test_data.data_rec[:, channel][test_data.data_rec_mask.bool()],
                                         test_data.data_rec[:, channel][test_data.data_rec_mask.bool()],
                                         true_weights=test_data.data_gen_weights[
                                             test_data.data_rec_mask[:len(test_data.data_signal_rec)].bool()].cpu(),
                                         reweighted_weights=(test_background_weights
                                                             * test_data.data_rec_weights
                                                             [test_data.data_rec_mask.bool()]).cpu(),
                                         fake_weights=test_data.data_rec_weights[
                                             test_data.data_rec_mask.bool()].cpu(),
                                         bins=test_data.observables[channel]["bins"],
                                         labels=[r"$p_{d,s}(x)_r$", r"$\nu \cdot p_{d,s+b}(x)_r$",
                                                 r"$p_{d,s+b}(x)_r$"],
                                         name=test_data.observables[channel]["tex_label"],
                                         yscale=test_data.observables[channel]["yscale"],
                                         leg_pos=test_data.observables[channel]["leg_pos"])

    with PdfPages(f"{plot_path}/acceptance_effects.pdf") as out:
        for channel in params["channels"]:
            plot_reweighted_distribution(out, test_data.data_signal_rec.cpu()[:, channel][
                (test_data.data_rec_mask[:len(test_data.data_signal_rec)].bool()) & (
                    test_data.data_gen_mask.bool())],
                                         test_data.data_rec[:, channel][test_data.data_rec_mask.bool()],
                                         test_data.data_rec[:, channel][test_data.data_rec_mask.bool()],
                                         true_weights=test_data.data_gen_weights[
                                             test_data.data_rec_mask[:len(test_data.data_signal_rec)].bool() & (
                                                 test_data.data_gen_mask.bool())].cpu(),
                                         reweighted_weights=acceptance_test.cpu()
                                                            * test_background_weights.cpu()
                                                            * test_data.data_rec_weights[test_data.data_rec_mask.bool()].cpu(),
                                         fake_weights=test_background_weights.cpu()
                                                      * test_data.data_rec_weights[test_data.data_rec_mask.bool()].cpu(),
                                         bins=test_data.observables[channel]["bins"],
                                         labels=[r"$p_{d,s}(x)_{r,g}$", r"$\delta \cdot \nu \cdot p_{d,s+b}(x)_r$",
                                                 "$p_{d,s}(x)_r$"],
                                         name=test_data.observables[channel]["tex_label"],
                                         yscale=test_data.observables[channel]["yscale"],
                                         leg_pos=test_data.observables[channel]["leg_pos"])

    with PdfPages(f"{plot_path}/prior_dependence_final.pdf") as out:
        for channel in params["channels"]:
            plot_prior_unfold(out,
                              test_data.data_gen[:, channel][(test_data.data_rec_mask[:len(test_data.data_signal_rec)].bool())
            &(test_data.data_gen_mask.bool())],
                              test_data.mc_gen[:, channel][(test_data.mc_rec_mask.bool()) &(test_data.mc_gen_mask.bool())],
                              data_unfold_inter[:, channel],
                            gen_weights=test_data.data_gen_weights[(test_data.data_rec_mask[:len(test_data.data_signal_rec)])
                                                                    &(test_data.data_gen_mask.bool())].cpu(),
                            unfolded_weights=(test_data.data_rec_weights[test_data.data_rec_mask.bool()]
                                              * test_background_weights
                                              * acceptance_test).cpu(),
                              prior_weights=torch.ones_like(test_data.mc_gen[:, channel][(test_data.mc_rec_mask.bool())
                                                            & (test_data.mc_gen_mask.bool())]),
                              bins=test_data.observables[channel]["bins"], name=test_data.observables[channel]["tex_label"],
                              yscale=test_data.observables[channel]["yscale"],
                              leg_pos=test_data.observables[channel]["leg_pos"])

    with PdfPages(f"{plot_path}/efficiency_acceptance_effects.pdf") as out:
        for channel in params["channels"]:
            plot_reweighted_distribution(out, test_data.data_gen[:, channel][test_data.data_gen_mask.bool()],
                                         data_unfold_inter[:, channel],
                                         data_unfold_inter[:, channel],#[unfolded_mask],
                                         true_weights= test_data.data_gen_weights[test_data.data_gen_mask.bool()].cpu(),
                                         reweighted_weights=data_weights.cpu(),#[unfolded_mask],
                                         fake_weights=(test_data.data_rec_weights[test_data.data_rec_mask.bool()]
                                                       * test_background_weights * acceptance_test).cpu(),
                                         bins=test_data.observables[channel]["bins"],
                                         labels=[r"$p_{d,s}(y)_g$", r"$\text{unfolded} / \epsilon$", "unfolded"],
                                         name=test_data.observables[channel]["tex_label"],
                                         yscale=test_data.observables[channel]["yscale"],
                                         leg_pos=test_data.observables[channel]["leg_pos"])

    with PdfPages(f"{plot_path}/final_unfolding.pdf") as out:
        for channel in params["channels"]:
            plot_naive_unfold(out, test_data.data_gen[:, channel][test_data.data_gen_mask.bool()],
                              test_data.data_rec[(test_data.data_rec_mask.bool())][:, channel],
                              data_unfold_inter[:, channel],#[unfolded_mask],
                              gen_weights=test_data.data_gen_weights[test_data.data_gen_mask.bool()].cpu(),
                              rec_weights= test_data.data_rec_weights[test_data.data_rec_mask.bool()].cpu(),
                              unfolded_weights=data_weights.cpu(),#[unfolded_mask],
                              bins=test_data.observables[channel]["bins"],
                              name=test_data.observables[channel]["tex_label"],
                              yscale=test_data.observables[channel]["yscale"],
                              includes="acceptance",
                              leg_pos=test_data.observables[channel]["leg_pos"])
            d = calculate_triangle_distance(feed_dict=
                                        {"truth": test_data.data_gen[:, channel][test_data.data_gen_mask.bool()],
                                         "unfolded": data_unfold_inter[:, channel]},#[unfolded_mask]},
                                         weights=
                                        {"truth": test_data.data_gen_weights[test_data.data_gen_mask.bool()].cpu(),
                                         "unfolded": data_weights.cpu()},#[unfolded_mask]},
                                        binning= test_data.observables[channel]["bins"], alternative_name="unfolded",
                                            reference_name="truth")

            logging.info(f"Triangle distance in {channel}.observable is {d}.")

    # plot histograms
    # plt.figure()
    # plt.hist(background_weights.cpu(), bins=50, label='data_rec')
    # plt.xlabel('weight')
    # plt.ylabel('Frequency')
    # plt.legend()
    # plt.yscale('log')
    # plt.savefig(os.path.join(plot_path, f"classifier_histograms.pdf"))
    # plt.close()
    # #
    logging.info("Finished.")
if __name__ == '__main__':
    main()