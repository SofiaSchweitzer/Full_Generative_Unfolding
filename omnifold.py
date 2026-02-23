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
from plots import plot_naive_unfold, plot_reweighted_distribution, plot_prior_unfold, calculate_triangle_distance
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
    args = parser.parse_args()

    # read in the parameters
    with open(args.path, 'r') as f:
        params = yaml.safe_load(f)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + params["run_name"]
    run_dir = os.path.join(dir_path, "results", run_name)
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "params.yaml"), 'w') as f:
        yaml.dump(params, f)

    plot_path = os.path.join(run_dir, f"run_{params['runs']}", "plots")
    os.makedirs(plot_path, exist_ok=True)
    setup_logging(run_dir)
    logging.info("Starting run: %s", run_name)

    log_dir = os.path.join(run_dir, "logs")
    logger = SummaryWriter(log_dir)

    logging.info(f"train_model: Logging to log_dir {log_dir}")
    params["omnifold_classifier_params"]["run_dir"] = run_dir
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
    if params["include_bkg"]:
        #background classifier
        bkg_true = torch.cat([train_data.data_rec[train_data.data_rec_mask.bool()], train_data.mc_bkg[train_data.mc_bkg_mask.bool()]])
        weights_true = torch.cat([torch.ones_like(train_data.data_rec[:,0][train_data.data_rec_mask.bool()]), torch.ones_like(train_data.mc_bkg[train_data.mc_bkg_mask.bool()][:,0]) * -1])
        bkg_false = train_data.data_rec[train_data.data_rec_mask.bool()]
        weights_false = torch.ones_like(bkg_false[:,0])

        background_classifier_params = params["background_classifier_params"]

        background_classifier = Classifier(dim, background_classifier_params, logger, "background").to(train_data.device)
        logging.info("background classifier has", sum(p.numel() for p in background_classifier.parameters()), "trainable parameters.")

        background_classifier.train_classifier(bkg_true, bkg_false, weights_true, weights_false)

        background_path = os.path.join(run_dir, "models", f"background.pt")
        torch.save(background_classifier.state_dict(), background_path)

        test_background_weights = background_classifier.evaluate(test_data.data_rec[test_data.data_rec_mask.bool()])
        test_background_weights = test_background_weights * (len(
        test_data.data_rec[test_data.data_rec_mask.bool()]) - len(test_data.mc_bkg[test_data.mc_bkg_mask]))/ test_background_weights.sum()

        train_background_weights = background_classifier.evaluate(train_data.data_rec[train_data.data_rec_mask.bool()])
        train_background_weights = train_background_weights * (len(
        train_data.data_rec[train_data.data_rec_mask.bool()]) - len(train_data.mc_bkg[train_data.mc_bkg_mask]))/ test_background_weights.sum()
    else:
        train_data.data_rec = train_data.data_rec[:len(train_data.data_signal_rec)]
        train_data.data_rec_mask = train_data.data_rec_mask[:len(train_data.data_rec)]
        train_background_weights = torch.ones_like(train_data.data_rec[train_data.data_rec_mask.bool()][:,0])

        test_data.data_rec = test_data.data_rec[:len(test_data.data_signal_rec)]
        test_data.data_rec_mask = test_data.data_rec_mask[:len(test_data.data_rec)]
        test_background_weights = torch.ones_like(test_data.data_rec[test_data.data_rec_mask.bool()][:, 0])
    # Omnifold
    omnifold_params = params["omnifold_classifier_params"]
    for j in range(omnifold_params["iterations"]):
        if j ==0:
            logging.info("Initalize classifiers")
            logging.info(f"Starting with the {j}.iteration.")
            reco_classifier = Classifier(dim, omnifold_params["classifier"], logger, f"reco_iterative_{j}").to(train_data.device)
            gen_classifier = Classifier(dim, omnifold_params["classifier"], logger, f"gen_iterative_{j}").to(
                train_data.device)

            logging.info("reco_classifier has", sum(p.numel() for p in reco_classifier.parameters()),
                         "trainable parameters.")
            logging.info("gen_classifier has", sum(p.numel() for p in gen_classifier.parameters()),
                     "trainable parameters.")

            mc_gen = train_data.mc_gen[(train_data.mc_rec_mask.bool())]
            mc_rec = train_data.mc_rec[(train_data.mc_rec_mask.bool())]
            data_rec = train_data.data_rec[(train_data.data_rec_mask.bool())]
            step_one_weights = torch.ones_like(mc_rec[:, 0])
            step_two_weights = torch.ones_like(mc_rec[:, 0])
            test_step_one_weights = torch.ones_like(test_data.mc_rec[test_data.mc_rec_mask.bool()][:, 0])
            test_step_two_weights = torch.ones_like(test_data.mc_rec[test_data.mc_rec_mask.bool()][:, 0])

            data_weights = train_background_weights

            data_rec_inter, _, _, _, _ = test_data.apply_preprocessing(test_data.data_rec.clone(),
                                                                     parameters=[test_data.mean, test_data.std,
                                                                                 test_data.shift,
                                                                                 test_data.factor], reverse=True)
            mc_rec_inter, _, _, _, _ = test_data.apply_preprocessing(test_data.mc_rec.clone(),
                                                                     parameters=[test_data.mean, test_data.std,
                                                                                 test_data.shift,
                                                                                 test_data.factor], reverse=True)
            mc_gen_inter, _, _, _, _ = test_data.apply_preprocessing(test_data.mc_gen.clone(),
                                                                     parameters=[test_data.mean, test_data.std,
                                                                                 test_data.shift,
                                                                                 test_data.factor], reverse=True)

        reco_classifier.train_classifier(data_rec, mc_rec, data_weights, step_two_weights)
        step_one_weights = reco_classifier.evaluate(mc_rec) * step_two_weights
        test_step_one_weights = reco_classifier.evaluate(test_data.mc_rec[test_data.mc_rec_mask.bool()]) * test_step_two_weights

        with PdfPages(f"{plot_path}/step_one_omnifold_{j}.pdf") as out:
            for channel in params["channels"]:
                plot_reweighted_distribution(out, data_rec_inter[test_data.data_rec_mask.bool()][:, channel],
                                             mc_rec_inter[:, channel][(test_data.mc_rec_mask.bool())],
                                             mc_rec_inter[:, channel][(test_data.mc_rec_mask.bool())],
                                             reweighted_weights=test_step_one_weights.cpu(),
                                             true_weights= test_background_weights.cpu(),
                                             fake_weights= torch.ones_like(test_step_one_weights).cpu(),
                                             bins=test_data.observables[channel]["bins"],
                                             name=test_data.observables[channel]["tex_label"],
                                             yscale=test_data.observables[channel]["yscale"],
                                             density=True,
                                             labels=[r"$p_{d,s+b}(r)_r$", r"$w \cdot p_{MC,s}(r)_r$", r"$p_{MC,s}(r)_r$"])
                plot_reweighted_distribution(out, test_data.data_gen[:, channel][test_data.data_rec_mask[:len(test_data.data_signal_rec)].bool()],
                                             mc_gen_inter[:, channel][(test_data.mc_rec_mask.bool())],
                                             mc_gen_inter[:, channel][(test_data.mc_rec_mask.bool())],
                                             reweighted_weights=test_step_one_weights.cpu(),
                                             true_weights=torch.ones_like(test_data.data_gen[:, channel][test_data.data_rec_mask[:len(test_data.data_signal_rec)].bool()]),
                                             fake_weights= torch.ones_like(test_step_one_weights).cpu(),
                                             bins=test_data.observables[channel]["bins"],
                                             name=test_data.observables[channel]["tex_label"],
                                             yscale=test_data.observables[channel]["yscale"],
                                             density=True,
                                             labels=[r"$p_{d,s}(g)_r$", r"$w \cdot p_{MC,s}(g)_r$", r"$p_{MC,s}(g)_r$"])


        gen_classifier.train_classifier(mc_gen, mc_gen, weights_true=step_one_weights)
        step_two_weights = gen_classifier.evaluate(mc_gen)
        test_step_two_weights = gen_classifier.evaluate(test_data.mc_gen[test_data.mc_rec_mask.bool()])

        with PdfPages(f"{plot_path}/step_two_omnifold_{j}.pdf") as out:
            for channel in params["channels"]:
                plot_reweighted_distribution(out, data_rec_inter[test_data.data_rec_mask.bool()][:, channel],
                                             mc_rec_inter[:, channel][(test_data.mc_rec_mask.bool())],
                                             mc_rec_inter[:, channel][(test_data.mc_rec_mask.bool())],
                                             reweighted_weights=test_step_two_weights.cpu(),
                                             true_weights= test_background_weights.cpu(),
                                             fake_weights=torch.ones_like(test_step_two_weights).cpu(),
                                             bins=test_data.observables[channel]["bins"],
                                             name=test_data.observables[channel]["tex_label"],
                                             yscale=test_data.observables[channel]["yscale"],
                                             density=True,
                                             labels=[r"$p_{d,s+b}(r)_r$", r"$w \cdot p_{MC,s}(r)_r$", r"$p_{MC,s}(r)_r$"])
                plot_reweighted_distribution(out, test_data.data_gen[:, channel][test_data.data_rec_mask[:len(test_data.data_signal_rec)].bool()],
                                             mc_gen_inter[:, channel][(test_data.mc_rec_mask.bool())],
                                             mc_gen_inter[:, channel][(test_data.mc_rec_mask.bool())],
                                             reweighted_weights=test_step_two_weights.cpu(),
                                             true_weights=torch.ones_like(test_data.data_gen[:, channel][test_data.data_rec_mask[:len(test_data.data_signal_rec)].bool()]),
                                             fake_weights=torch.ones_like(test_step_two_weights).cpu(),
                                             bins=test_data.observables[channel]["bins"],
                                             name=test_data.observables[channel]["tex_label"],
                                             yscale=test_data.observables[channel]["yscale"],
                                             density=True,
                                             labels=[r"$p_{d,s}(g)_r$", r"$w \cdot p_{MC,s}(g)_r$", r"$p_{MC,s}(g)_r$"])

                d = calculate_triangle_distance(feed_dict=
                                            {"truth": test_data.data_gen[:, channel][test_data.data_gen_mask.bool()],
                                             "unfolded": mc_gen_inter[:, channel][(test_data.mc_rec_mask.bool())]},
                                            weights=
                                            {"truth": np.ones_like(test_data.data_gen[:, channel][test_data.data_gen_mask.bool()]),
                                             "unfolded": test_step_two_weights.cpu()},
                                            binning= test_data.observables[channel]["bins"], alternative_name="unfolded",
                                                reference_name="truth")

                logging.info(f"Triangle distance in {channel}.observable is {d}.")
    reco_classifier_path = os.path.join(run_dir, "models", f"reco_classifier.pt")
    torch.save(reco_classifier.state_dict(), reco_classifier_path)
    gen_classifier_path = os.path.join(run_dir, "models", f"gen_classifier.pt")
    torch.save(gen_classifier.state_dict(), gen_classifier_path)
    np.save(f"{run_dir}/final_weights.npy", test_step_two_weights.cpu() )
    efficiency_classifier_params = params["classifier_params"]
    #
    efficiency_true = train_data.mc_gen[(train_data.mc_rec_mask.bool())]
    efficiency_false = train_data.mc_gen[~(train_data.mc_rec_mask.bool())]
    #
    efficiency_classifier = Classifier(dim, efficiency_classifier_params, logger, "efficiency").to(train_data.device)
    logging.info("efficiency_classifier has", sum(p.numel() for p in efficiency_classifier.parameters()),
          "trainable parameters.")
    #
    efficiency_classifier.train_classifier(efficiency_true, efficiency_false, balanced=False)
    #
    efficiency_path = os.path.join(run_dir, "models", f"efficiency.pt")
    torch.save(efficiency_classifier.state_dict(), efficiency_path)
    #
    efficiency = efficiency_classifier.evaluate(test_data.mc_gen[(train_data.mc_rec_mask.bool())], return_weights=False)
    data_weights = test_step_two_weights / efficiency
    data_weights = data_weights.clip(0, 10)


    test_data.data_rec, _, _ ,_ , _ = test_data.apply_preprocessing(test_data.data_rec, parameters=[test_data.mean,
                                                                                       test_data.std,
                                                                                       test_data.shift,
                                                                                       test_data.factor], reverse=True)
    test_data.mc_gen, _, _ ,_ , _ = test_data.apply_preprocessing(test_data.mc_gen, parameters=[test_data.mean,
                                                                                       test_data.std,
                                                                                       test_data.shift,
                                                                                       test_data.factor], reverse=True)
    #
    # with PdfPages(f"{plot_path}/background_suppression.pdf") as out:
    #     for channel in params["channels"]:
    #         plot_reweighted_distribution(out, test_data.data_signal_rec[:, channel][test_data.data_rec_mask[:len(test_data.data_signal_rec)].bool()],
    #                                  test_data.data_rec[:, channel][test_data.data_rec_mask.bool()],
    #                                  test_data.data_rec[:, channel][test_data.data_rec_mask.bool()],
    #                                  reweighted_weights=test_background_weights.cpu(),
    #                                  bins=test_data.observables[channel]["bins"],
    #                                  labels=[r"$p_{d,s}(r)_r$", r"$\nu \cdot p_{d,s+b}(r)_r$", r"$p_{d,s+b}(r)_r$"],
    #                                  name=test_data.observables[channel]["tex_label"], yscale=test_data.observables[channel]["yscale"])
    #
    #
    # # %%
    #
    # with PdfPages(f"{plot_path}/efficiency_acceptance_effects.pdf") as out:
    #     for channel in params["channels"]:
    #         plot_reweighted_distribution(out, test_data.data_gen[:, channel][test_data.data_gen_mask.bool()],
    #                                      test_data.mc_gen[:, channel][(test_data.mc_rec_mask.bool())],
    #                                      test_data.mc_gen[:, channel][(test_data.mc_rec_mask.bool()) & (test_data.mc_gen_mask.bool())],
    #                                      reweighted_weights=data_weights.cpu()[test_data.mc_gen_mask.bool()],
    #                                      fake_weights=test_step_two_weights.cpu(),
    #                                      bins=test_data.observables[channel]["bins"],
    #                                      labels=[r"$p_{d,s}(g)_g$", r"$\text{unfolded} / \delta$", "unfolded"],
    #                                      name=test_data.observables[channel]["tex_label"],
    #                                      yscale=test_data.observables[channel]["yscale"])

    with PdfPages(f"{plot_path}/final_unfolding.pdf") as out:
        for channel in params["channels"]:
            plot_naive_unfold(out, test_data.data_gen[:, channel][test_data.data_gen_mask.bool()],
                              test_data.data_rec[(test_data.data_rec_mask.bool())][:, channel],
                              test_data.mc_gen[:, channel][(test_data.data_rec_mask.bool())& (test_data.mc_gen_mask.bool())],
                              unfolded_weights=data_weights.cpu()[test_data.mc_gen_mask.bool()],
                              bins=test_data.observables[channel]["bins"],
                              name=test_data.observables[channel]["tex_label"],yscale=test_data.observables[channel]["yscale"],
                              includes="acceptance")

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