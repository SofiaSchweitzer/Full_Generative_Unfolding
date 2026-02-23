import argparse

import numpy as np
import yaml
import torch
import os

from gaussian_toy import GaussianToy
import logging
from torch.utils.tensorboard import SummaryWriter
from matplotlib.backends.backend_pdf import PdfPages

from datetime import datetime
from models import CFM
from models import Classifier
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
    train_data = GaussianToy(params["data_params"])
    test_data = GaussianToy(params["data_params"])

    logging.info(f"Loaded mc data with shape {train_data.mc_rec.shape}, pseudo data with shape {train_data.data_rec.shape} and background train_data"
          f"with shape {train_data.mc_background_rec.shape}")
    logging.info(
        f"Applied mc cuts rec survived {train_data.mc_rec_mask.sum()}, gen survived {train_data.mc_gen_mask.sum()}, bkg survived "
        f"{train_data.mc_background_mask.sum()}")
    logging.info(
        f"Applied data cuts rec survived {train_data.data_rec_mask.sum()}.")
    logging.info(f"Using device:{train_data.device}")

    bkg_true = torch.cat([train_data.data_rec[train_data.data_rec_mask.bool()], train_data.mc_background_rec[train_data.mc_background_mask.bool()]])
    weights_true = torch.cat([torch.ones_like(train_data.data_rec[:, 0][train_data.data_rec_mask.bool()]),
                              torch.ones_like(train_data.mc_background_rec[:, 0][train_data.mc_background_mask.bool()]) * -1])
    bkg_false = train_data.data_rec[train_data.data_rec_mask.bool()]
    weights_false = torch.ones_like(bkg_false[:, 0])

    background_classifier_params = params["classifier_params"]
    background_classifier = Classifier(dim, background_classifier_params, logger, "background").to(
        train_data.device)
    background_classifier.train_classifier(bkg_true, bkg_false, weights_true, weights_false)
    background_path = os.path.join(run_dir, "models", f"background.pt")
    torch.save(background_classifier.state_dict(), background_path)

    background_weights_train = background_classifier.evaluate(train_data.data_rec[train_data.data_rec_mask.bool()])
    background_weights_train = background_weights_train * len(
        train_data.mc_rec[train_data.mc_rec_mask.bool()]) / background_weights_train.sum()

    background_weights_test = background_classifier.evaluate(test_data.data_rec[test_data.data_rec_mask.bool()])
    background_weights_test = background_weights_test * len(
        test_data.mc_rec[test_data.mc_rec_mask.bool()]) / background_weights_test.sum()

    np.save(f"{run_dir}/bkg_weights.npy", background_weights_test.cpu().numpy())
    if params["include_learned_acceptance"]:
        acceptance_true = train_data.mc_rec[(train_data.mc_rec_mask.bool()) & (train_data.mc_gen_mask.bool())]
        acceptance_false = train_data.mc_rec[(train_data.mc_rec_mask.bool()) & ~(train_data.mc_gen_mask.bool())]
        acceptance_classifier_params = params["classifier_params"]
        acceptance_classifier = Classifier(dim, acceptance_classifier_params, logger, "acceptance").to(
        train_data.device)
        acceptance_classifier.train_classifier(acceptance_true, acceptance_false, balanced=False)
        acceptance_train = acceptance_classifier.evaluate(train_data.data_rec[train_data.data_rec_mask.bool()], return_weights=False)
        acceptance_test = acceptance_classifier.evaluate(test_data.data_rec[test_data.data_rec_mask.bool()], return_weights=False)
        train_mc_gen_mask = train_data.mc_gen_mask
        test_mc_gen_mask = test_data.mc_gen_mask

        acceptance_path = os.path.join(run_dir, "models", f"acceptance.pt")
        torch.save(acceptance_classifier.state_dict(), acceptance_path)

        np.save(f"{run_dir}/acceptance_weights.npy", acceptance_test.cpu().numpy())

        train_data_gen_mask = train_data.data_gen_mask
        test_data_gen_mask =test_data.data_gen_mask
    else:
        acceptance_train = torch.ones_like(train_data.data_rec[: ,0][train_data.data_rec_mask.bool()])
        acceptance_test = torch.ones_like(test_data.data_rec[: ,0][test_data.data_rec_mask.bool()])
        train_mc_gen_mask = torch.ones_like(train_data.mc_gen[:, 0])
        test_mc_gen_mask = torch.ones_like(test_data.mc_gen[:, 0])

        train_data_gen_mask = torch.ones_like(train_data.data_gen[:, 0])
        test_data_gen_mask = torch.ones_like(test_data.data_gen[:, 0])

    with PdfPages(f"{plot_path}/background_suppression.pdf") as out:
        plot_reweighted_distribution(out, test_data.data_signal_rec.cpu()[:, 0][test_data.data_rec_mask[:len(test_data.data_signal_rec)].bool()],
                                     test_data.data_rec.cpu()[:, 0][test_data.data_rec_mask.bool()],
                                     test_data.data_rec.cpu()[:, 0][test_data.data_rec_mask.bool()],
                                     true_weights=torch.ones_like(test_data.data_signal_rec.cpu()[:, 0]
                                                                  [test_data.data_rec_mask[:len(test_data.data_signal_rec)].bool()]).cpu(),
                                     reweighted_weights=background_weights_test.cpu(),
                                     fake_weights = torch.ones_like(background_weights_test).cpu(),
                                     range=[-3, 4], bins=40,
                                     labels=[r"$p_{d,s}(x)_r$", r"$\nu \cdot p_{d,s+b}(x)_r$", r"$p_{d,s+b}(x)_r$"],
                                     name=r"\mathcal{O}")
    with PdfPages(f"{plot_path}/acceptance_effects.pdf") as out:
        plot_reweighted_distribution(out, test_data.data_signal_rec.cpu()[:, 0][
            (test_data.data_rec_mask[:len(test_data.data_signal_rec)].bool()) & (test_data.data_gen_mask.bool())],
                                     test_data.data_rec.cpu()[:, 0][test_data.data_rec_mask.bool()],
                                     test_data.data_rec.cpu()[:, 0][test_data.data_rec_mask.bool()],
                                     true_weights=torch.ones_like(test_data.data_signal_rec.cpu()[:, 0][
            (test_data.data_rec_mask[:len(test_data.data_signal_rec)].bool()) & (test_data.data_gen_mask.bool())]).cpu(),
                                     reweighted_weights=acceptance_test.cpu() * background_weights_test.cpu(),
                                     fake_weights=background_weights_test.cpu(),
                                     range=[-3, 4], bins=40,
                                     labels=[r"$p_{d,s}(x)_{r,g}$", r"$\delta \cdot \nu \cdot p_{d,s+b}(x)_r$", "$p_{d,s}(x)_r$"], name=r"\mathcal{O}")
    # %%

    iterative_unfolding_params = {"iterations": params["iterations"],
                                  "generator": params["unfolder_params"],
                                  "classifier": params["classifier_params"]}

    for i in range(iterative_unfolding_params["iterations"]):
        logging.info(f"Starting with the {i}.iteration.")
        if i == 0:
            logging.info("Initalize unfolder")
            unfolder = CFM(dim, dim, iterative_unfolding_params["generator"], logger).to(train_data.device)
            logging.info("unfolder has", sum(p.numel() for p in unfolder.parameters()),
                         "trainable parameters.")
            mc_gen = train_data.mc_gen[(train_mc_gen_mask.bool()) & (train_data.mc_rec_mask.bool())]
            mc_rec = train_data.mc_rec[(train_mc_gen_mask.bool()) & (train_data.mc_rec_mask.bool())]
            mc_weights_train = torch.ones_like(mc_rec[:, 0])
            mc_weights_test = torch.ones_like(test_data.mc_gen[:,0][(test_mc_gen_mask.bool()) & (test_data.mc_rec_mask.bool())])
            data_weights = acceptance_train * background_weights_train

        if i > 0:
            iterative_classifier = Classifier(
                dim, iterative_unfolding_params["classifier"],logger, f"iterative_{i}"
            ).to(train_data.device)
            logging.info("iterator has", sum(p.numel() for p in iterative_classifier.parameters()),
                         "trainable parameters.")
            iterative_classifier.train_classifier(data_unfold_train, mc_gen, data_weights, mc_weights_train)
            mc_weights_train *= iterative_classifier.evaluate(mc_gen)
            mc_weights_test *= iterative_classifier.evaluate(
                test_data.mc_rec[(test_mc_gen_mask.bool()) & (test_data.mc_rec_mask.bool())]
            )

        unfolder.train_unfolder(mc_gen, mc_rec, mc_weights_train)
        print("unfold data")
        unfolder_path = os.path.join(run_dir, "models", f"{i}th_unfolder.pt")
        torch.save(unfolder.state_dict(), unfolder_path)
        data_unfold_train = unfolder.evaluate(train_data.data_rec[train_data.data_rec_mask.bool()])
        data_unfold_test = unfolder.evaluate(test_data.data_rec[test_data.data_rec_mask.bool()])

        with PdfPages(f"{plot_path}/prior_dependence_{i}.pdf") as out:
            plot_prior_unfold(out, test_data.data_gen.cpu()[:, 0][
                    (test_data.data_rec_mask[:len(test_data.data_signal_rec)].bool()) & (test_data_gen_mask.bool())],
                                  test_data.mc_gen.cpu()[:, 0][(test_data.mc_rec_mask.bool()) & (test_mc_gen_mask.bool())],
                                  data_unfold_test.cpu()[:, 0],
                                gen_weights=torch.ones_like(test_data.data_gen.cpu()[:, 0][
                    (test_data.data_rec_mask[:len(test_data.data_signal_rec)].bool()) & (test_data_gen_mask.bool())]).cpu(),
                                  unfolded_weights=acceptance_test.cpu() * background_weights_test.cpu(),
                                  prior_weights=mc_weights_test.cpu(),
                                  range=[-3, 4], name=r"\mathcal{O}", density=True, bins=40)

    iterator_path = os.path.join(run_dir, "models", f"iterator.pt")
    torch.save(iterative_classifier.state_dict(), iterator_path)
    unfolder_path = os.path.join(run_dir, "models", f"unfolder.pt")
    torch.save(unfolder.state_dict(), unfolder_path)
    efficiency_classifier_params = params["classifier_params"]
    efficiency_classifier = Classifier(dim, efficiency_classifier_params, logger, "efficiency").to(
        train_data.device)
    logging.info("efficiency_classifier has", sum(p.numel() for p in efficiency_classifier.parameters()),
                 "trainable parameters.")

    efficiency_true = train_data.mc_gen[(train_data.mc_rec_mask.bool()) & (train_mc_gen_mask.bool())]
    efficiency_false = train_data.mc_gen[~(train_data.mc_rec_mask.bool()) & (train_mc_gen_mask.bool())]

    efficiency_classifier.train_classifier(efficiency_true, efficiency_false, balanced=False)

    efficiency_path = os.path.join(run_dir, "models", f"efficiency.pt")
    torch.save(efficiency_classifier.state_dict(), efficiency_path)

    efficiency_test = efficiency_classifier.evaluate(data_unfold_test, return_weights=False)

    np.save(f"{run_dir}/efficiency.npy", efficiency_test.cpu().numpy())

    data_weights =acceptance_test *background_weights_test / efficiency_test
    data_weights = data_weights.clip(0, 10)

    if params["include_learned_acceptance"]:
        unfolded_mask = torch.ones_like(data_unfold_test[:, 0]).cpu()
    else:
        unfolded_mask = ~((data_unfold_test > 1.2).squeeze())

    np.save(f"{run_dir}/unfolded.npy", data_unfold_test.cpu()[unfolded_mask.bool()])
    np.save(f"{run_dir}/weights.npy", data_weights.cpu()[unfolded_mask.bool()])


    with PdfPages(f"{plot_path}/prior_dependence_final.pdf") as out:
        plot_prior_unfold(out, test_data.data_gen.cpu()[:, 0][
                (test_data.data_rec_mask[:len(test_data.data_signal_rec)].bool()) & (test_data_gen_mask.bool())],
                              test_data.mc_gen.cpu()[:, 0][(test_data.mc_rec_mask.bool()) & (test_mc_gen_mask.bool())],
                              data_unfold_test.cpu()[:, 0],
                            gen_weights=torch.ones_like(test_data.data_gen.cpu()[:, 0][
                    (test_data.data_rec_mask[:len(test_data.data_signal_rec)].bool()) & (test_data_gen_mask.bool())]).cpu(),
                              unfolded_weights=acceptance_test.cpu() * background_weights_test.cpu(),
                              prior_weights=torch.ones_like(mc_weights_test).cpu(),
                              range=[-3, 4], name=r"\mathcal{O}", bins=40)

    with PdfPages(f"{plot_path}/efficiency_effects.pdf") as out:
        plot_reweighted_distribution(out, test_data.data_gen.cpu()[:, 0][test_data.data_gen_mask.bool()],
                                         data_unfold_test.cpu()[:, 0],
                                         data_unfold_test.cpu()[:, 0][unfolded_mask.bool()],
                                        true_weights=torch.ones_like(test_data.data_gen.cpu()[:, 0][test_data.data_gen_mask.bool()]).cpu(),
                                         reweighted_weights=data_weights.cpu()[unfolded_mask.bool()],
                                         fake_weights=acceptance_test.cpu() * background_weights_test.cpu(),
                                         labels=[r"$p_{d,s}(y)_g$", r"$\text{unfolded} / \epsilon$", "unfolded"],
                                         range=[-3,4], name=r"\mathcal{O}", bins=40)

    with PdfPages(f"{plot_path}/final_unfolding.pdf") as out:
        plot_naive_unfold(out, test_data.data_gen.cpu()[:, 0][(test_data.data_gen_mask.bool())],
                          test_data.data_rec.cpu()[:, 0][(test_data.data_rec_mask.bool())],
                          data_unfold_test.cpu()[:, 0][unfolded_mask.bool()],
                          gen_weights=torch.ones_like(test_data.data_gen.cpu()[:, 0][(test_data.data_gen_mask.bool())]).cpu(),
                          rec_weights=torch.ones_like(test_data.data_rec.cpu()[:, 0][(test_data.data_rec_mask.bool())]).cpu(),
                          unfolded_weights=data_weights.cpu()[unfolded_mask.bool()],
                          range=[-3, 4], name=r"\mathcal{O}", includes="acceptance", bins=40)


    logging.info("Finished.")
if __name__ == '__main__':
    main()