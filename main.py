import argparse
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
from plots import plot_naive_unfold, plot_reweighted_distribution, plot_prior_unfold

def setup_logging(run_dir):
    log_file = os.path.join(run_dir, "run.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",  # Clean format without timestamp
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
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
    params["classifier_params"]["run_dir"] = run_dir
    dim = params["dim"]
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    data = Omnifold({"path": params["data_path"], "channels": params["channels"]})
    logging.info(f"Loaded mc data with shape {data.mc_rec.shape}, pseudo data with shape {data.data_rec.shape} and background data"
          f"with shape {data.mc_bkg.shape}")
    logging.info(
        f"Applied mc cuts rec survived {data.mc_rec_mask.sum()}, gen survived {data.mc_gen_mask.sum()}, bkg survived "
        f"{data.mc_bkg_mask.sum()}")
    logging.info(
        f"Applied data cuts rec survived {data.data_rec_mask.sum()}, gen survived {data.data_gen_mask.sum()}.")
    logging.info(f"Using device:{data.device}")
    if params["include_bkg"]:
        #background classifier
        bkg_true = torch.cat([data.data_rec[data.data_rec_mask.bool()][:,params["channels"]], data.mc_bkg[data.mc_bkg_mask.bool()][:,params["channels"]]])
        weights_true = torch.cat([torch.ones_like(data.data_rec[:,0][data.data_rec_mask.bool()]), torch.ones_like(data.mc_bkg[data.mc_bkg_mask.bool()][:,0]) * -1])
        bkg_false = data.data_rec[data.data_rec_mask.bool()][:,params["channels"]]
        weights_false = torch.ones_like(bkg_false[:,0])

        background_classifier_params = params["background_classifier_params"]

        background_classifier = Classifier(dim, background_classifier_params, logger, "background").to(data.device)
        logging.info("background classifier has", sum(p.numel() for p in background_classifier.parameters()), "trainable parameters.")

        background_classifier.train_classifier(bkg_true, bkg_false, weights_true, weights_false)

        background_path = os.path.join(run_dir, "models", f"background.pt")
        torch.save(background_classifier.state_dict(), background_path)

        background_weights = background_classifier.evaluate(data.data_rec[data.data_rec_mask.bool()][:,params["channels"]])
        background_weights = background_weights * (len(
        data.data_rec[data.data_rec_mask.bool()]) - len(data.mc_bkg[data.mc_bkg_mask]))/ background_weights.sum()
    else:
        data.data_rec = data.data_signal_rec
        data.data_rec_mask = data.data_rec_mask[:len(data.data_rec)]
        background_weights = torch.ones_like(data.data_rec[data.data_rec_mask.bool()][:,0])
    # #Iterative unfolder
    iterative_unfolding_params = {"iterations": params["iterations"],
                                  "generator": params["unfolder_params"],
                                  "classifier": params["classifier_params"]}

    for j in range(iterative_unfolding_params["iterations"]):
        logging.info(f"Starting with the {j}.iteration.")
        if j == 0:
            logging.info("Initalize unfolder")
            unfolder = CFM(dim, dim, iterative_unfolding_params["generator"], logger).to(data.device)
            logging.info("unfolder has", sum(p.numel() for p in unfolder.parameters()),
                  "trainable parameters.")
            mc_gen = data.mc_gen[(data.mc_rec_mask.bool())]
            mc_rec = data.mc_rec[(data.mc_rec_mask.bool())]
            mc_weights = torch.ones_like(mc_rec[:, 0])
            data_weights = background_weights
        if j > 0:
            iterative_classifier = Classifier(dim-1, iterative_unfolding_params["classifier"],logger, f"iterative_{j}").to(data.device)
            logging.info("iterator has", sum(p.numel() for p in iterative_classifier.parameters()),
                  "trainable parameters.")
            iterative_classifier.train_classifier(data_unfold[:,:6], mc_gen[:,:6], data_weights, mc_weights)
            mc_weights *= iterative_classifier.evaluate(mc_gen[:,:6])
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
                for i, observable in enumerate(data.observables):
                    plot_prior_unfold(out,  data_unfold_inter[:, i], data.mc_gen_inter[:, i][(data.mc_rec_mask.bool())],
                                      data.mc_gen_inter[:, i][(data.mc_rec_mask.bool())],
                                      prior_weights=mc_weights.cpu(),
                                      bins=observable["bins"], name=observable["tex_label"],
                                      yscale=observable["yscale"],
                                      density=True)
        unfolder.train_unfolder(mc_gen, mc_rec, mc_weights)
        logging.info("unfold data")
        data_unfold = unfolder.evaluate(data.data_rec[data.data_rec_mask.bool()])
        mc_unfold = unfolder.evaluate(mc_rec)
        with PdfPages(f"{plot_path}/prior_dependence_{j}.pdf") as out:
            data_unfold_inter, _, _, _,_ = data.apply_preprocessing(data_unfold.clone(), parameters=[data.mean, data.std, data.shift, data.factor], reverse=True)
            data.mc_gen_inter, _, _, _, _ = data.apply_preprocessing(data.mc_gen.clone(), parameters=[data.mean, data.std, data.shift, data.factor], reverse=True)
            data.mc_rec_inter, _, _, _, _ = data.apply_preprocessing(data.mc_rec.clone(),
                                                                     parameters=[data.mean, data.std, data.shift,
                                                                                 data.factor], reverse=True)
            mc_unfold, _, _, _, _ = data.apply_preprocessing(mc_unfold,
                                                                     parameters=[data.mean, data.std, data.shift,
                                                                                 data.factor], reverse=True)
            for i, observable in enumerate(data.observables):
                plot_prior_unfold(out, data.data_gen[:, i][data.data_rec_mask[:len(data.data_signal_rec)].bool()],
                                  data.mc_gen_inter[:, i][(data.mc_rec_mask.bool())], data_unfold_inter[:, i],
                                  unfolded_weights=background_weights.cpu(), prior_weights=mc_weights.cpu(),
                                  bins=observable["bins"], name=observable["tex_label"], yscale=observable["yscale"],
                                  density=True)
                plot_naive_unfold(out, data.mc_gen_inter[:, i][(data.mc_rec_mask.bool())],
                                  data.mc_rec_inter[:, i][(data.mc_rec_mask.bool())], mc_unfold[:, i],
                                  gen_weights=mc_weights.cpu(),
                                  bins=observable["bins"], name=observable["tex_label"], yscale=observable["yscale"],
                                  density=True)
    #
    iterator_path = os.path.join(run_dir, "models", f"iterator.pt")
    torch.save(iterative_classifier.state_dict(), iterator_path)
    unfolder_path = os.path.join(run_dir, "models", f"unfolder.pt")
    torch.save(unfolder.state_dict(), unfolder_path)

    efficiency_classifier_params = params["classifier_params"]

    efficiency_true = data.mc_gen[(data.mc_rec_mask.bool())]
    efficiency_false = data.mc_gen[~(data.mc_rec_mask.bool())]

    efficiency_classifier = Classifier(dim, efficiency_classifier_params, logger, "efficiency").to(data.device)
    logging.info("efficiency_classifier has", sum(p.numel() for p in efficiency_classifier.parameters()),
          "trainable parameters.")

    efficiency_classifier.train_classifier(efficiency_true, efficiency_false, balanced=False)

    efficiency_path = os.path.join(run_dir, "models", f"efficiency.pt")
    torch.save(efficiency_classifier.state_dict(), efficiency_path)

    efficiency = efficiency_classifier.evaluate(data_unfold, return_weights=False)
    data_weights = background_weights / efficiency
    data_weights = data_weights.clip(0, 10)

    data_unfold, _, _, _, _ = data.apply_preprocessing(data_unfold, parameters=[data.mean, data.std, data.shift, data.factor], reverse=True)
    data.data_rec, _,_,_, _  = data.apply_preprocessing(data.data_rec, parameters=[data.mean, data.std, data.shift, data.factor], reverse=True)
    data.mc_gen, _, _,_,_ = data.apply_preprocessing(data.mc_gen, parameters=[data.mean, data.std, data.shift, data.factor], reverse=True)

    unfolded_mask = ((data_unfold[:,6] > 150).squeeze())



    with PdfPages(f"{plot_path}/background_suppression.pdf") as out:
        for i, observable in enumerate(data.observables):
            plot_reweighted_distribution(out, data.data_signal_rec[:, i][data.data_rec_mask[:len(data.data_signal_rec)].bool()],
                                     data.data_rec[:, i][data.data_rec_mask.bool()],
                                     data.data_rec[:, i][data.data_rec_mask.bool()],
                                     reweighted_weights=background_weights.cpu(),
                                     bins=observable["bins"], labels=[r"$\text{signal} |_r$", "reweighted", "data $(s+b)|_r$"],
                                     name=observable["tex_label"], yscale=observable["yscale"])


    # %%
    with PdfPages(f"{plot_path}/prior_dependence_final.pdf") as out:
        for i, observable in enumerate(data.observables):
            plot_prior_unfold(out, data.data_gen[:, i][data.data_rec_mask[:len(data.data_signal_rec)].bool()],
                              data.mc_gen[:, i][(data.mc_rec_mask.bool())], data_unfold[:, i],
                            unfolded_weights=background_weights.cpu(),
                              bins=observable["bins"], name=observable["tex_label"],yscale=observable["yscale"])

    with PdfPages(f"{plot_path}/efficiency_effects.pdf") as out:
        for i, observable in enumerate(data.observables):
            plot_reweighted_distribution(out, data.data_gen[:, i],
                                         data_unfold[:, i],
                                         data_unfold[:, i],
                                         reweighted_weights=data_weights.cpu(),
                                         fake_weights=background_weights.cpu(),
                                         bins=observable["bins"],
                                         labels=[r"$\text{gen}|_g$", r"$\text{unfolded} / \delta$", "unfolded"],
                                         name=observable["tex_label"],yscale=observable["yscale"])

    with PdfPages(f"{plot_path}/final_unfolding.pdf") as out:
        for i, observable in enumerate(data.observables):
            plot_naive_unfold(out, data.data_gen[:, i][data.data_gen_mask.bool()],
                              data.data_rec[(data.data_rec_mask.bool())][:, i],
                              data_unfold[:, i][unfolded_mask],
                              unfolded_weights=data_weights.cpu()[unfolded_mask],
                              bins=observable["bins"], name=observable["tex_label"],yscale=observable["yscale"])

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