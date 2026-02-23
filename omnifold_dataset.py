import torch
import numpy as np
# import h5py


class Omnifold:

    def __init__(self, params, dataset_type="train"):
        self.params = params

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_type = dataset_type

        self.init_dataset()
        if self.dataset_type == "test":
            self.init_observables()

    def init_dataset(self):
        path = self.params["path"]

        #MC signal
        mc_rec = np.load(f"{path}/reco_MC_{self.dataset_type}.npy")
        mc_gen = np.load(f"{path}/gen_MC_{self.dataset_type}.npy")
        #data signal
        data_rec = np.load(f"{path}/reco_MC_{self.dataset_type}.npy")

        #bkg
        bkg = np.load(f"{path}/reco_bkg.npy")


        self.mc_rec = torch.tensor(mc_rec).float()
        self.mc_gen = torch.tensor(mc_gen).float()


        self.data_signal_rec = torch.tensor(data_rec).float()
        # self.data_rec = self.data_signal_rec
        self.mc_bkg, self.data_bkg = torch.chunk(torch.tensor(bkg).float(),2)



        self.data_rec = torch.cat([self.data_signal_rec, self.data_bkg])



        self.data_rec_mask = self.data_rec[:, 6] > 150
        self.mc_gen_mask = self.mc_gen[:, 6] > 150
        self.mc_rec_mask = self.mc_rec[:, 6] > 150
        self.mc_bkg_mask = self.mc_bkg[:, 6] > 150



        self.mc_rec, self.mean, self.std, self.shift, self.factor = self.apply_preprocessing(self.mc_rec)

        self.mc_bkg, _ ,_, _,_ = self.apply_preprocessing(self.mc_bkg, parameters=[self.mean, self.std,self.shift, self.factor])

        self.data_rec, _ ,_,_,_ = self.apply_preprocessing(self.data_rec, parameters=[self.mean, self.std,self.shift, self.factor])
        self.mc_gen, _, _, _, _ = self.apply_preprocessing(self.mc_gen, parameters=[self.mean, self.std, self.shift,
                                                                                    self.factor])

        data_gen = np.load(f"{path}/gen_MC_{self.dataset_type}.npy")
        self.data_gen = torch.tensor(data_gen).float()
        if self.dataset_type == "test":
            self.data_gen_mask = self.data_gen[:, 6] > 150
        else:
            self.data_gen, _ ,_ ,_,_ = self.apply_preprocessing(self.data_gen, parameters=[self.mean, self.std, self.shift,
                                                                                    self.factor])

        self.data_gen_weights = np.load(f"{path}/{self.dataset_type}_weights_6d.npy")
        # self.data_gen_weights = torch.ones_like(self.data_gen[:, 0])
        self.data_gen_weights = torch.tensor(self.data_gen_weights).float().to(self.device)

        self.data_rec_weights = torch.cat([self.data_gen_weights, torch.ones_like(self.data_bkg[:,0]).to(self.device)])

    def apply_preprocessing(self,dataset, parameters=None,reverse=False):
        if not reverse:

            # add noise to the jet multiplicity to smear out the integer structure
            # dataset[:, 0] = (dataset[:, 0] + 1.e-3).log()
            # dataset[:, 6] = (dataset[:, 6] - 11.6).log()
            dataset[:, 0] = (dataset[:, 0] + 15).log()
            dataset[:, 6] = (dataset[:, 6] ).log() #-5
            noise = torch.rand_like(dataset[:, 1]) - 0.5
            dataset[:, 1] = dataset[:, 1] + noise
            noise = torch.rand(size=dataset[:, 5].shape, device=dataset[:, 5].device)/1000. * 3 + 0.097
            dataset[:, 5] = torch.where(dataset[:, 5] < 0.1, noise, dataset[:, 5])
            dataset[:, 5] = dataset[:, 5].log()

            try:
                shift = parameters[2]
            except:
                shift = (dataset[:, 5].max() + dataset[:, 5].min())/2.
            dataset[:, 5] = dataset[:, 5]-shift
            try:
                factor = parameters[3]
            except:
                factor = max(dataset[:, 5].max(), -1 * dataset[:, 5].min())*1.001
            dataset[:, 5] = dataset[:, 5]/factor
            dataset[:, 5] = torch.erfinv(dataset[:, 5])
            dataset = dataset[:, self.params['channels']]
            # standardize events
            try:
                mean = parameters[0]
                std = parameters[1]
            except:
                mean = dataset.mean(0)
                std = dataset.std(0)

            dataset = ((dataset - mean)/std).to(self.device)


        else:
            # undo standardization
            mean = parameters[0]
            std = parameters[1]
            shift = parameters[2]
            factor = parameters[3]
            dataset = dataset.cpu() * std + mean
            zeros = torch.zeros((len(dataset), 7))
            zeros[:, self.params['channels']] = dataset
            dataset = zeros
            # round jet multiplicity back to integers
            dataset[..., 1] = torch.round(dataset[..., 1])

            dataset[..., 5] = torch.erf(dataset[..., 5]) * factor + shift
            dataset[..., 5] = dataset[..., 5].exp()
            dataset[..., 5] = torch.where(dataset[..., 5] < 0.1, 0, dataset[..., 5])
            # dataset[..., 0] = (dataset[..., 0]).exp() - 1.e-3
            # dataset[..., 6] = (dataset[..., 6]).exp() + 11.6
            dataset[..., 0] = (dataset[..., 0]).exp() - 15
            dataset[..., 6] = (dataset[..., 6]).exp() #+ 5
            # if hasattr(self, "unfolded"):
            #     self.unfolded = self.unfolded * self.gen_std + self.gen_mean
            #     self.unfolded[..., 1] = torch.round(self.unfolded[..., 1])


        return dataset, mean, std, shift, factor

    def init_observables(self):
        self.observables = []

        self.observables.append({
            "tex_label": r"\text{Jet mass } m \;[\text{GeV}]",
            "bins": torch.linspace(1, 60, 40 + 1),
            "yscale": "linear",
            "leg_pos": "upper right"
        })
        self.observables.append({
            "tex_label": r"\text{Jet multiplicity } N",
            "bins": torch.arange(3.5, 60.5, 2),
            "yscale": "linear",
            "leg_pos": "upper right"
        })
        self.observables.append({
            "tex_label": r"\text{Jet width } w",
            "bins": torch.linspace(0, 0.6, 40 + 1),
            "yscale": "log",
            "leg_pos": "lower center"
        })
        self.observables.append({
            "tex_label": r"\text{Groomed mass }\log \rho",
            "bins": torch.linspace(-13, -1.5, 40 + 1),
            "yscale": "linear",
            "leg_pos": "lower center"
        })
        self.observables.append({
            "tex_label": r"\text{N-subjettiness ratio } \tau_{21}",
            "bins": torch.linspace(0.2, 1.3, 40 + 1),
            "yscale": "linear",
            "leg_pos": "lower center"
        })
        self.observables.append({
            "tex_label": r"\text{Groomed momentum fraction }z_g",
            "bins": torch.linspace(0.05, 0.55, 45 + 1),
            "yscale": "log",
            "leg_pos": "upper right"
        })
        self.observables.append({
            "tex_label": r"\text{Jet transverse momentum }p_T",
            "bins": torch.linspace(50, 500, 40+ 1),
            "yscale": "log",
            "leg_pos": "upper right"
        })



