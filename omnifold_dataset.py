import torch
import numpy as np
import h5py


class Omnifold:

    def __init__(self, params):
        self.params = params

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_dataset()
        self.init_observables()

    def init_dataset(self):
        path = self.params["path"]

        #MC signal
        mc_rec = np.load(f"{path}/reco_MC.npy")
        mc_gen = np.load(f"{path}/gen_MC.npy")
        #data signal
        data_rec = np.load(f"{path}/reco_data.npy")
        data_gen = np.load(f"{path}/gen_data.npy")
        #bkg
        bkg = np.load(f"{path}/reco_bkg.npy")

        self.mc_gen = torch.tensor(mc_gen).float()#.to(self.device)
        self.mc_rec = torch.tensor(mc_rec).float()#.to(self.device)

        self.data_gen = torch.tensor(data_gen).float()#.to(self.device)
        self.data_signal_rec = torch.tensor(data_rec).float()#.to(self.device)
        self.data_rec = self.data_signal_rec
        self.mc_bkg, self.data_bkg = torch.chunk(torch.tensor(bkg).float(),2)#.to(self.device),2)
        # self.data_rec = torch.cat([self.data_signal_rec, self.data_bkg])
        #
        # self.data_rec = self.data_rec[self.data_rec[:,0] < 70]
        # self.mc_bkg = self.mc_bkg[self.mc_bkg[:, 0] < 70]
        # self.data_bkg = self.data_bkg[self.data_bkg[:, 0] < 70]
        # self.data_gen = self.data_gen[self.data_gen[:,0]  < 70]
        # self.mc_rec = self.mc_rec[self.mc_rec[:,0]  < 70]
        # self.mc_gen = self.mc_gen[self.mc_gen[:,0]  < 70]

        self.data_gen_mask = self.data_gen[:, 6] > 150
        self.data_rec_mask = self.data_rec[:, 6] > 150
        self.mc_gen_mask = self.mc_gen[:, 6] > 150
        self.mc_rec_mask = self.mc_rec[:, 6] > 150
        self.mc_bkg_mask = self.mc_bkg[:, 6] > 150

        self.mc_rec, self.mean, self.std, self.shift, self.factor = self.apply_preprocessing(self.mc_rec)
        self.mc_gen, _ ,_, _, _= self.apply_preprocessing(self.mc_gen, parameters=[self.mean, self.std,self.shift, self.factor])
        self.mc_bkg, _ ,_, _,_ = self.apply_preprocessing(self.mc_bkg, parameters=[self.mean, self.std,self.shift, self.factor])

        self.data_rec, _ ,_,_,_ = self.apply_preprocessing(self.data_rec, parameters=[self.mean, self.std,self.shift, self.factor])

    def apply_preprocessing(self,dataset, parameters=None,reverse=False):
        if not reverse:
            dataset = dataset[:, self.params['channels']]
            # add noise to the jet multiplicity to smear out the integer structure
            dataset[:, 0] = torch.log(dataset[:, 0] + 1.e-4)
            dataset[:, 6] = torch.log(dataset[:, 6])
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
            # round jet multiplicity back to integers
            dataset[..., 1] = torch.round(dataset[..., 1])

            dataset[..., 5] = torch.erf(dataset[..., 5]) * factor + shift
            dataset[..., 5] = dataset[..., 5].exp()
            dataset[..., 5] = torch.where(dataset[..., 5] < 0.1, 0, dataset[..., 5])
            dataset[..., 0] = torch.exp(dataset[:, 0]) - 1.e-4
            dataset[..., 6] = torch.exp(dataset[:, 6])
            # if hasattr(self, "unfolded"):
            #     self.unfolded = self.unfolded * self.gen_std + self.gen_mean
            #     self.unfolded[..., 1] = torch.round(self.unfolded[..., 1])

        return dataset, mean, std, shift, factor

    def init_observables(self):
        self.observables = []

        self.observables.append({
            "tex_label": r"\text{Jet mass } m",
            "bins": torch.linspace(1, 90, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"\text{Jet multiplicity } N",
            "bins": torch.arange(3.5, 60.5),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"\text{Jet width } w",
            "bins": torch.linspace(0, 0.6, 50 + 1),
            "yscale": "log"
        })
        self.observables.append({
            "tex_label": r"\text{Groomed mass }\log \rho",
            "bins": torch.linspace(-14, -2, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"\text{N-subjettiness ratio } \tau_{21}",
            "bins": torch.linspace(0.1, 1.1, 50 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"\text{Groomed momentum fraction }z_g",
            "bins": torch.linspace(0.05, 0.55, 50 + 1),
            "yscale": "log"
        })
        self.observables.append({
            "tex_label": r"\text{Jet transverse momentum }p_T",
            "bins": torch.linspace(50, 500, 50 + 1),
            "yscale": "log"
        })

