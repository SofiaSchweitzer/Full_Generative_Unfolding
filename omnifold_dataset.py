import torch
import numpy as np

import energyflow as ef
import os



class OmniFoldDataset:
    def __init__(self, params):
        self.params = params
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #Get OmniFold data
        self.get_omnifold(self.params["n_mc"],self.params["n_data"])
        #Load background sample
        self.get_background(self.params["n_background"])
        self.init_dataset()


        
    def get_substructure_obs(self,dataset):
        feature_names = ['widths','mults','sdms','zgs','tau2s']
        #Apply a 200 GeV cut to jet pT
        gen_mask = dataset['gen_jets'][:,0] > 200
        sim_mask = dataset['sim_jets'][:,0] > 200
        
        gen_features = [dataset['gen_jets'][:,3]]
        sim_features = [dataset['sim_jets'][:,3]]
        for feature in feature_names:
            gen_features.append(dataset['gen_'+feature])
            sim_features.append(dataset['sim_'+feature])


        gen_features = np.stack(gen_features,-1)
        sim_features = np.stack(sim_features,-1)
        #ln rho
        gen_features[:,3] = 2*np.ma.log(np.ma.divide(gen_features[:,3],dataset['gen_jets'][:,0]+10**-100).filled(0)).filled(0)
        sim_features[:,3] = 2*np.ma.log(np.ma.divide(sim_features[:,3],dataset['sim_jets'][:,0]+10**-100).filled(0)).filled(0)
        #tau21
        gen_features[:,5] = gen_features[:,5]/(10**-50 + gen_features[:,1])
        sim_features[:,5] = sim_features[:,5]/(10**-50 + sim_features[:,1])
        
        return tuple(torch.from_numpy(x).to(self.device, dtype=torch.float32) for x in (sim_features, gen_features, sim_mask, gen_mask))



    def get_omnifold(self,nmc=1_000_000, ndata = 1_000_000):
        zjets_mc = ef.zjets_delphes.load('Pythia21', num_data=nmc,
                                      cache_dir=self.params['path'],
                                      pad=False,
                                      exclude_keys=['particles','gen_particles'])

        zjets_data = ef.zjets_delphes.load('Herwig', num_data=ndata,
                                           cache_dir=self.params['path'],
                                           pad=False,
                                           exclude_keys=['particles','gen_particles'])
        
        self.mc_gen, self.mc_rec, self.mc_gen_mask, self.mc_rec_mask = self.get_substructure_obs(zjets_mc)
        self.data_gen, self.data_signal_rec, self.data_gen_mask, self.data_signal_rec_mask = self.get_substructure_obs(zjets_data)

    def get_background(self,nevts=100_000):
        file_name = os.path.join(self.params['path'],"zv_background.txt")
        data = {
            "gen_Zs": [], "gen_mults": [], "gen_jets": [], "gen_widths": [], "gen_tau2s": [], "gen_zgs": [], "gen_sdms": [],
            "sim_mults": [], "sim_jets": [], "sim_widths": [], "sim_tau2s": [], "sim_zgs": [], "sim_sdms": []
        }
        
        for line in open(file_name):
            parts = line.split()
            if len(parts) < 15:  # Ensure there are enough columns
                continue

            ZpT = float(parts[2])
            if ZpT <= 200:
                continue  # Skip if ZpT is not greater than 200

            # Determine if it's "truth" (gen) or "reco" (sim) data
            prefix = "gen" if "truth" in line else "sim" if "reco" in line else None
            if prefix is None:
                continue  # Skip lines that are neither "truth" nor "reco"

            # Extract required values
            mult = int(parts[14])
            jet = [float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])]
            width, tau2, zg, sdm = map(float, (parts[7], parts[8], parts[13], parts[12]))
            
            # Handle NaN in zg
            zg = zg if not np.isnan(zg) else 0.0

            # Append values to corresponding lists
            if prefix == "gen":
                data["gen_Zs"].append([ZpT, 0.0, 0.0])  # Only for gen-level
            data[f"{prefix}_mults"].append(mult)
            data[f"{prefix}_jets"].append(jet)
            data[f"{prefix}_widths"].append(width)
            data[f"{prefix}_tau2s"].append(tau2)
            data[f"{prefix}_zgs"].append(zg)
            data[f"{prefix}_sdms"].append(sdm)
        
        # Convert lists to NumPy arrays
        data = {key: np.array(value) for key, value in data.items()}
        _,self.data_background_rec, _, self.data_background_rec_mask = self.get_substructure_obs(data)
        
        #Split half of the background events for MC and for data
        self.mc_background_rec = self.data_background_rec[nevts:2*nevts]
        self.data_background_rec = self.data_background_rec[:nevts]

        self.mc_background_rec_mask = self.data_background_rec_mask[nevts:2*nevts]
        self.data_background_rec_mask = self.data_background_rec_mask[:nevts]

    def normalize(self,data):
        mean = torch.tensor([18.3763,  0.1527, 18.8572, -6.7593,  0.2396,  0.6759], device=self.device)
        std = torch.tensor([10.6587,  0.0993,  8.4504,  2.4235,  0.1192,  0.1958], device=self.device)
        return (data-mean)/std

    def init_dataset(self):
        
        self.data_signal_rec = self.normalize(self.data_signal_rec)
        self.data_gen = self.normalize(self.data_gen)
        self.mc_rec = self.normalize(self.mc_rec)
        self.mc_gen = self.normalize(self.mc_gen)

        self.data_background_rec = self.normalize(self.data_background_rec)
        self.mc_background_rec = self.normalize(self.mc_background_rec)

        self.data_rec = torch.cat([self.data_signal_rec, self.data_background_rec])
        self.data_rec_mask = torch.cat([self.data_signal_rec_mask, self.data_background_rec_mask])
        

        self.mc_rec[~self.mc_rec_mask.bool()] = self.params["empty_value"]*torch.ones_like(self.mc_rec[~self.mc_rec_mask.bool()])
        self.mc_gen[~self.mc_gen_mask.bool()] = self.params["empty_value"]*torch.ones_like(self.mc_gen[~self.mc_gen_mask.bool()])                    
        self.data_rec[~self.data_rec_mask.bool()] = self.params["empty_value"]*torch.ones_like(self.data_rec[~self.data_rec_mask.bool()])
        self.data_gen[~self.data_gen_mask.bool()] = self.params["empty_value"]*torch.ones_like(self.data_gen[~self.data_gen_mask.bool()])
=======
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
        # self.data_rec = self.data_signal_rec
        self.mc_bkg, self.data_bkg = torch.chunk(torch.tensor(bkg).float(),2)#.to(self.device),2)
        self.data_rec = torch.cat([self.data_signal_rec, self.data_bkg])
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
            dataset[:, 0] = (dataset[:, 0] + 1.e-3).log()
            dataset[:, 6] = (dataset[:, 6] - 11.6).log()
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
            dataset[..., 0] = (dataset[..., 0]).exp() - 1.e-3
            dataset[..., 6] = (dataset[..., 6]).exp() + 11.6
            # if hasattr(self, "unfolded"):
            #     self.unfolded = self.unfolded * self.gen_std + self.gen_mean
            #     self.unfolded[..., 1] = torch.round(self.unfolded[..., 1])


        return dataset, mean, std, shift, factor

    def init_observables(self):
        self.observables = []

        self.observables.append({
            "tex_label": r"\text{Jet mass } m",
            "bins": torch.linspace(1, 60, 45 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"\text{Jet multiplicity } N",
            "bins": torch.arange(3.5, 60.5),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"\text{Jet width } w",
            "bins": torch.linspace(0, 0.6, 45 + 1),
            "yscale": "log"
        })
        self.observables.append({
            "tex_label": r"\text{Groomed mass }\log \rho",
            "bins": torch.linspace(-13, -2, 45 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"\text{N-subjettiness ratio } \tau_{21}",
            "bins": torch.linspace(0.1, 1.1, 45 + 1),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"\text{Groomed momentum fraction }z_g",
            "bins": torch.linspace(0.05, 0.55, 45 + 1),
            "yscale": "log"
        })
        self.observables.append({
            "tex_label": r"\text{Jet transverse momentum }p_T",
            "bins": torch.linspace(50, 500, 45 + 1),
            "yscale": "log"
        })

        # self.observables = self.observables[:6]


