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
