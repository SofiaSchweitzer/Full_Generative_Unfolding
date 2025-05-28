import torch
import numpy as np
import h5py


class Omnifold:
    def __init__(self, path, nmc,ndata,nbkg,empty_value= -10.,val=False, use_weights = False):
        self.path = path
        self.val = val
        self.nmc = nmc
        self.ndata = ndata
        self.nbkg = nbkg
        self.empty_value=empty_value
        self.cut = 150.
        self.use_weights = use_weights
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_dataset()
        self.init_observables()

    def revert(self,dataset):
        d, _ ,_, _, _= self.apply_preprocessing(dataset, parameters=[self.mean, self.std,self.shift, self.factor],reverse=True)
        return d.cpu().detach().numpy()

    def set_preprocess_params(self):
        self.mean = torch.tensor([2.7450, 17.8664,  0.1363, -7.0481,  0.6714, -0.0195])
        self.std = torch.tensor([0.5391, 8.0726, 0.0942, 2.2142, 0.2056, 0.7286])
        self.shift = torch.tensor(-1.5131)
        self.factor = torch.tensor(0.8208)

    def init_dataset(self):
        path = self.path

        if self.val:
            #MC signal
            mc_rec = np.load(f"{path}/reco_MC.npy")[self.nmc:]
            mc_gen = np.load(f"{path}/gen_MC.npy")[self.nmc:]
            #data signal
            if self.use_weights:
                data_rec = np.load(f"{path}/reco_MC.npy")[self.ndata:] 
                data_gen = np.load(f"{path}/gen_MC.npy")[self.ndata:]  
            else:
                data_rec = np.load(f"{path}/reco_data.npy")[self.ndata:]
                data_gen = np.load(f"{path}/gen_data.npy")[self.ndata:]
        else:
            #MC signal
            mc_rec = np.load(f"{path}/reco_MC.npy")[:self.nmc]
            mc_gen = np.load(f"{path}/gen_MC.npy")[:self.nmc]
            #data signal
            if self.use_weights:
                data_rec = np.load(f"{path}/reco_MC.npy")[:self.ndata]
                data_gen = np.load(f"{path}/gen_MC.npy")[:self.ndata]
            else:
                data_rec = np.load(f"{path}/reco_data.npy")[:self.ndata]
                data_gen = np.load(f"{path}/gen_data.npy")[:self.ndata]

        bkg = np.load(f"{path}/reco_bkg.npy")[:self.nbkg]

        self.mc_gen = torch.tensor(mc_gen).float()#.to(self.device)
        self.mc_rec = torch.tensor(mc_rec).float()#.to(self.device)

        self.data_gen = torch.tensor(data_gen).float()#.to(self.device)
        self.data_signal_rec = torch.tensor(data_rec).float()#.to(self.device)
        # self.data_rec = self.data_signal_rec
        self.mc_bkg, self.data_bkg = torch.chunk(torch.tensor(bkg).float(),2)#.to(self.device),2)


        self.apply_cuts(self.cut)
        self.set_preprocess_params()
            
        self.mc_rec, _, _, _, _ = self.apply_preprocessing(self.mc_rec, parameters=[self.mean, self.std,self.shift, self.factor])
        self.mc_gen, _ ,_, _, _= self.apply_preprocessing(self.mc_gen, parameters=[self.mean, self.std,self.shift, self.factor])
        self.mc_bkg, _ ,_, _,_ = self.apply_preprocessing(self.mc_bkg, parameters=[self.mean, self.std,self.shift, self.factor])

        self.data_rec, _ ,_,_,_ = self.apply_preprocessing(self.data_rec, parameters=[self.mean, self.std,self.shift, self.factor])
        self.data_gen, _ ,_,_,_ = self.apply_preprocessing(self.data_gen, parameters=[self.mean, self.std,self.shift, self.factor])
        self.data_signal_rec, _ ,_,_,_ = self.apply_preprocessing(self.data_signal_rec, parameters=[self.mean, self.std,self.shift, self.factor])

        #Change default masked values based on empty_value
        self.mc_rec[~self.mc_rec_mask.bool()] = self.empty_value*torch.ones_like(self.mc_rec[~self.mc_rec_mask.bool()])
        self.mc_gen[~self.mc_gen_mask.bool()] = self.empty_value*torch.ones_like(self.mc_gen[~self.mc_gen_mask.bool()])
        self.data_rec[~self.data_rec_mask.bool()] = self.empty_value*torch.ones_like(self.data_rec[~self.data_rec_mask.bool()])
        self.data_gen[~self.data_gen_mask.bool()] = self.empty_value*torch.ones_like(self.data_gen[~self.data_gen_mask.bool()])
                    

    def apply_cuts(self,cut_val=150):
        
        self.data_gen_mask = self.data_gen[:, 6] > cut_val
        self.data_sig_mask = self.data_signal_rec[:, 6] > cut_val
        
        if self.use_weights:
            weights = torch.tensor(np.load('weights/train_weights.npy')[:self.ndata])
            self.data_gen_weights = weights.to(self.device)
            self.sig_rec_weights = weights.to(self.device)
        else:
            self.data_gen_weights = torch.ones_like(self.data_gen[:,0]).to(self.device)
            self.sig_rec_weights = torch.ones_like(self.data_gen[:,0]).to(self.device)

        #Remove events if both reco and gen do not pass the cuts
        mask = (self.data_gen_mask) | (self.data_sig_mask)
        self.data_gen_mask = self.data_gen_mask[mask]
        self.data_sig_mask = self.data_sig_mask[mask]
        self.data_gen = self.data_gen[mask]
        self.data_signal_rec = self.data_signal_rec[mask]
        self.data_gen_weights = self.data_gen_weights[mask]
        self.sig_rec_weights = self.sig_rec_weights[mask]
        
        self.data_rec = torch.cat([self.data_signal_rec, self.data_bkg])
        self.data_rec_weight = torch.cat([self.sig_rec_weights,
                                          torch.ones_like(self.data_bkg[:,0]).to(self.device)])
        

        self.mc_gen_mask = self.mc_gen[:, 6] > cut_val
        self.mc_rec_mask = self.mc_rec[:, 6] > cut_val
        
        mask = (self.mc_gen_mask) | (self.mc_rec_mask)
        self.mc_gen_mask = self.mc_gen_mask[mask]
        self.mc_rec_mask = self.mc_rec_mask[mask]
        self.mc_gen = self.mc_gen[mask]
        self.mc_rec = self.mc_rec[mask]
        
        self.mc_bkg_mask = self.mc_bkg[:, 6] > cut_val
        self.data_bkg_mask = self.data_bkg[:, 6] > cut_val

        self.data_rec_mask = torch.cat([self.data_sig_mask, self.data_bkg_mask])
        
        self.epsilon = 1.0 - 1.0*self.mc_rec_mask.sum()/self.mc_gen_mask.shape[0]
        self.delta = 1.0 - 1.0*self.mc_gen_mask.sum()/self.mc_gen_mask.shape[0]

        

    def apply_preprocessing(self,dataset, parameters=None,reverse=False):
        if not reverse:
            # add noise to the jet multiplicity to smear out the integer structure
            dataset[:, 0] = (dataset[:, 0] + 1.e-3).log()
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

            #Remove pT from the input list to match the standard omnifold

            dataset = dataset[:,:6]

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

        return dataset, mean, std, shift, factor

    def init_observables(self):
        self.observables = []

        self.observables.append({
            "tex_label": r"\text{Jet mass } m",
            "bins": torch.linspace(0, 75, 50),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"\text{Jet multiplicity } N",
            "bins": torch.arange(0, 80),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"\text{Jet width } w",
            "bins": torch.linspace(0, 0.6, 50),
            "yscale": "log"
        })
        self.observables.append({
            "tex_label": r"\text{Groomed mass }\log \rho",
            "bins": torch.linspace(-14, -2, 50),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"\text{N-subjettiness ratio } \tau_{21}",
            "bins": torch.linspace(0.0, 1.2, 50),
            "yscale": "linear"
        })
        self.observables.append({
            "tex_label": r"\text{Groomed momentum fraction }z_g",
            "bins": torch.linspace(0.0, 0.5, 50),
            "yscale": "log"
        })

