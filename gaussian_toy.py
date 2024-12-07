import torch
import numpy as np


class GaussianToy:

    def __init__(self, params):
        self.params = params
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.init_dataset()

    def apply_detector(self, x_gen):
        detector_effect = torch.randn_like(x_gen) * self.params["detector_sigma"] + self.params["detector_mu"]
        return x_gen + detector_effect.to(self.device)

    def apply_efficiency_acceptance_effects(self, x, factor = 0.1):
        pass_reco = np.random.binomial(1, 1. - factor, len(x))
        return torch.from_numpy(pass_reco).to(self.device)
    
    def gaussian(self,nevts,dim,mu,sigma):
        return (torch.randn(nevts,dim) * sigma + mu).to(self.device)

    def init_dataset(self):
        self.mc_gen = self.gaussian(self.params["n_mc"], self.params["n_dim"],self.params["mc_mu"],self.params["mc_sigma"])
        self.mc_rec = self.apply_detector(self.mc_gen)

        self.data_gen =  self.gaussian(self.params["n_data"], self.params["n_dim"],self.params["data_mu"],self.params["data_sigma"])
        #data_mus = torch.rand(self.params["n_dim"])*10. - 5.
        #data_sigmas = torch.rand(self.params["n_dim"]) * 2. + 3
        #print(data_mus, data_sigmas)
        #self.data_gen = (
        #            torch.randn(self.params["n_data"], self.params["n_dim"]) * data_sigmas + data_mus).to(self.device)
        self.data_signal_rec = self.apply_detector(self.data_gen).to(self.device)

        if self.params["n_background"] != 0:
            self.data_background_rec = self.gaussian(self.params["n_background"], self.params["n_dim"],self.params["background_mu"],self.params["background_sigma"])
            self.mc_background_rec = self.gaussian(self.params["n_background"], self.params["n_dim"],self.params["background_mu"],self.params["background_sigma"])            
            self.data_rec = torch.cat([self.data_signal_rec, self.data_background_rec])
                                                                              
        else:
            self.data_rec = self.data_signal_rec

        if self.params["mc_rec_cut"]:
            # cut_position = self.params["mc_rec_cut_position"]
            # self.mc_rec_mask = ~((self.mc_rec > cut_position[0]).squeeze() * (self.mc_rec < cut_position[1]).squeeze())
            self.mc_rec_mask = self.apply_efficiency_acceptance_effects(self.mc_rec, self.params["acceptance"])
            self.mc_rec[~self.mc_rec_mask.bool()] = self.params["empty_value"]*torch.ones_like(self.mc_rec[~self.mc_rec_mask.bool()])
            # self.mc_rec = self.mc_rec[self.mc_rec_mask]
            # self.mc_gen = self.mc_gen[self.mc_rec_mask]

        if self.params["mc_gen_cut"]:
            # cut_position = self.params["mc_gen_cut_position"]
            # self.mc_gen_mask = ~((self.mc_gen > cut_position[0]).squeeze() * (self.mc_gen < cut_position[1]).squeeze())
            self.mc_gen_mask = self.apply_efficiency_acceptance_effects(self.mc_gen, self.params["efficiency"])
            self.mc_gen[~self.mc_gen_mask.bool()] = self.params["empty_value"]*torch.ones_like(self.mc_gen[~self.mc_gen_mask.bool()])
            # self.mc_rec = self.mc_rec[self.mc_gen_mask]
            # self.mc_gen = self.mc_gen[self.mc_gen_mask]

        if self.params["data_rec_cut"]:
            # cut_position = self.params["data_rec_cut_position"]
            # self.data_rec_mask = ~((self.data_rec > cut_position[0]).squeeze() * (self.data_rec < cut_position[1]).squeeze())
            self.data_rec_mask = self.apply_efficiency_acceptance_effects(self.data_rec, self.params["acceptance"])
            self.data_rec[~self.data_rec_mask.bool()] = self.params["empty_value"]*torch.ones_like(self.data_rec[~self.data_rec_mask.bool()])
            # self.data_rec = self.data_rec[self.data_rec_mask]
            # self.data_gen = self.data_gen[self.data_rec_mask]

        if self.params["data_gen_cut"]:
            # cut_position = self.params["data_gen_cut_position"]
            # self.data_gen_mask = ~((self.data_gen > cut_position[0]).squeeze() * (self.data_gen < cut_position[1]).squeeze())
            self.data_gen_mask = self.apply_efficiency_acceptance_effects(self.data_gen, self.params["efficiency"])
            self.data_gen[~self.data_gen_mask.bool()] = self.params["empty_value"]*torch.ones_like(self.data_gen[~self.data_gen_mask.bool()])
            # self.data_rec = self.data_rec[self.data_gen_mask]
            # self.data_gen = self.data_gen[self.data_gen_mask]