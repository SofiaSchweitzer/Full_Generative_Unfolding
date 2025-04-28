import numpy as np
import energyflow as ef


def get_substructure_obs(dataset):
    feature_names = ['mults','widths','sdms','tau2s','zgs']


    #Jet Mass
    gen_features = [dataset['gen_jets'][:,3]]
    sim_features = [dataset['sim_jets'][:,3]]
    for feature in feature_names:
        gen_features.append(dataset['gen_'+feature])
        sim_features.append(dataset['sim_'+feature])

    #PT
    gen_features.append(dataset['gen_jets'][:,0])
    sim_features.append(dataset['sim_jets'][:,0])
        
    gen_features = np.stack(gen_features,-1)
    sim_features = np.stack(sim_features,-1)
    #ln rho
    gen_features[:,3] = 2*np.ma.log(np.ma.divide(gen_features[:,3],dataset['gen_jets'][:,0]+10**-100).filled(0)).filled(0)
    sim_features[:,3] = 2*np.ma.log(np.ma.divide(sim_features[:,3],dataset['sim_jets'][:,0]+10**-100).filled(0)).filled(0)
    #tau21
    gen_features[:,4] = gen_features[:,4]/(10**-50 + gen_features[:,2])
    sim_features[:,4] = sim_features[:,4]/(10**-50 + sim_features[:,2])

    mask_reco = (sim_features[:, 0] < 150) & (sim_features[:, 3] > -20)
    print(np.sum(mask_reco)/mask_reco.shape[0])
    mask_gen = (gen_features[:, 0] < 150) & (gen_features[:, 3] > -20)
    print(np.sum(mask_gen)/mask_gen.shape[0])
    mask = (mask_reco) & (mask_gen)
    gen_features = gen_features[mask]
    sim_features = sim_features[mask]
    
    return sim_features, gen_features


zjets_mc = ef.zjets_delphes.load('Pythia26', num_data=-1,
                                 cache_dir='/pscratch/sd/v/vmikuni/PET/QG/',
                                 pad=False,
                                 exclude_keys=['particles','gen_particles'])

full_reco,full_gen = get_substructure_obs(zjets_mc)
print(full_reco.shape)
np.save("reco_MC.npy", full_reco)
np.save("gen_MC.npy", full_gen)

zjets_data = ef.zjets_delphes.load('Herwig', num_data=-1,
                                   cache_dir='/pscratch/sd/v/vmikuni/PET/QG/',
                                   pad=False,
                                   exclude_keys=['particles','gen_particles'])


full_reco,full_gen = get_substructure_obs(zjets_data)
np.save("reco_data.npy", full_reco)
np.save("gen_data.npy", full_gen)
