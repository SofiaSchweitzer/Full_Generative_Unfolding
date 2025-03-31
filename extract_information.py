import numpy as np
full_gen = []
full_reco = []
for i in range(0,17):
    file = np.load(f"pythia/Pythia26_Zjet_pTZ-200GeV_{i}.npz")
    gen  = np.zeros((len(file["gen_jets"]), 7))
    reco = np.zeros((len(file["gen_jets"]), 7)) 
    #get jet momentum
    gen[:, 6] = file["gen_jets"][:, 0]
    reco[:, 6] = file["sim_jets"][:, 0]
    #get jet mass 
    gen[:, 0] = file["gen_jets"][:, 3]
    reco[:, 0] = file["sim_jets"][:, 3]
    #get multiplicity
    gen[:, 1] = file["gen_mults"]
    reco[:, 1] = file["sim_mults"]
    #get width
    gen[:, 2] = file["gen_widths"]
    reco[:, 2] = file["sim_widths"]
    #soft drop mass
    gen[:, 3] = file["gen_sdms"]
    reco[:, 3] = file["sim_sdms"]
    #N-subjetiness
    gen[:, 4] = file["gen_tau2s"]
    reco[:, 4] = file["sim_tau2s"]
    #groomed momentum
    gen[:, 5] = file["gen_zgs"]
    reco[:, 5] = file["sim_zgs"]
    gen[:, 4] = gen[:, 4]/(10**-50+gen[:, 2])
    reco[:, 4] = reco[:, 4]/(10**-50+reco[:, 2]) 

    gen[:, 3] = 2*np.ma.log(np.ma.divide(gen[:, 3],gen[:, 6]).filled(0)).filled(0)
    reco[:, 3] = 2*np.ma.log(np.ma.divide(reco[:, 3],reco[:, 6]).filled(0)).filled(0)
    full_gen.append(gen)
    full_reco.append(reco)
full_gen = np.concatenate(full_gen, axis=0)
full_reco = np.concatenate(full_reco, axis=0)

np.save("reco_MC.npy", full_reco)
np.save("gen_MC.npy", full_gen)

