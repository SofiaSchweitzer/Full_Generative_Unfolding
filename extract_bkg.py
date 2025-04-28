import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = pd.read_csv("zv_background.txt", sep=" ", header=None)

columns = ["egal_0", "level", "z_pt", "jet_pt", "jet_phi", "jet_eta","jet_m", "width", "tau", "egal_1", "egal_2","egal_3", "soft", "groomed", "mults"]
print(len(file))
file.columns = columns
file = file.query("level=='reco'")
print(len(file))
reco = np.zeros((len(file), 7))

reco[:, 0] = file["jet_m"].to_numpy()
reco[:, 1] = file["mults"].to_numpy()
reco[:, 2] = file["width"].to_numpy()
reco[:, 3] = file["soft"].to_numpy()
reco[:, 4] = file["tau"].to_numpy()
reco[:, 5] = file["groomed"].to_numpy()
reco[:, 6] = file["jet_pt"].to_numpy()
reco[:, 5][np.isnan(reco[:,5])] = 0
reco[: ,4] = reco[:, 4]/(10**-50+reco[:, 2])
reco[:, 3 ] = 2*np.ma.log(np.ma.divide(reco[:, 3], reco[:, 6]).filled(0)).filled(0)

mask = (reco[:, 0] < 150) & (reco[:, 3] > -20)
print(1.0*np.sum(mask)/reco.shape[0])
reco = reco[mask]

np.save("reco_bkg.npy", reco)

for i in range(reco.shape[-1]):
    plt.figure()
    y_true = plt.hist(reco[:,i],bins = 60,label = "True signal", histtype='step')    
    plt.legend()
    plt.xlabel(f"Feature {i}")
    plt.ylabel("Counts")
    plt.title(f"Signal Feature {i}")
    plt.savefig(f"Signal_feat{i}.pdf")
    plt.close()
    
