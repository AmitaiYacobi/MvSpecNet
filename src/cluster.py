import sys
import json
import torch
import random
import numpy as np
import wandb

from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

from _utils import *
from _data import load_data
from _metrics import Metrics
from _model import SpectralJoint



def set_seed(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    seeds = range(100, 200)
    dataset = sys.argv[1]

    config_path = f"config/{dataset}.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    dataset = config["dataset"]
    n_clusters = config["n_clusters"]

    Views_train, Views_test, labels_train, labels_test = load_data(dataset)



    # for seed in seeds:
    # set_seed(69)
    # run = wandb.init(
    #     # project="MultiSpectralNet-BDGP-CheckSeed",
    #     project="MultiSpectralNet-Caltech-CheckSeed",
    #     entity="amitaiyacobi",
    # )

    acc = []
    nmi = []
    ari = []
    

    for i in range(5):
        spectraljoint = SpectralJoint(n_clusters=n_clusters, config=config)
        spectraljoint.fit(Views_train, labels_train)

        embeddings = spectraljoint.predict(Views_test)
        cluster_assignments = spectraljoint.cluster(embeddings)
        labels = labels_test
        

        if labels is not None:
            acc_score = Metrics.acc_score(
                cluster_assignments,
                labels.detach().cpu().numpy(),
                n_clusters,
            )
            nmi_score = Metrics.nmi_score(
                cluster_assignments,
                labels.detach().cpu().numpy(),
            )
            ari_score = Metrics.ari_score(
                cluster_assignments,
                labels.detach().cpu().numpy(),
            )
            embeddings = spectraljoint.embeddings_
            print(f"ACC: {np.round(acc_score, 3)}")
            print(f"NMI: {np.round(nmi_score, 3)}")
            print(f"ARI: {np.round(ari_score, 3)}")

            acc.append(acc_score)
            nmi.append(nmi_score)
            ari.append(ari_score)

    print(f"Average acc: {np.mean(acc)}, std: {np.std(acc)}")
    print(f"Average nmi: {np.mean(nmi)}, std: {np.std(nmi)}")
    print(f"Average ari: {np.mean(ari)}, std: {np.std(ari)}")





        # wandb.log(
        # {
        #     "siamese_nbg": config["siamese"]["n_neighbors"],
        #     "acc": acc_score,
        #     "nmi": nmi_score,
        #     "seed": seed
        # })
        # run.finish()
        # for filename in os.listdir("weights"):
        #     os.remove(
        #         os.path.join(
        #             "weights", filename
        #         )
        #     )


    # ae_output1 = [50, 100, 200, 300, 400, 500, 600]
    # ae_output2 = [10, 20, 30, 40]
    # siamese_output1 = [100, 200, 300, 400, 500, 600]
    # siamese_output2 = [10, 20, 30, 40]
    # epochs = [35, 40, 50]
    # siamese_nbgs = [2, 3, 4, 5, 6]
    # spectral_nbgs = list(range(5, 40))
    # scales = list(range(5, 40))
    # is_local = [True, False]

    # for ae1 in ae_output1:
    #     for ae2 in ae_output2:
    #         for siamese1 in siamese_output1:
    #             for siamese2 in siamese_output2:
    #                 for epoch in epochs:
    #                     for ngb_siamese in siamese_nbgs:
    #                         for is_loc in is_local:
    #                             for nbg_spectral in spectral_nbgs:
    #                                 if is_loc == True:
    #                                     continue
    #                                 for scale_k in scales:
    #                                     if scale_k > nbg_spectral:
    #                                         continue
    #                                     config["ae"]["architectures"][0][
    #                                         "output_dim"
    #                                     ] = ae1
    #                                     config["ae"]["architectures"][1][
    #                                         "output_dim"
    #                                     ] = ae2
    #                                     config["siamese"]["architectures"][0][
    #                                         "output_dim"
    #                                     ] = siamese1
    #                                     config["siamese"]["architectures"][1][
    #                                         "output_dim"
    #                                     ] = siamese2
    #                                     config["siamese"]["n_neighbors"] = ngb_siamese
    #                                     config["spectral"]["epochs"] = epoch
    #                                     config["spectral"]["is_local_scale"] = is_loc
    #                                     config["spectral"]["n_neighbors"] = nbg_spectral
    #                                     config["spectral"]["scale_k"] = scale_k

    #                                     run = wandb.init(
    #                                         project="MultiSpectralNet-BDGP-Local&Global",
    #                                         entity="amitaiyacobi",
    #                                     )

    #                                     lr = config["spectral"]["lr"]
    #                                     try:
    #                                         multi_spectralnet = MultiSpectralNet(
    #                                             n_clusters=n_clusters, config=config
    #                                         )
    #                                         multi_spectralnet.fit(Views, labels)

    #                                         cluster_assignments = (
    #                                             multi_spectralnet.predict(Views)
    #                                         )

    #                                         if labels is not None:
    #                                             # labels = labels.detach().cpu().numpy()
    #                                             acc_score = Metrics.acc_score(
    #                                                 cluster_assignments,
    #                                                 labels.detach().cpu().numpy(),
    #                                                 n_clusters,
    #                                             )
    #                                             nmi_score = Metrics.nmi_score(
    #                                                 cluster_assignments,
    #                                                 labels.detach().cpu().numpy(),
    #                                             )
    #                                             embeddings = (
    #                                                 multi_spectralnet.embeddings_
    #                                             )
    #                                             print(f"ACC: {np.round(acc_score, 3)}")
    #                                             print(f"NMI: {np.round(nmi_score, 3)}")
    #                                             wandb.log(
    #                                                 {
    #                                                     "lr": lr,
    #                                                     "epochs": epoch,
    #                                                     "ae1_output": ae1,
    #                                                     "ae2_output": ae2,
    #                                                     "siamese1_output": siamese1,
    #                                                     "siamese2_output": siamese2,
    #                                                     "siamese_nbg": ngb_siamese,
    #                                                     "spectral_nbg": nbg_spectral,
    #                                                     "scale_k": scale_k,
    #                                                     "is_local": is_loc,
    #                                                     "acc": acc_score,
    #                                                     "nmi": nmi_score,
    #                                                 }
    #                                             )
    #                                             run.finish()
    #                                             for filename in os.listdir("weights"):
    #                                                 os.remove(
    #                                                     os.path.join(
    #                                                         "weights", filename
    #                                                     )
    #                                                 )
    #                                     except:
    #                                         continue
    return embeddings, cluster_assignments


if __name__ == "__main__":
   main()
 
