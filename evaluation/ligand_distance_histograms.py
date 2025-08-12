import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from tqdm import tqdm
import pickle
from scipy.stats import kstest

import sys

sys.path.insert(0, "..")
from utils import extract_resids, get_pairwise_distances, get_neighbors


def get_random_point(struc_fpath, structure=None):
    resids, structure = extract_resids(
        struc_fpath, with_coords=True, structure=structure
    )
    center_idx = random.randint(0, len(resids) - 1)
    k = random.randint(2, 14)
    pairwise_distances = get_pairwise_distances(structure, resids, with_coords=True)
    neighbor_idx = get_neighbors(center_idx, pairwise_distances, k)
    try:
        neighbor_coords = np.array([resids[i][1] for i in neighbor_idx])
    except:
        print([resids[i] for i in neighbor_idx])
        assert 1 == 0
    com = np.mean(neighbor_coords, axis=0)
    return com, structure


def get_dist_from_ligand(structure, coord, ligands):
    closest_dist = 99999999999999
    for model in structure:
        for chain in model:
            for res in chain:
                if res.get_resname().strip() in ligands:
                    for atom in res:
                        lig_coord = atom.get_coord()
                        dist = (
                            (coord[0] - lig_coord[0]) ** 2
                            + (coord[1] - lig_coord[1]) ** 2
                            + (coord[2] - lig_coord[2]) ** 2
                        ) ** 0.5
                        if dist < closest_dist:
                            closest_dist = dist
        break  # only using first nmr model, naively
    return closest_dist


def str_to_list(l):
    if type(l) == list:
        return l
    if l == "list()" or l == "[]":
        return []
    llist = l.strip("[]").replace("'", "").split(",")
    llist = [elem.strip() for elem in llist]
    return llist


def get_background_dist(struclist, df, background_dist={}):
    for strucid in tqdm(struclist):
        if strucid in background_dist.keys():
            continue
        subdf = df[df["structure id"] == strucid]
        n_lig = subdf["n ligands"].tolist()[0]
        ligands = str_to_list(subdf["ligands"].tolist()[0])
        dists = []
        structure = None
        for _ in range(n_lig):
            point, structure = get_random_point(
                "all_biolip_pdb/%s.pdb" % strucid, structure
            )
            dist = get_dist_from_ligand(structure, point, ligands)
            dists.append(dist)
        background_dist[strucid] = dists
    return background_dist


def ligand_dist_hist(df, background, color, title):
    df_pdbs = set(df["structure id"])
    bg_pdbs = set(background.keys())
    pdbs = df_pdbs.intersection(bg_pdbs)
    if len(df_pdbs) > len(pdbs):
        print("error: additional background pdb calculations needed")
        return
    pdbs = list(pdbs)
    pdblist = []
    distlist = []
    for k in pdbs:
        for d in background[k]:
            pdblist.append(k)
            distlist.append(d)
    background_df = pd.DataFrame({"pdb": pdblist, "dist": distlist})

    background_df = background_df[background_df["dist"] < 99999999999999]
    df = df[df["closest lig to pocket com"] < 99999999999999]

    ks_stat, p_val = kstest(df["closest lig to pocket com"], background_df["dist"])
    print("ks statistic: %0.3f" % ks_stat)
    print("p value: %0.3f" % p_val)

    plt.hist(background_df["dist"], bins=50, color="lightgray", alpha=0.7)
    plt.hist(df["closest lig to pocket com"], bins=50, color=color, alpha=0.7)
    plt.xlim([-5, 105])
    plt.ylim([0, 3800])
    plt.title("%s\np=%0.2e" % (title, p_val))


hotpocket_b_clust = pd.read_csv("largescale_eval_dfs/hotpocket_embs_clust_top_n.csv")
hotpocket_c_clust = pd.read_csv("largescale_eval_dfs/hotpocket_comb_clust_top_n.csv")

autosite = pd.read_csv("largescale_eval_dfs/autosite_top_n.csv")
castp = pd.read_csv("largescale_eval_dfs/castp_top_n.csv")
cavity = pd.read_csv("largescale_eval_dfs/cavity_top_n.csv")
fpocket = pd.read_csv("largescale_eval_dfs/fpocket_top_n.csv")
ligsite = pd.read_csv("largescale_eval_dfs/ligsite_top_n.csv")
pocketminer = pd.read_csv("largescale_eval_dfs/pocketminer_top_n.csv")
prank = pd.read_csv("largescale_eval_dfs/prank_top_n.csv")

background_dist = pickle.load(open("biolip_background_dist.p", "rb"))
