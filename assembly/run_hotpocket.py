import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import torch
from nn_sweep import get_model
import argparse
import warnings
from Bio.PDB import PDBParser

import sys

sys.path.insert(0, "..")
from utils import extract_resids, AA_3TO1


def retrieve_pockets(dfs, struc_id):
    all_pockets = []
    all_pockets_df = None
    for dfname in dfs.keys():
        df = dfs[dfname]
        pockets = df[df["structure id"] == struc_id]
        if not "pocket res" in pockets.columns:
            continue
        pockets_df = pockets.dropna(subset="pocket res")
        pockets_df["contributing method name"] = dfname
        pockets = pockets_df["pocket res"].tolist()
        for p in pockets:
            all_pockets.append(p.split(" "))
        if type(all_pockets_df) == type(None):
            all_pockets_df = pockets_df
        else:
            all_pockets_df = pd.concat([all_pockets_df, pockets_df])
    return all_pockets, all_pockets_df


def featurize_esm(struc_file, struc_id, pockets, embs_dir, rep_len, emb_size=2560):
    emb_rep = np.zeros((len(pockets), rep_len, emb_size))

    chains = set()
    for sphere in pockets:
        for elem in sphere:
            chains.add(elem.split("-")[0])

    emb = torch.zeros((1))
    emb_files = os.listdir(embs_dir)
    pdb_match = [elem for elem in emb_files if elem.split("_")[0] == struc_id]

    chain_seq_dict = dict()

    for chain in chains:
        for pt_file in pdb_match:
            pt_chain = pt_file.split("_")[1][:-3]  # format: XXXX_{chain info}.pt
            if chain in pt_chain.split(","):
                emb_dict = load_pt(os.path.join(embs_dir, pt_file))
                emb = emb_dict["representations"][36]
        res_map, _ = extract_resids(struc_file)
        res_map_chain = [elem for elem in res_map if elem[0] == chain]
        assert len(emb) == len(res_map_chain)
        chain_seq_dict[chain] = {"emb": emb, "res map chain": res_map_chain}

    for sphere_idx, sphere in enumerate(pockets):
        j = 0
        for i, res in enumerate(sphere):
            c = res.split("-")[0]
            if i - j >= rep_len:
                break
            if not res in chain_seq_dict[c]["res map chain"]:
                j += 1
                continue
            res_idx = chain_seq_dict[c]["res map chain"].index(res)
            emb_rep[sphere_idx, max(0, i - j)] = chain_seq_dict[c]["emb"][res_idx]

    return emb_rep, chain_seq_dict


def featurize_method_preds(pdb, pockets, rep_len, dfs, chain_seq_dict):
    emb_rep = np.zeros((len(pockets), rep_len, len(dfs.keys())))
    res_sets = dict()
    keys = []
    for df_idx, dfname in enumerate(dfs.keys()):
        df = dfs[dfname]
        if pdb in df["structure id"].tolist():
            subdf = df.loc[df["structure id"] == pdb]
            subdf = subdf.dropna(subset="pocket res")

            res_set = set(res for row in subdf["pocket res"] for res in row.split())
            res_sets[df_idx] = list(res_set)
            keys.append(df_idx)

    for sphere_idx, sphere in enumerate(pockets):
        j = 0
        for df_idx in keys:
            for res in res_sets[df_idx]:
                chain = res.split("-")[0]
                if not res in sphere:
                    continue
                idx = sphere.index(res)
                if not res in chain_seq_dict[chain]["res map chain"]:
                    j += 1
                    continue
                if idx - j >= rep_len:
                    continue
                emb_rep[sphere_idx, max(idx - j, 0), df_idx] = 1

    return emb_rep


def get_dfs(dfsdir="data", prefix=""):
    autosite_df = pd.read_csv(
        os.path.join(dfsdir, "%spred_autosite_pockets.csv" % prefix)
    )
    castp_df = pd.read_csv(os.path.join(dfsdir, "%spred_castp_pockets.csv" % prefix))
    cavity_df = pd.read_csv(
        os.path.join(dfsdir, "%spred_cavityspace_pockets.csv" % prefix)
    )
    fpocket_df = pd.read_csv(
        os.path.join(dfsdir, "%spred_fpocket_pockets_scored.csv" % prefix)
    )
    ligsite_df = pd.read_csv(
        os.path.join(dfsdir, "%spred_ligsite_pockets.csv" % prefix)
    )
    pocketminer_df = pd.read_csv(
        os.path.join(dfsdir, "%spred_pocketminer_pockets.csv" % prefix)
    )
    prank_df = pd.read_csv(os.path.join(dfsdir, "%spred_prank_pockets.csv" % prefix))

    dfs = {
        "autosite": autosite_df,
        "castp": castp_df,
        "cavity": cavity_df,
        "fpocket": fpocket_df,
        "ligsite": ligsite_df,
        "prank": prank_df,
        "pocketminer": pocketminer_df,
    }
    return dfs


def get_models():
    nn_embs_state_dict = torch.load("nn_augmented_esm_only_daily-sweep-128_733.pt")
    nn_comb_state_dict = torch.load("nn_augmented_combined_prime-sweep-241_733.pt")
    nn_embs_config_dict = {"features": "esm_only", "dropout": 0.5, "n_hidden": 1}
    nn_comb_config_dict = {"features": "combined", "dropout": 0.4, "n_hidden": 1}
    nn_embs_model = get_model(nn_embs_config_dict)
    nn_comb_model = get_model(nn_comb_config_dict)
    nn_embs_model.load_state_dict(nn_embs_state_dict)
    nn_comb_model.load_state_dict(nn_comb_state_dict)
    nn_embs_model.eval()
    nn_comb_model.eval()
    return nn_embs_model, nn_comb_model


def load_pt(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return torch.load(path)


def get_confidence(row, struc):
    pocket_res = row["pocket res"].split()

    atom_confidence = []
    model = struc[0]
    for chain in model:
        for residue in chain:
            resname3 = residue.get_resname()
            resnum = residue.get_id()[1]
            if not resname3 in AA_3TO1.keys():
                continue
            resname1 = AA_3TO1[resname3]
            resid = "%s-%s%d" % (chain.get_id(), resname1, resnum)
            if resid in pocket_res:
                atom_confidence += [atom.get_bfactor() for atom in residue.get_atoms()]
    return sum(atom_confidence) / len(atom_confidence)


def confidence_over_df(dfname):
    parser = PDBParser(QUIET=True)
    df = pd.read_csv(dfname)

    struc_dirs = [
        "/scratch/users/kcarp/proteome_wide_analysis/strucs",
        "/scratch/users/kcarp/cote_crohns/strucs",
        "/scratch/users/kcarp/all_biolip_pdb",
    ]

    af_df = df[df["structure type"] == "AF2"]
    new_df = df[df["structure type"] == "PDB"]
    new_df["avg confidence"] = 100

    print(len(af_df["structure id"].unique()))
    for strucname in tqdm(af_df["structure id"].unique()):
        struc = None
        for struc_dir in struc_dirs:
            fname = os.path.join(struc_dir, "%s.pdb" % strucname)
            if os.path.isfile(fname):
                struc = parser.get_structure("prot", fname)
                break
        if type(struc) == type(None):
            print("no structure found: %s" % strucname)
        subdf = af_df[af_df["structure id"] == strucname].copy()
        subdf["avg confidence"] = subdf.apply(get_confidence, args=(struc,), axis=1)
        new_df = pd.concat([new_df, subdf])
    return new_df


def main(args):
    outdir = args.outdir

    dfs = get_dfs(args.dfsdir, args.dfs_prefix)
    nn_embs_model, nn_comb_model = get_models()

    errf = open(args.errfile, "w")
    errf.close()

    if args.in_strucid:
        if not os.path.isfile(args.in_strucid):
            print("ERROR: File not found: %s" % args.in_strucid)
            return
        if args.in_strucid[-4:] == ".txt":
            struclist = open(args.in_strucid, "r").read().split("\n")
            struclist = [struc for struc in struclist if len(struc) > 0]
            struclist = [
                struc[:-4]
                for struc in struclist
                if len(struc) >= 8 and struc[-4:] == ".pdb"
            ]
        elif args.in_strucid[-4:] == ".csv":
            input_df = pd.read_csv(args.in_strucid)
            if "pdb" in input_df.columns:
                struclist = input_df["pdb"].unique().tolist()
            elif "structure id" in input_df.columns:
                struclist = input_df["structure id"].unique().tolist()
        else:
            print("ERROR: Unsupported file type: %s" % args.in_strucid)
            return
    else:
        print("using %s as input..." % args.pdbdir)
        all_strucfiles = os.listdir(args.pdbdir)
        struclist = [elem[:-4] for elem in all_strucfiles if elem[-4:] == ".pdb"]

    for struc_id in tqdm(struclist):
        if os.path.isfile(os.path.join(outdir, "%s.csv" % struc_id)):
            continue
        strucfile = os.path.join(args.pdbdir, "%s.pdb" % struc_id)
        if not os.path.isfile(strucfile):
            continue

        pockets, pockets_df = retrieve_pockets(dfs, struc_id)

        if len(pockets) == 0:
            pd.DataFrame().to_csv(os.path.join(outdir, "%s.csv" % struc_id))
            continue

        embs_dir = args.embdir
        max_len = 15

        try:
            X_esm_embs, pdb_res = featurize_esm(
                strucfile, struc_id, pockets, embs_dir, max_len
            )
            X_method_preds = featurize_method_preds(
                struc_id, pockets, max_len, dfs, pdb_res
            )
            X_combined = np.concatenate((X_method_preds, X_esm_embs), axis=2)
        except AssertionError:
            errorf = open(args.errfile, "a")
            errorf.write("%s\n" % struc_id)
            errorf.close()
            continue

        reshaped_X_esm_embs = torch.tensor(
            X_esm_embs.reshape((X_esm_embs.shape[0], -1))
        ).float()
        reshaped_X_combined = torch.tensor(
            X_combined.reshape((X_combined.shape[0], -1))
        ).float()

        esm_embs_Y_pred = nn_embs_model(reshaped_X_esm_embs)
        combined_Y_pred = nn_comb_model(reshaped_X_combined)

        pockets_df["embs NN score"] = esm_embs_Y_pred.detach().numpy()
        pockets_df["comb NN score"] = combined_Y_pred.detach().numpy()

        pockets_df.to_csv(os.path.join(outdir, "%s.csv" % struc_id))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_strucid",
        type=str,
        help="name of text file with structure ids for which to run pipeline",
    )
    parser.add_argument(
        "--embdir", type=str, help="name of directory containing ESM embeddings"
    )
    parser.add_argument(
        "--pdbdir",
        type=str,
        help="name of directory containing protein structure files",
    )
    parser.add_argument(
        "--outdir", type=str, help="name of output directory in which to dump files"
    )
    parser.add_argument(
        "--errfile", type=str, help="path to file in which to write errors"
    )
    parser.add_argument(
        "--dfsdir",
        type=str,
        help="directory containing prediction csvs for each method",
        default="data/",
    )
    parser.add_argument(
        "--dfs_prefix", type=str, help="prefix for output method pred csvs", default=""
    )
    args = parser.parse_args()
    main(args)
