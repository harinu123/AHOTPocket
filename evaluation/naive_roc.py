from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import auc, roc_curve
import os
import argparse

import sys

sys.path.insert(0, "..")
from utils import extract_resids

tqdm.pandas()


def get_pred_res_counts(struc_id, dfs):
    pred_res_dict = dict()
    for dfname in dfs.keys():
        df = dfs[dfname]
        pockets = df[df["structure id"] == struc_id]
        if not "pocket res" in pockets.columns:
            continue
        pockets = pockets.dropna(subset="pocket res")
        pockets = pockets["pocket res"].tolist()
        for p in pockets:
            for res in p.split(" "):
                if res in pred_res_dict.keys() and not dfname in pred_res_dict[res]:
                    pred_res_dict[res].append(dfname)
                else:
                    pred_res_dict[res] = [dfname]
    pred_res_counts = dict()
    for k in pred_res_dict.keys():
        pred_res_counts[k] = len(pred_res_dict[k])
    return pred_res_counts


def get_union(struc_id, dfs):
    return get_intersection(struc_id, dfs, thresh=1)


def get_intersection(struc_id, dfs, thresh=5):
    pred_res_counts = get_pred_res_counts(struc_id, dfs)
    intersection = [
        res for res in pred_res_counts.keys() if pred_res_counts[res] >= thresh
    ]
    return intersection


def get_tp(row, pred_colname, true_colname):
    assert type(row[pred_colname]) == list and type(row[true_colname]) == list
    pred_pos = set(
        [elem for elem in row[pred_colname] if elem.split("-")[0] == row["chain"]]
    )
    tp = len(pred_pos.intersection(set(row[true_colname])))
    return tp


def get_fp(row, pred_colname, false_colname):
    assert type(row[pred_colname]) == list and type(row[false_colname]) == list
    pred_pos = set(
        [elem for elem in row[pred_colname] if elem.split("-")[0] == row["chain"]]
    )
    fp = len(pred_pos.intersection(set(row[false_colname])))
    return fp


def get_overall_fpr(df, prefix):
    fp = sum(df["%s fp" % prefix])
    neg = sum([len(l) for l in df["nonbinding res"]])
    return fp / neg


def get_overall_tpr(df, prefix):
    tp = sum(df["%s tp" % prefix])
    pos = sum([len(l) for l in df["binding res"]])
    return tp / pos


def make_roc_plot(
    df,
    autosite_y_true,
    autosite_y_pred,
    castp_y_true,
    castp_y_pred,
    cavity_y_true,
    cavity_y_pred,
    fpocket_y_true,
    fpocket_y_pred,
    ligsite_y_true,
    ligsite_y_pred,
    prank_y_true,
    prank_y_pred,
    pocketminer_y_true,
    pocketminer_y_pred,
):
    unfiltered_fpr = [1]
    unfiltered_tpr = [1]
    unfiltered_fpr += [
        get_overall_fpr(df, "unfiltered intersection k=%d" % i) for i in range(1, 8)
    ]
    unfiltered_tpr += [
        get_overall_tpr(df, "unfiltered intersection k=%d" % i) for i in range(1, 8)
    ]
    unfiltered_fpr += [0]
    unfiltered_tpr += [0]

    filtered_fpr = [1]
    filtered_tpr = [1]
    filtered_fpr += [
        get_overall_fpr(df, "filtered intersection k=%d" % i) for i in range(1, 8)
    ]
    filtered_tpr += [
        get_overall_tpr(df, "filtered intersection k=%d" % i) for i in range(1, 8)
    ]
    filtered_fpr += [0]
    filtered_tpr += [0]

    fpocket_fpr, fpocket_tpr, _ = roc_curve(fpocket_y_true, fpocket_y_pred)
    prank_fpr, prank_tpr, _ = roc_curve(prank_y_true, prank_y_pred)
    autosite_fpr, autosite_tpr, _ = roc_curve(autosite_y_true, autosite_y_pred)
    ligsite_fpr, ligsite_tpr, _ = roc_curve(ligsite_y_true, ligsite_y_pred)
    pocketminer_fpr, pocketminer_tpr, _ = roc_curve(
        pocketminer_y_true, pocketminer_y_pred
    )
    cavity_fpr, cavity_tpr, _ = roc_curve(cavity_y_true, cavity_y_pred)
    castp_fpr, castp_tpr, _ = roc_curve(castp_y_true, castp_y_pred)

    unfiltered_auroc = auc(unfiltered_fpr, unfiltered_tpr)
    filtered_auroc = auc(filtered_fpr, filtered_tpr)

    fpocket_auroc = auc(fpocket_fpr, fpocket_tpr)
    prank_auroc = auc(prank_fpr, prank_tpr)
    autosite_auroc = auc(autosite_fpr, autosite_tpr)
    ligsite_auroc = auc(ligsite_fpr, ligsite_tpr)
    pocketminer_auroc = auc(pocketminer_fpr, pocketminer_tpr)
    cavity_auroc = auc(cavity_fpr, cavity_tpr)
    castp_auroc = auc(castp_fpr, castp_tpr)

    plt.figure()
    plt.plot(
        unfiltered_fpr,
        unfiltered_tpr,
        marker="x",
        label="Naive unfiltered, AUROC=%0.3f" % unfiltered_auroc,
    )
    plt.plot(
        filtered_fpr,
        filtered_tpr,
        marker="x",
        label="Naive filtered, AUROC=%0.3f" % filtered_auroc,
    )
    plt.plot(fpocket_fpr, fpocket_tpr, label="Fpocket, AUROC=%0.3f" % fpocket_auroc)
    plt.plot(prank_fpr, prank_tpr, label="P2Rank, AUROC=%0.3f" % prank_auroc)
    plt.plot(
        pocketminer_fpr,
        pocketminer_tpr,
        label="PocketMiner, AUROC=%0.3f" % pocketminer_auroc,
    )
    plt.plot(
        autosite_fpr[1],
        autosite_tpr[1],
        marker="x",
        label="AutoSite, AUROC=%0.3f" % autosite_auroc,
    )
    plt.plot(
        castp_fpr[1], castp_tpr[1], marker="x", label="CASTp, AUROC=%0.3f" % castp_auroc
    )
    plt.plot(
        cavity_fpr[1],
        cavity_tpr[1],
        marker="x",
        label="CAVITY, AUROC=%0.3f" % cavity_auroc,
    )
    plt.plot(
        ligsite_fpr[1],
        ligsite_tpr[1],
        marker="x",
        label="LIGSITEcs, AUROC=%0.3f" % ligsite_auroc,
    )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()


def get_binding_res(row, df):
    subdf = df[(df["pdb"] == row["pdb"]) & (df["chain"] == row["chain"])]
    binding_set = set()
    for pocket in subdf["binding res"]:
        for res in pocket:
            binding_set.add(res)
    return list(binding_set)


def get_nonbinding_res(row, struc_dir):
    resids, _ = extract_resids(os.path.join(struc_dir, "%s.pdb" % row["pdb"]))
    resids = [res for res in resids if res.split("-")[0] == row["chain"]]
    resids = [res for res in resids if not res in row["binding res"]]
    return resids


def get_top_k(df, k=5):
    outdf = None
    for pdb in tqdm(df["structure id"].unique()):
        struc_df = df[df["structure id"] == pdb]
        if len(struc_df) > k:
            struc_df = struc_df.sort_values("pocket id", ascending=True)
            struc_df = struc_df.head(k)
        if type(outdf) == type(None):
            outdf = struc_df
        else:
            outdf = pd.concat([outdf, struc_df])
    return outdf


def get_prank_probs(row, datadir=".."):
    fname = os.path.join(
        datadir, "data/prank/pdb_outs", "%s.pdb_predictions.csv" % row["structure id"]
    )
    if not os.path.isfile(fname):
        return None
    df = pd.read_csv(fname)
    df.columns = [col.strip() for col in df.columns]
    df["stripped name"] = df.apply(lambda row: row[df.columns[0]].strip(), axis=1)
    df = df[df["stripped name"] == row["pocket id"]]
    assert len(df) == 1
    return df["probability"].tolist()[0]


def get_residue_preds(df, struc_id, chain, resids, scorecol=None, binary_only=False):
    chain_resids = [r for r in resids if r.split("-")[0] == chain]
    preds = np.zeros((len(chain_resids),))
    subdf = df[df["structure id"] == struc_id]
    if not binary_only:
        scores = subdf[scorecol].tolist()
    for row_idx, pocket in enumerate(subdf["pocket res"]):
        pocket_list = pocket.split()
        for res in pocket_list:
            if not res in chain_resids:
                continue
            res_idx = chain_resids.index(res)
            if binary_only:
                preds[res_idx] = 1
            elif preds[res_idx] < scores[row_idx]:
                preds[res_idx] = scores[row_idx]
    return preds


def get_true_and_labels(
    df,
    struc_dir,
    biolip_df,
    scorecol=None,
    binary_only=False,
    pocketminer=False,
    memo={},
):
    print(len(memo))
    y_true = np.array([])
    y_pred = np.array([])
    for idx, struc_id in tqdm(enumerate(biolip_df["pdb"])):
        chain = biolip_df["chain"].tolist()[idx]
        if not struc_id in memo.keys():
            memo[struc_id] = get_residue_labels(struc_id, biolip_df, struc_dir)
        resids = memo[struc_id]["resids"]
        labels = memo[struc_id]["labels"][chain]
        if pocketminer:
            preds = get_pocketminer_residue_preds(struc_id, chain, struc_dir, resids)
            if type(preds) == type(None):
                continue
        else:
            preds = get_residue_preds(
                df, struc_id, chain, resids, scorecol=scorecol, binary_only=binary_only
            )
        y_true = np.concatenate([y_true, labels])
        y_pred = np.concatenate([y_pred, preds])
    return y_true, y_pred, memo


def get_pocketminer_residue_preds(struc_id, chain, struc_dir, resids, datadir=".."):
    scorefile = os.path.join(
        datadir, "data/pocketminer/%s_out/%s-preds.npy" % (struc_id, struc_id)
    )
    if not os.path.isfile(scorefile):
        return None
    scores = np.load(scorefile)
    if len(resids) != scores.shape[1]:
        return None
    all_pocketminer_scores = scores[0]
    chain_pocketminer_scores = []
    for idx, resid in enumerate(resids):
        curr_chain = resid.split("-")[0]
        if curr_chain == chain:
            chain_pocketminer_scores.append(all_pocketminer_scores[idx])
    return chain_pocketminer_scores


def get_residue_labels(struc_id, biolip_df, struc_dir):
    dic = dict()
    resids, _ = extract_resids(os.path.join(struc_dir, "%s.pdb" % struc_id))
    dic["resids"] = resids
    dic["labels"] = dict()
    biolip_df = biolip_df[biolip_df["pdb"] == struc_id]
    labels = np.zeros((len(resids),))
    for pocket in biolip_df["binding res"]:
        for res in pocket:
            if not res in resids:
                continue
            res_idx = resids.index(res)
            labels[res_idx] = 1
    for idx, resid in enumerate(resids):
        curr_chain = resid.split("-")[0]
        if not curr_chain in dic["labels"].keys():
            dic["labels"][curr_chain] = []
        dic["labels"][curr_chain].append(labels[idx])
    return dic


def main(args):
    autosite_df = pd.read_csv("../data/pred_autosite_pockets.csv")
    castp_df = pd.read_csv("../data/pred_castp_pockets.csv")
    cavity_df = pd.read_csv("../data/pred_cavityspace_pockets.csv")
    fpocket_df = pd.read_csv("../data/pred_fpocket_pockets_scored.csv")
    ligsite_df = pd.read_csv("../data/pred_ligsite_pockets.csv")
    pocketminer_df = pd.read_csv("../data/pred_pocketminer_pockets_scored.csv")
    prank_df = pd.read_csv("../data/pred_prank_pockets_scored.csv")

    dfs = {
        "autosite": autosite_df,
        "castp": castp_df,
        "cavity": cavity_df,
        "fpocket": fpocket_df,
        "ligsite": ligsite_df,
        "prank": prank_df,
        "pocketminer": pocketminer_df,
    }

    prank_df["score"] = prank_df.progress_apply(get_prank_probs, axis=1)

    biolip_ttd = pd.read_csv("../data/biolip_TTD.csv")
    biolip_ttd = biolip_ttd[
        ["uniprot", "pdb", "chain", "binding site residues, pdb numbering"]
    ]

    autosite_df = autosite_df[
        autosite_df["structure id"].isin(biolip_ttd["pdb"].tolist())
    ]
    castp_df = castp_df[castp_df["structure id"].isin(biolip_ttd["pdb"].tolist())]
    cavity_df = cavity_df[cavity_df["structure id"].isin(biolip_ttd["pdb"].tolist())]
    fpocket_df = fpocket_df[fpocket_df["structure id"].isin(biolip_ttd["pdb"].tolist())]
    ligsite_df = ligsite_df[ligsite_df["structure id"].isin(biolip_ttd["pdb"].tolist())]
    pocketminer_df = pocketminer_df[
        pocketminer_df["structure id"].isin(biolip_ttd["pdb"].tolist())
    ]
    prank_df = prank_df[prank_df["structure id"].isin(biolip_ttd["pdb"].tolist())]

    filt_autosite_df = get_top_k(autosite_df)
    filt_castp_df = get_top_k(castp_df)
    filt_ligsite_df = get_top_k(ligsite_df)

    filt_fpocket_df = fpocket_df[fpocket_df["score"] >= 0.5]
    filt_pocketminer_df = pocketminer_df[pocketminer_df["score"] >= 0.8]
    filt_prank_df = prank_df[prank_df["score"] >= 0.5]

    filt_cavity_df = pd.read_csv("../data/pred_cavityspace_pockets_strong.csv")

    filt_dfs = {
        "autosite": filt_autosite_df,
        "castp": filt_castp_df,
        "cavity": filt_cavity_df,
        "fpocket": filt_fpocket_df,
        "ligsite": filt_ligsite_df,
        "prank": filt_prank_df,
        "pocketminer": filt_pocketminer_df,
    }

    biolip_ttd["binding res"] = biolip_ttd.apply(
        lambda row: [
            "%s-%s" % (row["chain"], resid)
            for resid in row["binding site residues, pdb numbering"].split()
        ],
        axis=1,
    )

    biolip_ttd["binding res"] = biolip_ttd.progress_apply(
        get_binding_res, args=(biolip_ttd,), axis=1
    )
    biolip_ttd = biolip_ttd.drop_duplicates(subset=["pdb", "chain"])
    biolip_ttd = biolip_ttd[
        ~biolip_ttd["pdb"].isin(["6bcu", "6dr0", "6msg", "6msj", "6wev", "8d4c"])
    ]  # getting rid of problematic structure
    biolip_ttd["nonbinding res"] = biolip_ttd.progress_apply(
        get_nonbinding_res, args=(args.struc_dir,), axis=1
    )

    for i in range(1, 8):
        biolip_ttd["unfiltered intersection k=%d preds" % i] = (
            biolip_ttd.progress_apply(
                lambda row: get_intersection(row["pdb"], dfs, thresh=i), axis=1
            )
        )
        biolip_ttd["unfiltered intersection k=%d fp" % i] = biolip_ttd.apply(
            get_fp,
            args=("unfiltered intersection k=%d preds" % i, "nonbinding res"),
            axis=1,
        )
        biolip_ttd["unfiltered intersection k=%d tp" % i] = biolip_ttd.apply(
            get_tp,
            args=("unfiltered intersection k=%d preds" % i, "binding res"),
            axis=1,
        )

        biolip_ttd["filtered intersection k=%d preds" % i] = biolip_ttd.progress_apply(
            lambda row: get_intersection(row["pdb"], filt_dfs, thresh=i), axis=1
        )
        biolip_ttd["filtered intersection k=%d fp" % i] = biolip_ttd.apply(
            get_fp,
            args=("filtered intersection k=%d preds" % i, "nonbinding res"),
            axis=1,
        )
        biolip_ttd["filtered intersection k=%d tp" % i] = biolip_ttd.apply(
            get_tp, args=("filtered intersection k=%d preds" % i, "binding res"), axis=1
        )

    autosite_y_true, autosite_y_pred, memo = get_true_and_labels(
        autosite_df, args.struc_dir, biolip_ttd, binary_only=True, memo={}
    )
    cavity_y_true, cavity_y_pred, memo = get_true_and_labels(
        cavity_df, args.struc_dir, biolip_ttd, binary_only=True, memo=memo
    )
    castp_y_true, castp_y_pred, memo = get_true_and_labels(
        castp_df, args.struc_dir, biolip_ttd, binary_only=True, memo=memo
    )
    ligsite_y_true, ligsite_y_pred, memo = get_true_and_labels(
        ligsite_df, args.struc_dir, biolip_ttd, binary_only=True, memo=memo
    )
    fpocket_y_true, fpocket_y_pred, memo = get_true_and_labels(
        fpocket_df, args.struc_dir, biolip_ttd, scorecol="score", memo=memo
    )
    prank_y_true, prank_y_pred, memo = get_true_and_labels(
        prank_df, args.struc_dir, biolip_ttd, scorecol="score", memo=memo
    )
    pocketminer_y_true, pocketminer_y_pred, memo = get_true_and_labels(
        pocketminer_df, args.struc_dir, biolip_ttd, pocketminer=True, memo=memo
    )

    make_roc_plot(
        biolip_ttd,
        autosite_y_true,
        autosite_y_pred,
        castp_y_true,
        castp_y_pred,
        cavity_y_true,
        cavity_y_pred,
        fpocket_y_true,
        fpocket_y_pred,
        ligsite_y_true,
        ligsite_y_pred,
        prank_y_true,
        prank_y_pred,
        pocketminer_y_true,
        pocketminer_y_pred,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--struc_dir", help="path to directory with pdb structures")
    args = parser.parse_args()
    main(args)
