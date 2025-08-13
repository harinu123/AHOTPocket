import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import matplotlib

matplotlib.use("agg")
from matplotlib import pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


import sys

sys.path.insert(0, "..")
from utils import extract_resids

tqdm.pandas()


def get_residue_preds(
    df, struc_id, struc_dir, scorecol=None, structure=None, binary_only=False
):
    resids, structure = extract_resids(
        os.path.join(struc_dir, "%s.pdb" % struc_id), structure=structure
    )
    preds = np.zeros((len(resids),))
    subdf = df[df["structure id"] == struc_id]
    if not binary_only:
        scores = subdf[scorecol].tolist()
    for row_idx, pocket in enumerate(subdf["pocket res"]):
        pocket_list = pocket.split()
        for res in pocket_list:
            if not res in resids:
                continue
            res_idx = resids.index(res)
            if binary_only:
                preds[res_idx] = 1
            elif preds[res_idx] < scores[row_idx]:
                preds[res_idx] = scores[row_idx]
    return preds, structure


def get_residue_labels(df, struc_id, struc_dir, structure=None, thresh=5):
    resids, structure = extract_resids(
        os.path.join(struc_dir, "%s.pdb" % struc_id),
        structure=structure,
        with_coords=True,
    )
    labels = np.zeros((len(resids),))
    subdf = df[df["structure id"] == struc_id]
    ligands = subdf["ligands"].tolist()[0]
    ligands = str_to_list(ligands)
    for idx, res in enumerate(resids):
        if len(res) != 2:
            continue
        closest_dist = ligand_res_dist(ligands, structure, res[1])
        if closest_dist <= thresh:
            labels[idx] = 1
    return labels, structure


def ligand_res_dist(ligands, structure, res_coord):
    closest_dist = 99999999999999
    for model in structure:
        for chain in model:
            for res in chain:
                if res.get_resname().strip() in ligands:
                    for atom in res:
                        coord = atom.get_coord()
                        dist = (
                            (coord[0] - res_coord[0]) ** 2
                            + (coord[1] - res_coord[1]) ** 2
                            + (coord[2] - res_coord[2]) ** 2
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


def get_residue_preds_naive_filter(dfs, struc_id, struc_dir, structure=None):
    resids, structure = extract_resids(
        os.path.join(struc_dir, "%s.pdb" % struc_id), structure=structure
    )
    preds = np.zeros((len(resids),))

    pred_res_counts = get_pred_res_counts(struc_id, dfs)
    for k in pred_res_counts.keys():
        if not k in resids:
            continue
        res_idx = resids.index(k)
        preds[res_idx] = pred_res_counts[k] / len(dfs)

    return preds, structure


def get_true_and_labels(
    df,
    struc_dir,
    scorecol=None,
    binary_only=False,
    thresh=5,
    pocketminer=False,
    naive=False,
    dfs=None,
    memo={},
):
    print(len(memo))
    y_true = np.array([])
    y_pred = np.array([])
    for struc_id in tqdm(df["structure id"].unique()):
        if not struc_id in memo.keys():
            memo[struc_id] = get_residue_labels(df, struc_id, struc_dir, thresh=thresh)
        labels, structure = memo[struc_id]
        if pocketminer:
            preds, _ = get_pocketminer_residue_preds(
                struc_id, struc_dir, structure=structure
            )
            if type(preds) == type(None):
                continue
        elif naive:
            preds, _ = get_residue_preds_naive_filter(
                dfs, struc_id, struc_dir, structure=structure
            )
        else:
            preds, _ = get_residue_preds(
                df,
                struc_id,
                struc_dir,
                scorecol=scorecol,
                structure=structure,
                binary_only=binary_only,
            )
        y_true = np.concatenate([y_true, labels])
        y_pred = np.concatenate([y_pred, preds])
    return y_true, y_pred, memo


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


def get_pocketminer_residue_preds(struc_id, struc_dir, structure=None):
    resids, structure = extract_resids(
        os.path.join(struc_dir, "%s.pdb" % struc_id), structure=structure
    )
    scorefile = os.path.join(
        struc_dir, "../data/pocketminer/%s_out/%s-preds.npy" % (struc_id, struc_id)
    )
    scores = np.load(scorefile)
    if len(resids) != scores.shape[1]:
        return None, structure
    return scores[0], structure


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


def make_roc_plot(
    hotpocket_embs_vals,
    hotpocket_comb_vals,
    hotpocket_naive_vals,
    fpocket_vals,
    prank_vals,
    autosite_vals,
    cavity_vals,
    castp_vals,
    ligsite_vals,
    pocketminer_vals,
    title,
    out,
):
    hotpocket_embs_y_true, hotpocket_embs_y_pred = hotpocket_embs_vals
    hotpocket_comb_y_true, hotpocket_comb_y_pred = hotpocket_comb_vals
    hotpocket_naive_y_true, hotpocket_naive_y_pred = hotpocket_naive_vals
    fpocket_y_true, fpocket_y_pred = fpocket_vals
    prank_y_true, prank_y_pred = prank_vals
    autosite_y_true, autosite_y_pred = autosite_vals
    ligsite_y_true, ligsite_y_pred = ligsite_vals
    pocketminer_y_true, pocketminer_y_pred = pocketminer_vals

    hotpocket_embs_fpr, hotpocket_embs_tpr, _ = roc_curve(
        hotpocket_embs_y_true, hotpocket_embs_y_pred
    )
    hotpocket_comb_fpr, hotpocket_comb_tpr, _ = roc_curve(
        hotpocket_comb_y_true, hotpocket_comb_y_pred
    )
    hotpocket_naive_fpr, hotpocket_naive_tpr, _ = roc_curve(
        hotpocket_naive_y_true, hotpocket_naive_y_pred
    )
    fpocket_fpr, fpocket_tpr, _ = roc_curve(fpocket_y_true, fpocket_y_pred)
    prank_fpr, prank_tpr, _ = roc_curve(prank_y_true, prank_y_pred)
    autosite_fpr, autosite_tpr, _ = roc_curve(autosite_y_true, autosite_y_pred)
    ligsite_fpr, ligsite_tpr, _ = roc_curve(ligsite_y_true, ligsite_y_pred)
    pocketminer_fpr, pocketminer_tpr, _ = roc_curve(
        pocketminer_y_true, pocketminer_y_pred
    )

    hotpocket_embs_auroc = auc(hotpocket_embs_fpr, hotpocket_embs_tpr)
    hotpocket_comb_auroc = auc(hotpocket_comb_fpr, hotpocket_comb_tpr)
    hotpocket_naive_auroc = auc(hotpocket_naive_fpr, hotpocket_naive_tpr)
    fpocket_auroc = auc(fpocket_fpr, fpocket_tpr)
    prank_auroc = auc(prank_fpr, prank_tpr)
    autosite_auroc = auc(autosite_fpr, autosite_tpr)
    ligsite_auroc = auc(ligsite_fpr, ligsite_tpr)
    pocketminer_auroc = auc(pocketminer_fpr, pocketminer_tpr)

    if type(cavity_vals) != type(None) and len(cavity_vals[0]) > 0:
        cavity_y_true, cavity_y_pred = cavity_vals
        cavity_fpr, cavity_tpr, _ = roc_curve(cavity_y_true, cavity_y_pred)
        cavity_auroc = auc(cavity_fpr, cavity_tpr)
    if type(castp_vals) != type(None) and len(castp_vals[0]) > 0:
        castp_y_true, castp_y_pred = castp_vals
        castp_fpr, castp_tpr, _ = roc_curve(castp_y_true, castp_y_pred)
        castp_auroc = auc(castp_fpr, castp_tpr)

    plt.figure()
    plt.plot(
        hotpocket_embs_fpr,
        hotpocket_embs_tpr,
        label="hotpocket Feature Set B, AUROC=%0.3f" % hotpocket_embs_auroc,
    )
    plt.plot(
        hotpocket_comb_fpr,
        hotpocket_comb_tpr,
        label="hotpocket Feature Set C, AUROC=%0.3f" % hotpocket_comb_auroc,
    )
    plt.plot(
        hotpocket_naive_fpr,
        hotpocket_naive_tpr,
        label="Consensus w/ naive filter, AUROC=%0.3f" % hotpocket_naive_auroc,
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
        label="AutoSite, AUROC=%0.3f" % autosite_auroc,
        marker="x",
    )
    plt.plot(
        ligsite_fpr[1],
        ligsite_tpr[1],
        label="LIGSITEcs, AUROC=%0.3f" % ligsite_auroc,
        marker="x",
    )

    if type(cavity_vals) != type(None) and len(cavity_vals[0]) > 0:
        plt.plot(
            cavity_fpr[1],
            cavity_tpr[1],
            label="CAVITY, AUROC=%0.3f" % cavity_auroc,
            marker="x",
        )
    if type(castp_vals) != type(None) and len(castp_vals[0]) > 0:
        plt.plot(
            castp_fpr[1],
            castp_tpr[1],
            label="CASTp, AUROC=%0.3f" % castp_auroc,
            marker="x",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize="small")
    plt.grid()
    plt.savefig(out)


# fixing misleading lines from anchoring at (0,1) and (1,0)
def correct_prc(recall, precision):
    precision[-1] = precision[-2]
    precision = np.append(precision, [1])
    recall = np.append(recall, [0])
    return recall, precision


def make_prc_plot(
    hotpocket_embs_vals,
    hotpocket_comb_vals,
    hotpocket_naive_vals,
    fpocket_vals,
    prank_vals,
    autosite_vals,
    cavity_vals,
    castp_vals,
    ligsite_vals,
    pocketminer_vals,
    title,
    baseline,
    out,
    omit_anchor=False,
):
    hotpocket_embs_y_true, hotpocket_embs_y_pred = hotpocket_embs_vals
    hotpocket_comb_y_true, hotpocket_comb_y_pred = hotpocket_comb_vals
    hotpocket_naive_y_true, hotpocket_naive_y_pred = hotpocket_naive_vals
    fpocket_y_true, fpocket_y_pred = fpocket_vals
    prank_y_true, prank_y_pred = prank_vals
    autosite_y_true, autosite_y_pred = autosite_vals
    ligsite_y_true, ligsite_y_pred = ligsite_vals
    pocketminer_y_true, pocketminer_y_pred = pocketminer_vals

    hotpocket_embs_precision, hotpocket_embs_recall, _ = precision_recall_curve(
        hotpocket_embs_y_true, hotpocket_embs_y_pred
    )
    hotpocket_comb_precision, hotpocket_comb_recall, _ = precision_recall_curve(
        hotpocket_comb_y_true, hotpocket_comb_y_pred
    )
    hotpocket_naive_precision, hotpocket_naive_recall, _ = precision_recall_curve(
        hotpocket_naive_y_true, hotpocket_naive_y_pred
    )
    fpocket_precision, fpocket_recall, _ = precision_recall_curve(
        fpocket_y_true, fpocket_y_pred
    )
    prank_precision, prank_recall, _ = precision_recall_curve(
        prank_y_true, prank_y_pred
    )
    autosite_precision, autosite_recall, _ = precision_recall_curve(
        autosite_y_true, autosite_y_pred
    )
    ligsite_precision, ligsite_recall, _ = precision_recall_curve(
        ligsite_y_true, ligsite_y_pred
    )
    pocketminer_precision, pocketminer_recall, _ = precision_recall_curve(
        pocketminer_y_true, pocketminer_y_pred
    )

    if type(cavity_vals) != type(None) and len(cavity_vals[0]) > 0:
        cavity_y_true, cavity_y_pred = cavity_vals
        cavity_precision, cavity_recall, _ = precision_recall_curve(
            cavity_y_true, cavity_y_pred
        )
        cavity_auprc = average_precision_score(cavity_y_true, cavity_y_pred)
    if type(castp_vals) != type(None) and len(castp_vals[0]) > 0:
        castp_y_true, castp_y_pred = castp_vals
        castp_precision, castp_recall, _ = precision_recall_curve(
            castp_y_true, castp_y_pred
        )
        castp_auprc = average_precision_score(castp_y_true, castp_y_pred)

    hotpocket_embs_auprc = average_precision_score(
        hotpocket_embs_y_true, hotpocket_embs_y_pred
    )
    hotpocket_comb_auprc = average_precision_score(
        hotpocket_comb_y_true, hotpocket_comb_y_pred
    )
    hotpocket_naive_auprc = average_precision_score(
        hotpocket_naive_y_true, hotpocket_naive_y_pred
    )
    fpocket_auprc = average_precision_score(fpocket_y_true, fpocket_y_pred)
    prank_auprc = average_precision_score(prank_y_true, prank_y_pred)
    autosite_auprc = average_precision_score(autosite_y_true, autosite_y_pred)
    ligsite_auprc = average_precision_score(ligsite_y_true, ligsite_y_pred)
    pocketminer_auprc = average_precision_score(pocketminer_y_true, pocketminer_y_pred)

    plt.figure()
    if omit_anchor:
        hotpocket_embs_recall, hotpocket_embs_precision = correct_prc(
            hotpocket_embs_recall, hotpocket_embs_precision
        )
        hotpocket_comb_recall, hotpocket_comb_precision = correct_prc(
            hotpocket_comb_recall, hotpocket_comb_precision
        )
        hotpocket_naive_recall, hotpocket_naive_precision = correct_prc(
            hotpocket_naive_recall, hotpocket_naive_precision
        )
        fpocket_recall, fpocket_precision = correct_prc(
            fpocket_recall, fpocket_precision
        )
        prank_recall, prank_precision = correct_prc(prank_recall, prank_precision)
        pocketminer_recall, pocketminer_precision = correct_prc(
            pocketminer_recall, pocketminer_precision
        )
    plt.plot(
        hotpocket_embs_recall,
        hotpocket_embs_precision,
        label="hotpocket Feature Set B, AUPRC=%0.3f" % hotpocket_embs_auprc,
    )
    plt.plot(
        hotpocket_comb_recall,
        hotpocket_comb_precision,
        label="hotpocket Feature Set C, AUPRC=%0.3f" % hotpocket_comb_auprc,
    )
    plt.plot(
        hotpocket_naive_recall,
        hotpocket_naive_precision,
        label="Consensus w/ naive filter, AUPRC=%0.3f" % hotpocket_naive_auprc,
    )
    plt.plot(
        fpocket_recall,
        fpocket_precision,
        label="Fpocket, AUPRC=%0.3f" % fpocket_auprc,
    )
    plt.plot(prank_recall, prank_precision, label="P2Rank, AUPRC=%0.3f" % prank_auprc)
    plt.plot(
        pocketminer_recall,
        pocketminer_precision,
        label="PocketMiner, AUPRC=%0.3f" % pocketminer_auprc,
    )
    plt.plot(
        autosite_recall[1],
        autosite_precision[1],
        label="AutoSite, AUPRC=%0.3f" % autosite_auprc,
        marker="x",
    )
    plt.plot(
        ligsite_recall[1],
        ligsite_precision[1],
        label="LIGSITEcs, AUPRC=%0.3f" % ligsite_auprc,
        marker="x",
    )

    if type(cavity_vals) != type(None) and len(cavity_vals[0]) > 0:
        plt.plot(
            cavity_recall[1],
            cavity_precision[1],
            label="CAVITY, AUPRC=%0.3f" % cavity_auprc,
            marker="x",
        )
    if type(castp_vals) != type(None) and len(castp_vals[0]) > 0:
        plt.plot(
            castp_recall[1],
            castp_precision[1],
            label="CASTp, AUPRC=%0.3f" % castp_auprc,
            marker="x",
        )

    plt.hlines(
        baseline,
        0,
        1,
        colors="gray",
        linestyles="dashed",
        label=f"Baseline = {baseline:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="upper right", fontsize="small")
    plt.grid()
    plt.savefig(out)


def baseline_auprc(y_true):
    return sum(y_true) / len(y_true)
