import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio.PDB import PDBParser

import sys

sys.path.insert(0, "..")
from utils import extract_resids

tqdm.pandas()
parser = PDBParser(QUIET=True)


def get_df_from_outs_dir(outs_dir):
    df = None

    out_csvs = os.listdir(outs_dir)
    out_csvs = [f for f in out_csvs if len(f) > 4 and f[-4:] == ".csv"]

    for fname in tqdm(out_csvs):
        struc_df = pd.read_csv(os.path.join(outs_dir, fname))
        if len(struc_df) == 0:
            continue
        if type(df) == type(None):
            df = struc_df
        else:
            df = pd.concat([df, struc_df])
    return df


def get_top_n(df, scorecol, k=0):
    ascending = scorecol == "pocket id"
    outdf = None
    for pdb in tqdm(df["structure id"].unique()):
        struc_df = df[df["structure id"] == pdb]
        n_ligs = struc_df["n ligands"].tolist()[0]
        if len(struc_df) > n_ligs + k:
            struc_df = struc_df.sort_values(scorecol, ascending=ascending)
            struc_df = struc_df.head(n_ligs + k)
        if type(outdf) == type(None):
            outdf = struc_df
        else:
            outdf = pd.concat([outdf, struc_df])
    return outdf


def prepare_df(df, biolip_df):
    def get_row_pdb(row):
        if "pdb match" in row:
            return row["pdb match"][0]
        else:
            return row["structure id"]

    df = df.drop("uniprot id", axis=1)
    df = df.drop([col for col in df.columns if "Unnamed" in col], axis=1)
    df = df.drop_duplicates(
        [
            col
            for col in [
                "structure id",
                "pocket id",
                "score",
                "contributing method name",
            ]
            if col in df.columns
        ]
    )

    if "pdb match" in df.columns:  # AF2 structure, which is single chain
        count_cols = ["pdb", "chain"]
        match_col = "pdb match"
    else:
        count_cols = "pdb"
        match_col = "structure id"
    n_ligands = biolip_df[count_cols].value_counts()
    df["n ligands"] = df.progress_apply(
        lambda row: (
            n_ligands[row[match_col]]
            if get_row_pdb(row) in biolip_df["pdb"].tolist()
            else 0
        ),
        axis=1,
    )

    df = df[df["n ligands"] > 0]
    df["pocket set"] = df.apply(get_pocket_set, axis=1)
    df["ligands"] = df.apply(
        lambda row: biolip_df[biolip_df["pdb"] == get_row_pdb(row)]["ligand id"]
        .unique()
        .tolist(),
        axis=1,
    )
    return df


def get_pocket_set(row):
    s = set()
    for res in row["pocket res"].split():
        s.add(res)
    return s


def ligand_com_dist_with_memo(
    row,
    memo_df,
    parser,
    pocketcol,
    strucid_col,
    strucs_dir,
    allow_dupes=False,
):
    match = memo_df[
        (memo_df[strucid_col] == row[strucid_col])
        & (memo_df[pocketcol] == row[pocketcol])
    ]
    if (
        "contributing method name" in row
        and "contributing method name" in match.columns
    ):
        match = match[
            match["contributing method name"] == row["contributing method name"]
        ]
    if "score" in row and "score" in match.columns:
        match = match[match["score"] == row["score"]]
    if len(match) == 1 or (allow_dupes and len(match) > 1):
        return match["closest lig to pocket com"].tolist()[0]
    elif not allow_dupes and len(match) > 0:
        print("WARNING: unexpected number of matching rows")
        print(row)
        return ligand_com_dist(row, parser, strucs_dir, pocketcol, strucid_col)
    else:
        return ligand_com_dist(row, parser, strucs_dir, pocketcol, strucid_col)


def str_to_set(s):
    if type(s) == set:
        return s
    if s == "set()":
        return set()
    sset = set(s.strip("{}").replace("'", "").split(","))
    sset = set(elem.strip() for elem in sset)
    return sset


def get_struc_coords(row, parser, strucs_dir, strucid_col="pdb"):
    fname = os.path.join(strucs_dir, "%s.pdb" % row[strucid_col])
    if not os.path.isfile(fname):
        return None
    structure = parser.get_structure("protein", fname)
    resids, _ = extract_resids(None, structure=structure, with_coords=True)
    return resids, structure


def pocket_center_of_mass(
    row, parser, strucs_dir, pocketcol="pred pockets", strucid_col="pdb"
):
    pred_res = str_to_set(row[pocketcol])
    result = get_struc_coords(row, parser, strucs_dir, strucid_col=strucid_col)
    if type(result) == type(None):
        return None
    resids, structure = result
    pred_res_coords = [elem for elem in resids if elem[0] in pred_res]
    if len(pred_res_coords) == 0:
        return None
    com_x = sum([c[1][0] for c in pred_res_coords]) / len(pred_res_coords)
    com_y = sum([c[1][1] for c in pred_res_coords]) / len(pred_res_coords)
    com_z = sum([c[1][2] for c in pred_res_coords]) / len(pred_res_coords)
    return (com_x, com_y, com_z), structure


def ligand_com_dist(
    row, parser, strucs_dir, pocketcol="pred pockets", strucid_col="pdb"
):
    result = pocket_center_of_mass(
        row, parser, strucs_dir, pocketcol=pocketcol, strucid_col=strucid_col
    )
    if type(result) == type(None):
        return None
    pocket_com, structure = result
    closest_dist = 99999999999999
    for model in structure:
        for chain in model:
            for res in chain:
                if res.get_resname().strip() in row["ligands"]:
                    for atom in res:
                        coord = atom.get_coord()
                        dist = (
                            (coord[0] - pocket_com[0]) ** 2
                            + (coord[1] - pocket_com[1]) ** 2
                            + (coord[2] - pocket_com[2]) ** 2
                        ) ** 0.5
                        if dist < closest_dist:
                            closest_dist = dist
        break  # only using first nmr model, naively
    return closest_dist


def get_dccriterion(df, thresh=4):
    return len(df[df["closest lig to pocket com"] <= thresh]) / len(df)


def cluster_pockets_with_score(pocket_list, scores_list, neighbors=0):
    clustered_list = []
    new_scores_list = []
    for idx_p, p in enumerate(pocket_list):
        new_clust = True
        for idx_c, clust in enumerate(clustered_list):
            if len(clust.intersection(set(p))) > 0:
                orig_clust_len = len(clust)
                for res in p:
                    clust.add(res)
                new_clust = False
                new_scores_list[idx_c] = (
                    new_scores_list[idx_c] * orig_clust_len
                    + scores_list[idx_p] * len(p)
                ) / (orig_clust_len + len(p))
                break
            if neighbors > 0:
                for res1 in p:
                    p_chain, p_resnum = extract_resid_components(res1)
                    for res2 in clust:
                        c_chain, c_resnum = extract_resid_components(res2)
                        if p_chain != c_chain:
                            continue
                        if abs(p_resnum - c_resnum) <= neighbors:
                            orig_clust_len = len(clust)
                            for res3 in p:
                                clust.add(res3)
                            new_clust = False
                            new_scores_list[idx_c] = (
                                new_scores_list[idx_c] * orig_clust_len
                                + scores_list[idx_p] * len(p)
                            ) / (orig_clust_len + len(p))
                            break
                    if not new_clust:
                        break
                if not new_clust:
                    break
        if new_clust:
            clustered_list.append(set(p))
            new_scores_list.append(scores_list[idx_p])
    if len(pocket_list) == len(clustered_list):
        return clustered_list, new_scores_list
    else:
        return cluster_pockets_with_score(
            clustered_list, new_scores_list, neighbors=neighbors
        )


def extract_resid_components(resid):
    chain = resid.split("-")[0]
    res = "-".join(resid.split("-")[1:])
    resnum = int(res[1:])
    return chain, resnum


def get_top_n_clusters(df, scorecol, k=2):
    clust_df = None
    for pdb in tqdm(df["structure id"].unique()):
        subdf = df[df["structure id"] == pdb]
        pockets = subdf["pocket set"].tolist()
        scores = subdf[scorecol].tolist()
        clusters, cluster_scores = cluster_pockets_with_score(
            pockets, scores, neighbors=1
        )
        assert len(subdf["n ligands"].unique()) == 1
        n_ligands = subdf["n ligands"].tolist()[0]
        if len(clusters) > n_ligands + k:
            sort_idx = np.argsort(cluster_scores)[::-1]
            sorted_clusters = np.array(clusters)[sort_idx]
            sorted_scores = np.array(cluster_scores)[sort_idx]
            clusters = sorted_clusters[: n_ligands + k]
            cluster_scores = sorted_scores[: n_ligands + k]
        new_df = pd.DataFrame(
            {
                "structure id": [pdb] * len(clusters),
                scorecol: cluster_scores,
                "n ligands": [n_ligands] * len(clusters),
                "pocket set": clusters,
                "ligands": [subdf["ligands"].tolist()[0]] * len(clusters),
            }
        )
        if type(clust_df) == type(None):
            clust_df = new_df
        else:
            clust_df = pd.concat([clust_df, new_df])
    return clust_df


def run_benchmark(dataset, df, strucs_dir, hotpocket_df=None, prefix=""):
    if type(hotpocket_df) == type(None):
        hotpocket_df = get_df_from_outs_dif(os.path.join(dataset, "outs"))

        print("outputs compiled. saving...")
        hotpocket_df.to_csv(
            os.path.join(dataset, "data/%shotpocket_df.csv" % prefix), index=False
        )

    n_ligands = df["pdb"].value_counts()

    hotpocket_df = prepare_df(hotpocket_df, n_ligands, df)

    print("dataset prepared. saving...")
    hotpocket_df.to_csv(
        os.path.join(dataset, "data/%shotpocket_df.csv" % prefix), index=False
    )

    hotpocket_embs_clust = hotpocket_df[hotpocket_df["embs NN score"] >= 0.4].copy()
    hotpocket_comb_clust = hotpocket_df[hotpocket_df["comb NN score"] >= 0.4].copy()

    hotpocket_prank_only = hotpocket_df[
        hotpocket_df["contributing method name"] == "prank"
    ]
    hotpocket_fpocket_only = hotpocket_df[
        hotpocket_df["contributing method name"] == "fpocket"
    ]
    hotpocket_autosite_only = hotpocket_df[
        hotpocket_df["contributing method name"] == "autosite"
    ]
    hotpocket_castp_only = hotpocket_df[
        hotpocket_df["contributing method name"] == "castp"
    ]
    hotpocket_ligsite_only = hotpocket_df[
        hotpocket_df["contributing method name"] == "ligsite"
    ]
    hotpocket_pocketminer_only = hotpocket_df[
        hotpocket_df["contributing method name"] == "pocketminer"
    ]

    hotpocket_autosite_only["pocket id"] = hotpocket_autosite_only.apply(
        lambda row: int(row["pocket id"]), axis=1
    )
    hotpocket_castp_only["pocket id"] = hotpocket_castp_only.apply(
        lambda row: int(row["pocket id"]), axis=1
    )
    hotpocket_ligsite_only["pocket id"] = hotpocket_ligsite_only.apply(
        lambda row: int(row["pocket id"]), axis=1
    )
    hotpocket_pocketminer_only["pocket id"] = hotpocket_pocketminer_only.apply(
        lambda row: int(row["pocket id"]), axis=1
    )

    print("clustered and prank only base datasets prepared. saving...")
    hotpocket_embs_clust.to_csv(
        os.path.join(dataset, "data/%shotpocket_embs_clust.csv" % prefix), index=False
    )
    hotpocket_comb_clust.to_csv(
        os.path.join(dataset, "data/%shotpocket_comb_clust.csv" % prefix), index=False
    )

    hotpocket_embs_top_n_plus_2 = get_top_n(hotpocket_df, "embs NN score", k=2)
    hotpocket_comb_top_n_plus_2 = get_top_n(hotpocket_df, "comb NN score", k=2)
    hotpocket_embs_prank_only_top_n_plus_2 = get_top_n(
        hotpocket_prank_only, "embs NN score", k=2
    )
    hotpocket_comb_prank_only_top_n_plus_2 = get_top_n(
        hotpocket_prank_only, "comb NN score", k=2
    )
    fpocket_top_n_plus_2 = get_top_n(hotpocket_fpocket_only, "score", k=2)
    prank_top_n_plus_2 = get_top_n(hotpocket_prank_only, "score", k=2)
    autosite_top_n_plus_2 = get_top_n(hotpocket_autosite_only, "pocket id", k=2)
    castp_top_n_plus_2 = get_top_n(hotpocket_castp_only, "pocket id", k=2)
    ligsite_top_n_plus_2 = get_top_n(hotpocket_ligsite_only, "pocket id", k=2)
    pocketminer_top_n_plus_2 = get_top_n(hotpocket_pocketminer_only, "score", k=2)

    print("top n+2 pockets found. saving...")
    hotpocket_embs_top_n_plus_2.to_csv(
        os.path.join(dataset, "data/%shotpocket_embs_top_n_plus_2.csv" % prefix),
        index=False,
    )
    hotpocket_comb_top_n_plus_2.to_csv(
        os.path.join(dataset, "data/%shotpocket_comb_top_n_plus_2.csv" % prefix),
        index=False,
    )
    hotpocket_embs_prank_only_top_n_plus_2.to_csv(
        os.path.join(
            dataset, "data/%shotpocket_embs_prank_only_top_n_plus_2.csv" % prefix
        ),
        index=False,
    )
    hotpocket_comb_prank_only_top_n_plus_2.to_csv(
        os.path.join(
            dataset, "data/%shotpocket_comb_prank_only_top_n_plus_2.csv" % prefix
        ),
        index=False,
    )
    fpocket_top_n_plus_2.to_csv(
        os.path.join(dataset, "data/%sfpocket_top_n_plus_2.csv" % prefix), index=False
    )
    prank_top_n_plus_2.to_csv(
        os.path.join(dataset, "data/%sprank_top_n_plus_2.csv" % prefix), index=False
    )
    autosite_top_n_plus_2.to_csv(
        os.path.join(dataset, "data/autosite_top_n_plus_2.csv"), index=False
    )
    castp_top_n_plus_2.to_csv(
        os.path.join(dataset, "data/castp_top_n_plus_2.csv"), index=False
    )
    ligsite_top_n_plus_2.to_csv(
        os.path.join(dataset, "data/ligsite_top_n_plus_2.csv"), index=False
    )
    pocketminer_top_n_plus_2.to_csv(
        os.path.join(dataset, "data/pocketminer_top_n_plus_2.csv"), index=False
    )

    hotpocket_embs_clust_top_n_plus_2 = get_top_n_clusters(
        hotpocket_embs_clust, "embs NN score"
    )
    hotpocket_comb_clust_top_n_plus_2 = get_top_n_clusters(
        hotpocket_comb_clust, "comb NN score"
    )

    print("top n+2 clusters found. saving...")
    hotpocket_embs_clust_top_n_plus_2.to_csv(
        os.path.join(dataset, "data/%shotpocket_embs_clust_top_n_plus_2.csv" % prefix),
        index=False,
    )
    hotpocket_comb_clust_top_n_plus_2.to_csv(
        os.path.join(dataset, "data/%shotpocket_comb_clust_top_n_plus_2.csv" % prefix),
        index=False,
    )

    hotpocket_embs_top_n_plus_2["closest lig to pocket com"] = (
        hotpocket_embs_top_n_plus_2.progress_apply(
            ligand_com_dist,
            args=(
                parser,
                "pocket set",
                "structure id",
                os.path.join(dataset, strucs_dir),
            ),
            axis=1,
        )
    )
    hotpocket_embs_top_n_plus_2.to_csv(
        os.path.join(dataset, "data/%shotpocket_embs_top_n_plus_2.csv" % prefix),
        index=False,
    )

    hotpocket_embs_clust_top_n_plus_2["closest lig to pocket com"] = (
        hotpocket_embs_clust_top_n_plus_2.progress_apply(
            ligand_com_dist,
            args=(
                parser,
                "pocket set",
                "structure id",
                os.path.join(dataset, strucs_dir),
            ),
            axis=1,
        )
    )
    hotpocket_embs_clust_top_n_plus_2.to_csv(
        os.path.join(dataset, "data/%shotpocket_embs_clust_top_n_plus_2.csv" % prefix),
        index=False,
    )

    hotpocket_comb_top_n_plus_2["closest lig to pocket com"] = (
        hotpocket_comb_top_n_plus_2.progress_apply(
            ligand_com_dist_with_memo,
            args=(
                hotpocket_embs_top_n_plus_2,
                parser,
                "pocket set",
                "structure id",
                os.path.join(dataset, strucs_dir),
            ),
            axis=1,
        )
    )
    hotpocket_comb_top_n_plus_2.to_csv(
        os.path.join(dataset, "data/%shotpocket_comb_top_n_plus_2.csv" % prefix),
        index=False,
    )

    hotpocket_comb_clust_top_n_plus_2["closest lig to pocket com"] = (
        hotpocket_comb_clust_top_n_plus_2.progress_apply(
            ligand_com_dist_with_memo,
            args=(
                hotpocket_embs_clust_top_n_plus_2,
                parser,
                "pocket set",
                "structure id",
                os.path.join(dataset, strucs_dir),
            ),
            axis=1,
        )
    )
    hotpocket_comb_clust_top_n_plus_2.to_csv(
        os.path.join(dataset, "data/%shotpocket_comb_clust_top_n_plus_2.csv" % prefix),
        index=False,
    )

    fpocket_top_n_plus_2["closest lig to pocket com"] = (
        fpocket_top_n_plus_2.progress_apply(
            ligand_com_dist_with_memo,
            args=(
                hotpocket_embs_top_n_plus_2,
                parser,
                "pocket set",
                "structure id",
                os.path.join(dataset, strucs_dir),
            ),
            axis=1,
        )
    )
    fpocket_top_n_plus_2.to_csv(
        os.path.join(dataset, "data/%sfpocket_top_n_plus_2.csv" % prefix), index=False
    )

    prank_top_n_plus_2["closest lig to pocket com"] = prank_top_n_plus_2.progress_apply(
        ligand_com_dist_with_memo,
        args=(
            hotpocket_embs_top_n_plus_2,
            parser,
            "pocket set",
            "structure id",
            os.path.join(dataset, strucs_dir),
        ),
        axis=1,
    )
    prank_top_n_plus_2.to_csv(
        os.path.join(dataset, "data/%sprank_top_n_plus_2.csv" % prefix), index=False
    )

    hotpocket_embs_prank_only_top_n_plus_2["closest lig to pocket com"] = (
        hotpocket_embs_prank_only_top_n_plus_2.progress_apply(
            ligand_com_dist_with_memo,
            args=(
                prank_top_n_plus_2,
                parser,
                "pocket set",
                "structure id",
                os.path.join(dataset, strucs_dir),
            ),
            axis=1,
        )
    )
    hotpocket_embs_prank_only_top_n_plus_2.to_csv(
        os.path.join(
            dataset, "data/%shotpocket_embs_prank_only_top_n_plus_2.csv" % prefix
        ),
        index=False,
    )

    hotpocket_comb_prank_only_top_n_plus_2["closest lig to pocket com"] = (
        hotpocket_comb_prank_only_top_n_plus_2.progress_apply(
            ligand_com_dist_with_memo,
            args=(
                prank_top_n_plus_2,
                parser,
                "pocket set",
                "structure id",
                os.path.join(dataset, strucs_dir),
            ),
            axis=1,
        )
    )
    hotpocket_comb_prank_only_top_n_plus_2.to_csv(
        os.path.join(
            dataset, "data/%shotpocket_comb_prank_only_top_n_plus_2.csv" % prefix
        ),
        index=False,
    )

    autosite_top_n_plus_2["closest lig to pocket com"] = (
        autosite_top_n_plus_2.progress_apply(
            ligand_com_dist_with_memo,
            args=(
                hotpocket_embs_top_n_plus_2,
                parser,
                "pocket set",
                "structure id",
                os.path.join(dataset, strucs_dir),
            ),
            axis=1,
        )
    )
    autosite_top_n_plus_2.to_csv(
        os.path.join(dataset, "data/autosite_top_n_plus_2.csv"), index=False
    )

    castp_top_n_plus_2["closest lig to pocket com"] = castp_top_n_plus_2.progress_apply(
        ligand_com_dist_with_memo,
        args=(
            hotpocket_embs_top_n_plus_2,
            parser,
            "pocket set",
            "structure id",
            os.path.join(dataset, strucs_dir),
        ),
        axis=1,
    )
    castp_top_n_plus_2.to_csv(
        os.path.join(dataset, "data/castp_top_n_plus_2.csv"), index=False
    )

    ligsite_top_n_plus_2["closest lig to pocket com"] = (
        ligsite_top_n_plus_2.progress_apply(
            ligand_com_dist_with_memo,
            args=(
                hotpocket_embs_top_n_plus_2,
                parser,
                "pocket set",
                "structure id",
                os.path.join(dataset, strucs_dir),
            ),
            axis=1,
        )
    )
    ligsite_top_n_plus_2.to_csv(
        os.path.join(dataset, "data/ligsite_top_n_plus_2.csv"), index=False
    )

    pocketminer_top_n_plus_2["closest lig to pocket com"] = (
        pocketminer_top_n_plus_2.progress_apply(
            ligand_com_dist_with_memo,
            args=(
                hotpocket_embs_top_n_plus_2,
                parser,
                "pocket set",
                "structure id",
                os.path.join(dataset, strucs_dir),
            ),
            axis=1,
        )
    )
    pocketminer_top_n_plus_2.to_csv(
        os.path.join(dataset, "data/pocketminer_top_n_plus_2.csv"), index=False
    )

    print("top n+2 ligand distances found.")

    hotpocket_embs_top_n = get_top_n(hotpocket_embs_top_n_plus_2, "embs NN score")
    hotpocket_comb_top_n = get_top_n(hotpocket_comb_top_n_plus_2, "comb NN score")
    hotpocket_embs_clust_top_n = get_top_n(
        hotpocket_embs_clust_top_n_plus_2, "embs NN score"
    )
    hotpocket_comb_clust_top_n = get_top_n(
        hotpocket_comb_clust_top_n_plus_2, "comb NN score"
    )
    hotpocket_embs_prank_only_top_n = get_top_n(
        hotpocket_embs_prank_only_top_n_plus_2, "embs NN score"
    )
    hotpocket_comb_prank_only_top_n = get_top_n(
        hotpocket_comb_prank_only_top_n_plus_2, "comb NN score"
    )
    fpocket_top_n = get_top_n(fpocket_top_n_plus_2, "score")
    prank_top_n = get_top_n(prank_top_n_plus_2, "score")
    autosite_top_n = get_top_n(autosite_top_n_plus_2, "pocket id")
    castp_top_n = get_top_n(castp_top_n_plus_2, "pocket id")
    ligsite_top_n = get_top_n(ligsite_top_n_plus_2, "pocket id")
    pocketminer_top_n = get_top_n(pocketminer_top_n_plus_2, "score")

    print("top n ligand distances found. saving...")
    hotpocket_embs_top_n.to_csv(
        os.path.join(dataset, "data/%shotpocket_embs_top_n.csv" % prefix), index=False
    )
    hotpocket_comb_top_n.to_csv(
        os.path.join(dataset, "data/%shotpocket_comb_top_n.csv" % prefix), index=False
    )
    hotpocket_embs_prank_only_top_n.to_csv(
        os.path.join(dataset, "data/%shotpocket_embs_prank_only_top_n.csv" % prefix),
        index=False,
    )
    hotpocket_comb_prank_only_top_n.to_csv(
        os.path.join(dataset, "data/%shotpocket_comb_prank_only_top_n.csv" % prefix),
        index=False,
    )
    fpocket_top_n.to_csv(
        os.path.join(dataset, "data/%sfpocket_top_n.csv" % prefix), index=False
    )
    prank_top_n.to_csv(
        os.path.join(dataset, "data/%sprank_top_n.csv" % prefix), index=False
    )
    hotpocket_embs_clust_top_n.to_csv(
        os.path.join(dataset, "data/%shotpocket_embs_clust_top_n.csv" % prefix),
        index=False,
    )
    hotpocket_comb_clust_top_n.to_csv(
        os.path.join(dataset, "data/%shotpocket_comb_clust_top_n.csv" % prefix),
        index=False,
    )
    autosite_top_n.to_csv(os.path.join(dataset, "data/autosite_top_n.csv"), index=False)
    castp_top_n.to_csv(os.path.join(dataset, "data/castp_top_n.csv"), index=False)
    ligsite_top_n.to_csv(os.path.join(dataset, "data/ligsite_top_n.csv"), index=False)
    pocketminer_top_n.to_csv(
        os.path.join(dataset, "data/pocketminer_top_n.csv"), index=False
    )

    print("hotpocket embs top n: %0.4f" % get_dccriterion(hotpocket_embs_top_n))
    print(
        "hotpocket embs top n+2: %0.4f" % get_dccriterion(hotpocket_embs_top_n_plus_2)
    )
    print("hotpocket comb top n: %0.4f" % get_dccriterion(hotpocket_comb_top_n))
    print(
        "hotpocket comb top n+2: %0.4f" % get_dccriterion(hotpocket_comb_top_n_plus_2)
    )
    print(
        "hotpocket embs (clustered) top n: %0.4f"
        % get_dccriterion(hotpocket_embs_clust_top_n)
    )
    print(
        "hotpocket embs (clustered) top n+2: %0.4f"
        % get_dccriterion(hotpocket_embs_clust_top_n_plus_2)
    )
    print(
        "hotpocket comb (clustered) top n: %0.4f"
        % get_dccriterion(hotpocket_comb_clust_top_n)
    )
    print(
        "hotpocket comb (clustered) top n+2: %0.4f"
        % get_dccriterion(hotpocket_comb_clust_top_n_plus_2)
    )
    print(
        "hotpocket embs (prank only) top n: %0.4f"
        % get_dccriterion(hotpocket_embs_prank_only_top_n)
    )
    print(
        "hotpocket embs (prank only) top n+2: %0.4f"
        % get_dccriterion(hotpocket_embs_prank_only_top_n_plus_2)
    )
    print(
        "hotpocket comb (prank only) top n: %0.4f"
        % get_dccriterion(hotpocket_comb_prank_only_top_n)
    )
    print(
        "hotpocket comb (prank only) top n+2: %0.4f"
        % get_dccriterion(hotpocket_comb_prank_only_top_n_plus_2)
    )
    print("fpocket top n: %0.4f" % get_dccriterion(fpocket_top_n))
    print("fpocket top n+2: %0.4f" % get_dccriterion(fpocket_top_n_plus_2))
    print("prank top n: %0.4f" % get_dccriterion(prank_top_n))
    print("prank top n+2: %0.4f" % get_dccriterion(prank_top_n_plus_2))
    print("autosite top n: %0.4f" % get_dccriterion(autosite_top_n))
    print("autosite top n+2: %0.4f" % get_dccriterion(autosite_top_n_plus_2))
    print("castp top n: %0.4f" % get_dccriterion(castp_top_n))
    print("castp top n+2: %0.4f" % get_dccriterion(castp_top_n_plus_2))
    print("ligsite top n: %0.4f" % get_dccriterion(ligsite_top_n))
    print("ligsite top n+2: %0.4f" % get_dccriterion(ligsite_top_n_plus_2))
    print("pocketminer top n: %0.4f" % get_dccriterion(pocketminer_top_n))
    print("pocketminer top n+2: %0.4f" % get_dccriterion(pocketminer_top_n_plus_2))


def run_benchmark_with_exclude(data_dir, exclude=[]):
    dfs = {
        "hotpocket embs": {
            "top n": pd.read_csv(os.path.join(data_dir, "hotpocket_embs_top_n.csv")),
            "top n+2": pd.read_csv(
                os.path.join(data_dir, "hotpocket_embs_top_n_plus_2.csv")
            ),
        },
        "hotpocket comb": {
            "top n": pd.read_csv(os.path.join(data_dir, "hotpocket_comb_top_n.csv")),
            "top n+2": pd.read_csv(
                os.path.join(data_dir, "hotpocket_comb_top_n_plus_2.csv")
            ),
        },
        "hotpocket clustered embs": {
            "top n": pd.read_csv(
                os.path.join(data_dir, "hotpocket_embs_clust_top_n.csv")
            ),
            "top n+2": pd.read_csv(
                os.path.join(data_dir, "hotpocket_embs_clust_top_n_plus_2.csv")
            ),
        },
        "hotpocket clustered comb": {
            "top n": pd.read_csv(
                os.path.join(data_dir, "hotpocket_comb_clust_top_n.csv")
            ),
            "top n+2": pd.read_csv(
                os.path.join(data_dir, "hotpocket_comb_clust_top_n_plus_2.csv")
            ),
        },
        "hotpocket prank only embs": {
            "top n": pd.read_csv(
                os.path.join(data_dir, "hotpocket_embs_prank_only_top_n.csv")
            ),
            "top n+2": pd.read_csv(
                os.path.join(data_dir, "hotpocket_embs_prank_only_top_n_plus_2.csv")
            ),
        },
        "hotpocket prank only comb": {
            "top n": pd.read_csv(
                os.path.join(data_dir, "hotpocket_comb_prank_only_top_n.csv")
            ),
            "top n+2": pd.read_csv(
                os.path.join(data_dir, "hotpocket_comb_prank_only_top_n_plus_2.csv")
            ),
        },
        "fpocket": {
            "top n": pd.read_csv(os.path.join(data_dir, "fpocket_top_n.csv")),
            "top n+2": pd.read_csv(os.path.join(data_dir, "fpocket_top_n_plus_2.csv")),
        },
        "prank": {
            "top n": pd.read_csv(os.path.join(data_dir, "prank_top_n.csv")),
            "top n+2": pd.read_csv(os.path.join(data_dir, "prank_top_n_plus_2.csv")),
        },
        "autosite": {
            "top n": pd.read_csv(os.path.join(data_dir, "autosite_top_n.csv")),
            "top n+2": pd.read_csv(os.path.join(data_dir, "autosite_top_n_plus_2.csv")),
        },
        "castp": {
            "top n": pd.read_csv(os.path.join(data_dir, "castp_top_n.csv")),
            "top n+2": pd.read_csv(os.path.join(data_dir, "castp_top_n_plus_2.csv")),
        },
        "ligsite": {
            "top n": pd.read_csv(os.path.join(data_dir, "ligsite_top_n.csv")),
            "top n+2": pd.read_csv(os.path.join(data_dir, "ligsite_top_n_plus_2.csv")),
        },
        "pocketminer": {
            "top n": pd.read_csv(os.path.join(data_dir, "pocketminer_top_n.csv")),
            "top n+2": pd.read_csv(
                os.path.join(data_dir, "pocketminer_top_n_plus_2.csv")
            ),
        },
        "cavity": {
            "top n": pd.read_csv(os.path.join(data_dir, "cavity_top_n.csv")),
            "top n+2": pd.read_csv(os.path.join(data_dir, "cavity_top_n_plus_2.csv")),
        },
    }

    for k1 in dfs.keys():
        for k2 in dfs[k1].keys():
            if len(exclude) > 0:
                tmp_df = dfs[k1][k2].copy()
                tmp_df["include"] = tmp_df.apply(
                    lambda row: not row["structure id"] in exclude, axis=1
                )
                dfs[k1][k2] = tmp_df[tmp_df["include"]]
            print("%s %s: %0.4f" % (k1, k2, get_dccriterion(dfs[k1][k2])))
