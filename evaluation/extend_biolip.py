from Bio import PDB
from Bio.PDB import PDBParser
from make_db import extract_resids
import pickle
import pandas as pd
import os
import numpy as np


def str_to_set(s):
    if type(s) == set:
        return s
    if s == "set()":
        return set()
    sset = set(s.strip("{}").replace("'", "").split(","))
    sset = set(elem.strip() for elem in sset)
    return sset


def no_polymer(df):
    return df[
        (df["ligand id"] != "peptide")
        & (df["ligand id"] != "dna")
        & (df["ligand id"] != "rna")
    ]


def extend_pocket(row, parser, col, thresh):
    result = get_struc_coords(row, parser)
    if type(result) == type(None):
        return row[col]
    resids, _ = result
    biolip_res_with_coord = [elem for elem in resids if elem[0] in str_to_set(row[col])]
    biolip_extend = [
        elem
        for elem in resids
        if dist_within_thresh(biolip_res_with_coord, elem, thresh)
    ]
    biolip_extend = [elem[0] for elem in biolip_extend]
    biolip_extend = set(biolip_extend)
    return biolip_extend


def get_struc_coords(row, parser, strucid_col="pdb", strucsdir="all_biolip_pdb"):
    fname = os.path.join(strucsdir, "%s.pdb" % row[strucid_col])
    if not os.path.isfile(fname):
        return None
    structure = parser.get_structure("protein", fname)
    resids, _ = extract_resids(None, structure=structure, with_coords=True)
    return resids, structure


def dist_within_thresh(coords, c, thresh=5):
    if len(c[1]) != 3:
        return False
    x1, y1, z1 = c[1]
    for res in coords:
        if len(res[1]) != 3:
            continue
        x2, y2, z2 = res[1]
        if ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5 <= thresh:
            return True
    return False


def pocket_center_of_mass(
    row, parser, pocketcol="pred pockets", strucid_col="pdb", strucsdir="all_biolip_pdb"
):
    pred_res = str_to_set(row[pocketcol])
    result = get_struc_coords(row, parser, strucid_col=strucid_col, strucsdir=strucsdir)
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
    row, parser, pocketcol="pred pockets", strucid_col="pdb", strucsdir="all_biolip_pdb"
):
    result = pocket_center_of_mass(
        row, parser, pocketcol=pocketcol, strucid_col=strucid_col, strucsdir=strucsdir
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


parser = PDBParser(QUIET=True)

# embs_filt = pd.read_csv("largescale_eval_dfs/emb_filtered.csv")
# comb_filt = pd.read_csv("largescale_eval_dfs/comb_filtered.csv")
# unfilt = pd.read_csv("largescale_eval_dfs/unfiltered.csv")
df = pd.read_csv("largescale_eval_dfs/emb_no_polymer_redundant.csv")
