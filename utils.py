from Bio import PDB
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
import os
import numpy as np

AA_3TO1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "PTR": "Y",  # phosphotyrosine
    "TPO": "T",  # phosphothreonine
    "MSE": "M",  # selenomethionine
    "CSO": "C",  # hydroxycysteine
    "CAS": "C",  # dimethylarseniccysteine
    "LLP": "K",  # pyridoxyllysine-monophosphate
    "SCY": "C",  # s-acetyl-cysteine
    "DSN": "S",  # d-serine
    "CSS": "C",  # mercaptocysteine
    "SEP": "S",  # phosphoserine
    "CSX": "C",  # s-oxy cysteine
    "ALS": "S",  # (3S)-3-(sulfooxy)-L-serine
    "NEP": "H",  # n1-phosphonohistidine
    "MK8": "L",  # 2-methyl-L-norleucine
    "MLY": "K",  # dimethyl-lysine
}


def pdbs_to_fasta(pdbdir, outname):
    parser = PDB.PDBParser(QUIET=True)
    sequences = []
    longest = 0
    for pdbname in os.listdir(pdbdir):
        pdbfile = os.path.join(pdbdir, pdbname)
        structure = parser.get_structure("protein", pdbfile)
        struc_id = pdbname[:-4]

        for i, chain in enumerate(structure.get_chains()):
            seq = [
                residue.get_resname()
                for residue in PDB.Selection.unfold_entities(chain, "R")
                if PDB.is_aa(residue)
            ]
            seq = [res for res in seq if res in AA_3TO1.keys()]
            seq = [AA_3TO1[residue] for residue in seq]
            seq = "".join(seq)
            if len(seq) > longest:
                longest = len(seq)
            record = SeqRecord(
                Seq(seq), id="%s_%s" % (struc_id, chain.id), description=""
            )
            sequences.append(record)
    SeqIO.write(sequences, outname, "fasta")
    print("longest seq: %d" % longest)


def structure_type_proportion(dfname):
    df = pd.read_csv(dfname)
    tot_prot = len(df["uniprot id"].unique())
    df_af2 = df.loc[df["structure type"] == "AF2"]
    df_pdb = df.loc[df["structure type"] == "PDB"]
    af2_prot = len(df_af2["uniprot id"].unique())
    pdb_prot = len(df_pdb["uniprot id"].unique())

    print("Total # proteins: %d" % tot_prot)
    print("# AF2 proteins: %d" % af2_prot)
    print("# PDB proteins: %d" % pdb_prot)


# a function that will extract the residue id numbers and amino acid identity
# from a specified pdb file; principally meant for use with pocketminer outputs
# can also extract only resids within a certain radius r of a central coordinate
#
# params:
# fpath (str) - string giving location of pdb file for processing
# cen (np.ndarray) - optional, central coordinates from which to measure distance
# r (int) - optional, radius in angstroms
# structure (BioPython PDBParser structure object) - optional, preloaded structure object
# with_coords (bool) - flag for whether or not to include coordinates with ids
#
# returns:
# resids (list of str) - residue id of the desired residue, in the form [Chain]-[1letter AA code][pdb index]
def extract_resids(fpath, cen=None, r=None, structure=None, with_coords=False):
    if structure is None:
        pdbparser = PDB.PDBParser(QUIET=True)
        try:
            structure = pdbparser.get_structure("protein", fpath)
        except:
            try:
                pocketminer_fpath = os.path.join(
                    "data",
                    "pocketminer",
                    "%s.pdb" % os.path.split(fpath)[1].split(".")[0],
                )
                structure = pdbparser.get_structure("protein", pocketminer_fpath)
            except:
                fpath = move_process_struct(fpath, os.getcwd())
                structure = pdbparser.get_structure("protein", fpath)
                rm_parts = ["rm", fpath]
                subprocess.run(rm_parts)
    resids = []
    for model in structure:
        for chain in model:
            chainname = chain.get_id()
            for residue in chain:
                resname3 = residue.get_resname()
                resnum = residue.get_id()[1]
                if not resname3 in AA_3TO1.keys():
                    continue
                resname1 = AA_3TO1[resname3]
                resid = "%s-%s%d" % (chainname, resname1, resnum)
                if with_coords and "CA" in residue:
                    resid = (resid, residue["CA"].coord)
                if not cen is None and not r is None:
                    for atom in residue:
                        coord = atom.get_coord()
                        if (
                            (coord[0] - cen[0]) ** 2
                            + (coord[1] - cen[1]) ** 2
                            + (coord[2] - cen[2]) ** 2
                        ) ** 0.5 <= r:
                            resids.append(resid)
                            break
                else:
                    resids.append(resid)
        break  # only considering first model -- naive!!
    return resids, structure


# method to get all pairwise distances from a pdb file
# for use in pocketminer clustering
# params:
# structure (biopython pdbparser structure object) - structure object to iterate over with pdb information
# seq_resids (list of str) - residue ids in [CHAIN]-[AA ID LENGTH1][AA NUM] format
#
# returns:
# dists (np array) - all pairwise c-alpha distances
def get_pairwise_distances(structure, seq_resids, with_coords=False):
    if with_coords:
        coords = [r[1] for r in seq_resids]
    else:
        chains = set()
        for res in seq_resids:
            chains.add(res.split("-")[0])
        coords = []
        for model in structure:
            for chain in model:
                if not chain.get_id() in chains:
                    continue
                for residue in chain:
                    if not PDB.is_aa(residue, standard=True):
                        continue
                    for atom in residue:
                        if atom.get_name() == "CA":
                            coords.append(atom.get_coord())
            break
    dists = np.zeros((len(coords), len(coords)))
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if len(coords[i]) < 3 or len(coords[j]) < 3:
                d = 999999999999999999999999999999
            else:
                x1, y1, z1 = coords[i]
                x2, y2, z2 = coords[j]
                d = ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
            dists[i, j] = d
            dists[j, i] = d
    return dists


# returns the n_neighbors closest neighbors to the specified residue
#
# params:
# idx (int) - index of the central residue
# pairwise_distances (np array) - all pairwise c-alpha distances
# n_neighbors (int) - number of neighbors to return
#
# returns:
# neighbor_idx (list) - list of indices of neighbors and the original index
def get_neighbors(idx, pairwise_distances, n_neighbors):
    dist_to_center = pairwise_distances[idx]
    neighbor_idx = np.argsort(dist_to_center)[
        : n_neighbors + 1
    ]  # add 1 because distance to self is zero
    return neighbor_idx
