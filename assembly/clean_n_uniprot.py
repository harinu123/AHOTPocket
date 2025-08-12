import pandas as pd
from tqdm import tqdm
import argparse
import pickle
import requests
import json


def pdb_chain_map(pdb):
    map_dict = dict()
    response = requests.get(
        "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/%s" % pdb.lower()
    )
    if response.status_code != 200:
        return map_dict
    j = json.loads(response.text)
    j_uniprot = j[pdb.lower()]["UniProt"]
    for uniprot in j_uniprot:
        map_dict[uniprot] = [
            elem["chain_id"] for elem in j_uniprot[uniprot]["mappings"]
        ]
    return map_dict


def uniprot_pdb_to_chain(row, memo):
    if not row["structure id"] in memo.keys():
        return ""
    chain_map = memo[row["structure id"]]
    if not row["uniprot id"] in chain_map.keys():
        return ""
    return ", ".join(chain_map[row["uniprot id"]])


def pocket_to_chain(row):
    pocket_res = row["pocket res"].split()
    chains = set()
    for r in pocket_res:
        chains.add(r.split("-")[0])
    return ", ".join(list(chains))


def chains_match(row):
    uniprot_chain = row["chain for uniprot"]
    pocket_chain = row["chain for pocket"]
    u_chain_set = set(uniprot_chain.split(", "))
    p_chain_set = set(pocket_chain.split(", "))
    if len(u_chain_set) == 0:
        return "error"
    if len(u_chain_set.intersection(p_chain_set)) > 0:
        return True
    else:
        return False


def process_chain_df(df, memo={}):
    df = df.loc[df["structure type"] == "PDB"]
    all_pdbs = df["structure id"].unique().tolist()
    for pdb in all_pdbs:
        if not pdb in memo.keys():
            memo[pdb] = pdb_chain_map(pdb)

    df["chain for uniprot"] = df.apply(uniprot_pdb_to_chain, args=(memo,), axis=1)
    df["chain for pocket"] = df.apply(pocket_to_chain, axis=1)
    df["chains match"] = df.apply(chains_match, axis=1)
    df = df.loc[df["chains match"] == True]
    df = df.drop(columns=["chains match", "chain for uniprot", "chain for pocket"])
    return df, memo


def main(args):
    pdbs = pd.read_csv("data/pdbs.csv", index_col=0)
    stop = min(args.start + args.n, len(pdbs))
    pdbs = pdbs[args.start : stop]["pdb"]
    memo = dict()
    for pdb in tqdm(pdbs):
        memo[pdb] = pdb_chain_map(pdb)
    pickle.dump(memo, open("data/memo_%d_%d.p" % (args.start, stop), "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, help="index from which to start")
    parser.add_argument("-n", type=int, help="how many pdbs to process")
    args = parser.parse_args()
    main(args)
