import pandas as pd
import argparse


def extract_id(res_str):
    chain, res = res_str.split("-")
    assert res[1:].isnumeric()
    return "(chain %s and resi %s)" % (chain, res[1:])


def pymol_sel(pocket_str, sel_name, chain=None):
    if len(pocket_str) == 0:
        return ""
    reslist = pocket_str.strip().split(" ")
    sel_str = ""
    i = 0
    while len(sel_str) == 0 and i < len(reslist):
        if not type(chain) == type(None) and reslist[i].split("-")[0] != chain:
            i += 1
        else:
            sel_str = "select %s, (%s" % (sel_name, extract_id(reslist[i]))
    if i < len(reslist):
        for res in reslist[i + 1 :]:
            if not type(chain) == type(None):
                if res.split("-")[0] != chain:
                    continue
            try:
                sel_str += " or %s" % extract_id(res)
            except:
                continue
        sel_str += ")"
    return sel_str


def heatmap_sel(pred_sets):
    res_dict = dict()
    for s in pred_sets:
        for res in s:
            if res in res_dict.keys():
                res_dict[res] += 1
            else:
                res_dict[res] = 1
    heat_dict = dict()
    for key, value in res_dict.items():
        if value in heat_dict.keys():
            heat_dict[value].append(key)
        else:
            heat_dict[value] = [key]
    sel_strings = []
    for n, reslist in heat_dict.items():
        sel_strings.append(pymol_sel(" ".join(reslist), n))
    return sel_strings


def add_chain(row, chain_col, pocket_res_col):
    s = ""
    for elem in row[pocket_res_col].split(" "):
        s += "%s-%s " % (row[chain_col], elem)
    return s.strip()


def get_pocket_res(pockets):
    res = set()
    for p in pockets:
        reslist = p.strip().split(" ")
        for r in reslist:
            res.add(r)
    return res


def get_pocket_sel(pockets, pocket_ids):
    sel_strings = []
    for pid, p in zip(pocket_ids, pockets):
        sel_string = pymol_sel(p, pid)
        sel_strings.append(sel_string)
    return sel_strings


def prep_pockets(
    df,
    method,
    prot,
    prot_id_col,
    pocket_res_col,
    pocket_id_col=False,
    chain_col=False,
    filter=False,
    thresh=0.5,
):
    df = df.loc[df[prot_id_col] == prot]
    if filter:
        df = df.loc[df["score"] >= thresh]
    if chain_col:
        pockets = df.apply(add_chain, args=(chain_col, pocket_res_col), axis=1).tolist()
    else:
        pockets = df[pocket_res_col].tolist()
    if pocket_id_col:
        pocket_ids = ["%s_%s" % (method, idx) for idx in df[pocket_id_col].tolist()]
    else:
        pocket_ids = ["%s_%d" % (method, idx) for idx in range(1, len(pockets) + 1)]
    return pockets, pocket_ids


def main(args):
    biolip = pd.read_csv("data/known_pockets.csv")
    ligsite = pd.read_csv("data/pred_ligsite_pockets.csv")
    castp = pd.read_csv("data/pred_castp_pockets.csv")

    if args.filter:
        fpocket = pd.read_csv("%s_scored_fpocket_pockets.csv" % args.prot)
        # castp = pd.read_csv("%s_scored_castp_pockets.csv" % args.prot)
        cavity = pd.read_csv("data/pred_cavityspace_pockets_strong.csv")
    else:
        fpocket = pd.read_csv("data/pred_fpocket_pockets.csv")
        castp = pd.read_csv("data/pred_castp_pockets.csv")
        cavity = pd.read_csv("data/pred_cavityspace_pockets.csv")

    biolip_dict = {
        "df": biolip,
        "prot_id_col": "pdb",
        "pocket_res_col": "binding site residues, pdb numbering",
        "pocket_id_col": False,
        "chain_col": "chain",
        "filter": False,
        "thresh": None,
    }
    castp_dict = {
        "df": castp,
        "prot_id_col": "structure id",
        "pocket_res_col": "pocket res",
        "pocket_id_col": "pocket id",
        "chain_col": False,
        "filter": False,
        "thresh": None,
    }
    # "filter": args.filter,
    # "thresh": 50}
    cavity_dict = {
        "df": cavity,
        "prot_id_col": "structure id",
        "pocket_res_col": "pocket res",
        "pocket_id_col": "pocket id",
        "chain_col": False,
        "filter": False,
        "thresh": None,
    }
    fpocket_dict = {
        "df": fpocket,
        "prot_id_col": "structure id",
        "pocket_res_col": "pocket res",
        "pocket_id_col": "pocket id",
        "chain_col": False,
        "filter": args.filter,
        "thresh": 0.5,
    }
    ligsite_dict = {
        "df": ligsite,
        "prot_id_col": "structure id",
        "pocket_res_col": "pocket res",
        "pocket_id_col": "pocket id",
        "chain_col": False,
        "filter": False,
        "thresh": None,
    }
    info_dict = {
        "biolip": biolip_dict,
        "castp": castp_dict,
        "cavityspace": cavity_dict,
        "fpocket": fpocket_dict,
        "ligsite": ligsite_dict,
    }
    sel_strings = []
    res_list = []
    for method, d in info_dict.items():
        pockets, pocket_ids = prep_pockets(
            d["df"],
            method,
            args.prot,
            d["prot_id_col"],
            d["pocket_res_col"],
            pocket_id_col=d["pocket_id_col"],
            chain_col=d["chain_col"],
            filter=d["filter"],
            thresh=d["thresh"],
        )
        if args.heatmap:
            if method == "biolip":
                continue
            res_list.append(get_pocket_res(pockets))
        else:
            sel_strings += get_pocket_sel(pockets, pocket_ids)
    if args.heatmap:
        sel_strings = heatmap_sel(res_list)
    out = open(args.out, "w")
    for sel_string in sel_strings:
        out.write("%s;" % sel_string)
    out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prot", type=str, help="structure id of protein for which to obtain pockets"
    )
    parser.add_argument("--out", type=str, help="outfile name")
    parser.add_argument(
        "--filter",
        action="store_true",
        help="flag to filter for high scoring pockets only",
    )
    parser.add_argument(
        "--heatmap",
        action="store_true",
        help="flag to get heatmap-style consensus pockets instead of individual pockets",
    )
    args = parser.parse_args()
    main(args)
