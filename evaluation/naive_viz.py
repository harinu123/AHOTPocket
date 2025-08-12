import pandas as pd


def get_prefilter_viz(struc, dfs, filtered=False):
    if filtered:
        prefix = "filtered_"
    else:
        prefix = ""
    sel_strs_list = []
    for dfname in dfs.keys():
        subdf = dfs[dfname]
        subdf = subdf[subdf["structure id"] == struc]
        for idx, pocket in enumerate(subdf["pocket res"].tolist()):
            sel_str = pymol_select(pocket.split(), "%s%s%d" % (prefix, dfname, idx))
            sel_strs_list.append(sel_str)

            if len(sel_strs_list) > 70:
                print(";".join(sel_strs_list))
                print("\n\n")
                sel_strs_list = []
    print(";".join(sel_strs_list))


def pymol_select(res_list, selname):
    sel_str = "select %s," % selname
    for res in res_list:
        chain = res.split("-")[0]
        res_num = int("-".join(res.split("-")[1:])[1:])
        sel_str += "(chain %s and resi %d) or " % (chain, res_num)
    sel_str = sel_str[:-4]  # get rid of trailing " or "
    return sel_str


def heatmap_sel(pred_sets, filtered=False):
    if filtered:
        prefix = "filtered_"
    else:
        prefix = ""

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
        sel_strings.append(pymol_select(reslist, "%s%d" % (prefix, n)))
    return sel_strings


def get_pocket_res(pockets):
    res = set()
    for p in pockets:
        reslist = p.strip().split(" ")
        for r in reslist:
            res.add(r)
    return res


def get_heatmap_for_prot(struc, dfs, filtered=False):
    res_list = []
    for dfname in dfs.keys():
        subdf = dfs[dfname]
        subdf = subdf[subdf["structure id"] == struc]
        pockets = subdf["pocket res"].tolist()
        res_list.append(get_pocket_res(pockets))
    sel_strings = heatmap_sel(res_list, filtered=filtered)
    for sel_string in sel_strings:
        print(sel_string)
        print("\n\n")
