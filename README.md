## The Human Omnibus of Targetable Pockets
#### Kristy Carpenter & Russ Altman

![HOTPocket logo](hotpocket.png)

:construction: Repository under construction! :construction:

This repository accompanies the manuscript "The Human Omnibus of Targetable Pockets", which is currently unpublished. HOTPocket = "The Human Omnibus of Targetable Pockets", aka the dataset. *hotpocketNN* = the method.

### Citation
In addition to citing our manuscript, please cite **all seven** constituent methods directly:
> B. Huang and M. Schroeder, “LIGSITEcsc: predicting ligand binding sites using the Connolly surface and degree of conservation,” BMC Struct. Biol., vol. 6, no. 19, 2006, doi: 10.1186/1472-6807-6-19.
>
> R. Krivák and D. Hoksza, “P2Rank: machine learning based tool for rapid and accurate prediction of ligand binding sites from protein structure,” J. Cheminformatics, vol. 10, no. 1, p. 39, Dec. 2018, doi: 10.1186/s13321-018-0285-8.
>
> V. Le Guilloux, P. Schmidtke, and P. Tuffery, “Fpocket: An open source platform for ligand pocket detection,” BMC Bioinformatics, vol. 10, no. 1, p. 168, Dec. 2009, doi: 10.1186/1471-2105-10-168.
>
> A. Meller et al., “Predicting locations of cryptic pockets from single protein structures using the PocketMiner graph neural network,” Nat. Commun., vol. 14, no. 1, p. 1177, Mar. 2023, doi: 10.1038/s41467-023-36699-3.
>
> P. A. Ravindranath and M. F. Sanner, “AutoSite: an automated approach for pseudo-ligands prediction—from ligand-binding sites identification to predicting key ligand atoms,” Bioinformatics, vol. 32, no. 20, pp. 3142–3149, Oct. 2016, doi: 10.1093/bioinformatics/btw367.
>
> W. Tian, C. Chen, X. Lei, J. Zhao, and J. Liang, “CASTp 3.0: computed atlas of surface topography of proteins,” Nucleic Acids Res., vol. 46, no. W1, pp. W363–W367, Jul. 2018, doi: 10.1093/nar/gky473.
>
> S. Wang, J. Xie, J. Pei, and L. Lai, “CavityPlus 2022 Update: An Integrated Platform for Comprehensive Protein Cavity Detection and Property Analyses with User-friendly Tools and Cavity Databases,” J. Mol. Biol., vol. 435, no. 14, p. 168141, Jul. 2023, doi: 10.1016/j.jmb.2023.168141.


### Installation
1. Clone this repository: `git clone https://github.com/Helix-Research-Lab/HOTPocket`
2. Install dependencies:
	1. Set up `hotpocket` environment (required): `conda env create --name hotpocket --file=environment.yml`
	2. Install dependencies for PocketMiner (required to run PocketMiner on new structures): follow instructions [here](https://github.com/Mickdub/gvp/tree/pocket_pred)
	3. Install dependencies for ESM2 (required to run *hotpocket* on new structures): follow instructions [here](https://github.com/facebookresearch/esm)
3. Download the data: `./download_data.sh`

### Usage
:construction: Coming soon! :construction: