#!/bin/bash

DATA_ORIGIN=/oak/stanford/groups/rbaltman/kcarp/hotpocket2.0/data

mkdir data
cp $DATA_ORIGIN/proteome_structure_files_canonical.csv data
cp $DATA_ORIGIN/known_pockets.csv data
cp $DATA_ORIGIN/pred_autosite_pockets_cleaned.csv $DATA_ORIGIN/pred_castp_pockets_cleaned.csv $DATA_ORIGIN/pred_cavityspace_pockets_cleaned.csv $DATA_ORIGIN/pred_fpocket_pockets_scored_cleaned.csv $DATA_ORIGIN/pred_ligsite_pockets_cleaned.csv $DATA_ORIGIN/pred_pocketminer_pockets_scored_cleaned.csv $DATA_ORIGIN/pred_prank_pockets_scored_cleaned.csv data
cp $DATA_ORIGIN/proteome_hotpocket_comb_NN_score.csv $DATA_ORIGIN/proteome_hotpocket_embs_NN_score.csv data
cp $DATA_ORIGIN/BioLiP.txt data

mkdir data/astex
mkdir data/posebusters
mkdir data/human_biolip

mkdir data/uniprot
cp $DATA_ORIGIN/uniprot/* data/uniprot