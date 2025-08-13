#!/bin/bash

DATA_ORIGIN=/oak/stanford/groups/rbaltman/kcarp/hotpocket2.0/data

mkdir data
cp $DATA_ORIGIN/proteome_structure_files_canonical.csv data
cp $DATA_ORIGIN/known_pockets.csv data
cp $DATA_ORIGIN/pred_autosite_pockets_cleaned.csv $DATA_ORIGIN/pred_castp_pockets_cleaned.csv $DATA_ORIGIN/pred_cavityspace_pockets_cleaned.csv $DATA_ORIGIN/pred_fpocket_pockets_scored_cleaned.csv $DATA_ORIGIN/pred_ligsite_pockets_cleaned.csv $DATA_ORIGIN/pred_pocketminer_pockets_scored_cleaned.csv $DATA_ORIGIN/pred_prank_pockets_scored_cleaned.csv data
cp $DATA_ORIGIN/pred_cavityspace_pockets_strong.csv data
cp $DATA_ORIGIN/proteome_hotpocket_comb_NN_score.csv $DATA_ORIGIN/proteome_hotpocket_embs_NN_score.csv data
cp $DATA_ORIGIN/BioLiP.txt data
cp $DATA_ORIGIN/biolip_TTD.csv data

mkdir data/astex
mkdir data/posebusters
mkdir data/human_biolip

mkdir data/uniprot
cp $DATA_ORIGIN/uniprot/* data/uniprot

mkdir data/model_checkpoints
cp $DATA_ORIGIN/nn_augmented_esm_only_daily-sweep-128_733.pt $DATA_ORIGIN/nn_augmented_combined_prime-sweep-241_733.pt data/model_checkpoints

mkdir data/model_inputs
cp $DATA_ORIGIN/biolip_ttd_augmented*.npy $DATA_ORIGIN/biolip_ttd_ordered*.npy $DATA_ORIGIN/biolip_ttd_nonzero*.npy data/model_inputs
cp $DATA_ORIGIN/biolip_ttd_train*X.npy $DATA_ORIGIN/biolip_ttd_val*X.npy $DATA_ORIGIN/biolip_ttd_test*X.npy data/model_inputs
cp $DATA_ORIGIN/biolip_ttd_train_Y.npy $DATA_ORIGIN/biolip_ttd_val_Y.npy $DATA_ORIGIN/biolip_ttd_test_Y.npy data/model_inputs