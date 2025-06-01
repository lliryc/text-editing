#!/bin/bash

printf "QALB-2014 Dev: Seq2Seq++ Vs. Sweet^2_NoPnx+Sweet_Pnx\n"
python significance.py \
    --system1_scores /scratch/ba63/backup/gec/models/gec/qalb14/full/t5_w_camelira_ged_pred_worst/checkpoint-33000/qalb14_dev.preds.txt.pp.m2.scores \
    --system2_scores /scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers_arabertv02/qalb14-nopnx-prune-30-a100/checkpoint-1500/dev.txt.2.pnx_edit.pp.m2.scores


printf "QALB-2014 Dev: Seq2Seq++ Vs. 4-Ensemble\n"
python significance.py \
    --system1_scores /scratch/ba63/backup/gec/models/gec/qalb14/full/t5_w_camelira_ged_pred_worst/checkpoint-33000/qalb14_dev.preds.txt.pp.m2.scores \
    --system2_scores /scratch/ba63/arabic-text-editing/ensemble/gec/qalb14/dev/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt.pp.m2.scores


printf "QALB-2014 Dev: Sweet^2_NoPnx+Sweet_Pnx Vs. 4-Ensemble\n"
python significance.py \
    --system1_scores /scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers_arabertv02/qalb14-nopnx-prune-30-a100/checkpoint-1500/dev.txt.2.pnx_edit.pp.m2.scores \
    --system2_scores /scratch/ba63/arabic-text-editing/ensemble/gec/qalb14/dev/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt.pp.m2.scores

printf "QALB-2014 Test: Seq2Seq++ Vs. Sweet^2_NoPnx+Sweet_Pnx\n"
python significance.py \
    --system1_scores /scratch/ba63/backup/gec/models/gec/qalb14/full/bart_w_ged_pred_worst/checkpoint-3000/qalb14_test.preds.txt.m2.scores \
    --system2_scores /scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers_arabertv02/qalb14-nopnx-prune-30-a100/checkpoint-1500/test.txt.2.pnx_edit.m2.scores


printf "QALB-2014 Test: Seq2Seq++ Vs. 4-Ensemble\n"
python significance.py \
    --system1_scores /scratch/ba63/backup/gec/models/gec/qalb14/full/bart_w_ged_pred_worst/checkpoint-3000/qalb14_test.preds.txt.m2.scores \
    --system2_scores /scratch/ba63/arabic-text-editing/ensemble/gec/qalb14/test/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt.m2.scores

printf "QALB-2014 Test: Sweet^2_NoPnx+Sweet_Pnx Vs. 4-Ensemble\n"
python significance.py \
    --system1_scores /scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers_arabertv02/qalb14-nopnx-prune-30-a100/checkpoint-1500/test.txt.2.pnx_edit.m2.scores \
    --system2_scores /scratch/ba63/arabic-text-editing/ensemble/gec/qalb14/test/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt.m2.scores


printf "QALB-2015-L1 Test: Seq2Seq++ Vs. 4-Ensemble\n"
python significance.py \
    --system1_scores /scratch/ba63/backup/gec/models/gec/qalb14-15/full/bart_w_camelira_ged_pred_worst/checkpoint-7000/qalb15_test-L1.preds.check.txt.m2.scores \
    --system2_scores /scratch/ba63/arabic-text-editing/ensemble/gec/qalb14/test_qalb15_L1/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt.m2.scores


printf "QALB-2015-L1 Test: Sweet^2_NoPnx+Sweet_Pnx Vs. 4-Ensemble\n"
python significance.py \
    --system1_scores /scratch/ba63/arabic-text-editing/gec_taggers/qalb14/pnx_taggers_arabertv02/qalb14-nopnx-prune-30-a100/checkpoint-1500/test_qalb15_L1.txt.2.pnx_edit.m2.scores \
    --system2_scores /scratch/ba63/arabic-text-editing/ensemble/gec/qalb14/test_qalb15_L1/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt.m2.scores


printf "ZAEBUC Dev: Seq2Seq++ Vs. 4-Ensemble\n"
python significance.py \
    --system1_scores /scratch/ba63/backup/gec/models/gec/mix/coarse/bart_w_camelira_ged_pred_worst/checkpoint-11000/zaebuc_dev.preds.check.txt.m2.scores \
    --system2_scores /scratch/ba63/arabic-text-editing/ensemble/gec/zaebuc/dev/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt.m2.scores


printf "ZAEBUC Test: Seq2Seq++ Vs. 4-Ensemble\n"
python significance.py \
    --system1_scores /scratch/ba63/backup/gec/models/gec/mix/coarse/bart_w_ged_pred_worst/checkpoint-9000/zaebuc_test.preds.check.txt.m2.scores \
    --system2_scores /scratch/ba63/arabic-text-editing/ensemble/gec/zaebuc/test/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt.m2.scores


printf "ZAEBUC Test: Sweet^2_NoPnx+Sweet_Pnx Vs. 4-Ensemble\n"
python significance.py \
    --system1_scores /scratch/ba63/arabic-text-editing/gec_taggers/qalb14+zaebuc_x10/pnx_taggers_arabertv02_15_iter/qalb14+zaebuc_x10-nopnx-prune-30-a100/checkpoint-7000/test.txt.2.pnx_edit.m2.scores \
    --system2_scores /scratch/ba63/arabic-text-editing/ensemble/gec/zaebuc/test/ensemble.seq2seq+seq2edit+seq2edit-nopnx-pnx+gpt4o.txt.m2.scores



printf "CODA Dev: Seq2Seq++ Vs. Sweet\n"
python significance.py \
    --system1_scores /scratch/ba63/codafication/models/t5/city/city_pred.dev.gen.txt.m2.scores \
    --system2_scores /scratch/ba63/arabic-text-editing/coda_taggers/madar/taggers_arabertv02/madar-prune-10-a100/checkpoint-1500/dev.txt.1.m2.scores


printf "CODA Dev: Seq2Seq++ Vs. 4-Ensemble\n"
python significance.py \
    --system1_scores /scratch/ba63/codafication/models/t5/city/city_pred.dev.gen.txt.m2.scores \
    --system2_scores /scratch/ba63/arabic-text-editing/ensemble/coda/madar/dev/ensemble.seq2seq+seq2edit-camelbert+seq2edit-arabert+gpt4o.txt.m2.scores


printf "CODA Dev: Sweet^2_NoPnx+Sweet_Pnx Vs. 4-Ensemble\n"
python significance.py \
    --system1_scores /scratch/ba63/arabic-text-editing/coda_taggers/madar/taggers_arabertv02/madar-prune-10-a100/checkpoint-1500/dev.txt.1.m2.scores \
    --system2_scores /scratch/ba63/arabic-text-editing/ensemble/coda/madar/dev/ensemble.seq2seq+seq2edit-camelbert+seq2edit-arabert+gpt4o.txt.m2.scores


printf "CODA Test: Seq2Seq++ Vs. Sweet\n"
python significance.py \
    --system1_scores /scratch/ba63/codafication/models/t5/da_phrase/da_phrase_pred.test.gen.txt.m2.scores \
    --system2_scores /scratch/ba63/arabic-text-editing/coda_taggers/madar/taggers_arabertv02/madar-prune-10-a100/checkpoint-1500/test.txt.m2.scores


printf "CODA Test: Seq2Seq++ Vs. 3-Ensemble\n"
python significance.py \
    --system1_scores /scratch/ba63/codafication/models/t5/da_phrase/da_phrase_pred.test.gen.txt.m2.scores \
    --system2_scores /scratch/ba63/arabic-text-editing/ensemble/coda/madar/test/ensemble.seq2seq+seq2edit-camelbert+seq2edit-arabert.txt.m2.scores


printf "CODA Test: Sweet^2_NoPnx+Sweet_Pnx Vs. 3-Ensemble\n"
python significance.py \
    --system1_scores /scratch/ba63/arabic-text-editing/coda_taggers/madar/taggers_arabertv02/madar-prune-10-a100/checkpoint-1500/test.txt.m2.scores \
    --system2_scores /scratch/ba63/arabic-text-editing/ensemble/coda/madar/test/ensemble.seq2seq+seq2edit-camelbert+seq2edit-arabert.txt.m2.scores