
seed=42

# AAEC
python data_process/data_preprocess_aaec.py --dataset_type essay_level --use_mv --seed $seed
python data_process/data_preprocess_aaec.py --dataset_type essay_level --seed $seed
python data_process/data_preprocess_aaec.py --dataset_type paragraph_level --seed $seed
python data_process/data_preprocess_aaec.py --dataset_type paragraph_level --use_mv --seed $seed

python data_process/data_preprocess_aaec.py --dataset_type essay_level --use_mv --seed $seed --use_oracle_span
python data_process/data_preprocess_aaec.py --dataset_type essay_level --seed $seed --use_oracle_span
python data_process/data_preprocess_aaec.py --dataset_type paragraph_level --seed $seed --use_oracle_span
python data_process/data_preprocess_aaec.py --dataset_type paragraph_level --use_mv --seed $seed --use_oracle_span

# CDCP
python data_process/data_preprocess_cdcp.py --dataset_type cdcp --use_mv --seed $seed
python data_process/data_preprocess_cdcp.py --dataset_type cdcp --seed $seed

python data_process/data_preprocess_cdcp.py --dataset_type cdcp --use_mv --seed $seed --use_oracle_span
python data_process/data_preprocess_cdcp.py --dataset_type cdcp --seed $seed --use_oracle_span

# AbstRCT
python data_process/data_preprocess_abstrct.py --dataset_type abstrct --use_mv --seed $seed
python data_process/data_preprocess_abstrct.py --dataset_type abstrct --seed $seed

python data_process/data_preprocess_abstrct.py --dataset_type abstrct --use_mv --seed $seed --use_oracle_span
python data_process/data_preprocess_abstrct.py --dataset_type abstrct --seed $seed --use_oracle_span


# RR
python data_process/data_preprocess_rr.py --dataset_type rr --use_mv --max_sent_len 200
python data_process/data_preprocess_rr.py --dataset_type rr --max_sent_len 200

# QAM
python data_process/data_preprocess_qam.py --dataset_type qam --use_mv
python data_process/data_preprocess_qam.py --dataset_type qam


# MTC
# for fold in {0..49}
for fold in 0
do
    python data_process/data_preprocess_mtc.py --dataset_type mtc --use_mv --fold $fold
    python data_process/data_preprocess_mtc.py --dataset_type mtc --fold $fold

    python data_process/data_preprocess_mtc.py --dataset_type mtc --use_mv --fold $fold --use_oracle_span
    python data_process/data_preprocess_mtc.py --dataset_type mtc --fold $fold --use_oracle_span
done

# AASD
# for fold in {0..49}
for fold in 0
do
    python data_process/data_preprocess_aasd.py --dataset_type aasd --use_mv --fold $fold
    python data_process/data_preprocess_aasd.py --dataset_type aasd --fold $fold

    python data_process/data_preprocess_aasd.py --dataset_type aasd --use_mv --fold $fold --use_oracle_span
    python data_process/data_preprocess_aasd.py --dataset_type aasd --fold $fold --use_oracle_span
done

