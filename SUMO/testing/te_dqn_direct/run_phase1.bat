@echo off
echo Running command: python train_transformer_main.py --energy_csv merged_dag1.csv --energy_weight 0.7 --episodes 200 --max_steps 1000 --duration 300 --output_dir te_dqn_direct\phase1 --seq_length 16 --d_model 128 --nheads 4
python train_transformer_main.py --energy_csv merged_dag1.csv --energy_weight 0.7 --episodes 200 --max_steps 1000 --duration 300 --output_dir te_dqn_direct\phase1 --seq_length 16 --d_model 128 --nheads 4
echo n
echo Batch file execution complete
