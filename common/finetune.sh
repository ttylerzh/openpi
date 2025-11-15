export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export HF_LEROBOT_HOME=~/zzh/openpi/data

# uv run scripts/compute_norm_stats.py --config-name pi05_surgery_config1

uv run scripts/train.py pi05_surgery_config1 --exp-name=1116 --overwrite