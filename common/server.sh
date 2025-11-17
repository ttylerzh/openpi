export XLA_PYTHON_CLIENT_PREALLOCATE=false

uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_surgery_config1 \
    --policy.dir=checkpoints/pi05_surgery_config1/1116/20000