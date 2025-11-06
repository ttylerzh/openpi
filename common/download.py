import os
os.environ["HF_LEROBOT_HOME"] = os.path.expanduser("~/zzh/openpi/data")

def openpi_download():
    from openpi.shared import download
    download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")

def huggingface_download():
    from huggingface_hub import snapshot_download

    # snapshot_download(repo_id="openvla/modified_libero_rlds", repo_type="dataset",
    #                 local_dir="data/",allow_patterns="libero_10_no_noops/**")
    snapshot_download(repo_id="lerobot/pi05_base", repo_type="model",
                      local_dir="/home/dell/zzh/SRT/data/model",
                      endpoint="https://hf-mirror.com")
    
if __name__ == "__main__":
    # huggingface_download()
    openpi_download()