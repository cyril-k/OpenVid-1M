from huggingface_hub import snapshot_download

def main():

    snapshot_download(
        repo_id="nkp37/OpenVid-1M",
        repo_type="dataset",
        local_dir="/datasets/nkp37/OpenVid-1M",
        max_workers=128,
    )

if __name__ == "__main__":
    main()