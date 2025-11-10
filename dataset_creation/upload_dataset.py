from pathlib import Path

from huggingface_hub import create_repo, upload_folder

REPO_ID = "StephanAkkerman/chart-info-yolo"
LOCAL_DATASET = Path("local_datasets")


def main() -> None:
    # Create or reuse dataset repo
    create_repo(
        REPO_ID,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )

    # Upload entire YOLO folder
    upload_folder(
        repo_id=REPO_ID,
        repo_type="dataset",
        folder_path=str(LOCAL_DATASET),
        path_in_repo=".",  # put content at repo root
        ignore_patterns=["*.cache", "**/__pycache__/**"],
    )

    # Could also use HfApi().upload_large_folder(...)

    print(f"Uploaded YOLO dataset layout to https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
