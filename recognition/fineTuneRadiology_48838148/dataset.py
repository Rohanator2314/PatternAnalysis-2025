from huggingface_hub import login
from dotenv import dotenv_values

from datasets import load_dataset


if __name__ == "__main__":
    config = dotenv_values("recognition/fineTuneRadiology_48838148/.env")
    login(config["HF_TOKEN"])

    ds = load_dataset("BioLaySumm/BioLaySumm2025-LaymanRRG-opensource-track")

    ds.save_to_disk("recognition/fineTuneRadiology_48838148/data/BioLaySumm2025-LaymanRRG-opensource-track")