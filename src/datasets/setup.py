# python src/datasets/setup.py
import gdown


if __name__ == "__main__":
    url = "https://drive.google.com/file/d/1tZfsYQgWmc2gG340ru5VbrZ5aLIZ41_6"
    output = "data/raw/XLCoST_data.zip"
    gdown.download(url, output, quiet=False, fuzzy=True)
    # PyTorrent - download from Zenodo
