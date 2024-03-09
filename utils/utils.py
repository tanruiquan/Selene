from pathlib import Path


def read_file(file_path):
    return Path(file_path).read_text()
