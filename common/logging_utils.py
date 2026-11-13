import json
from pathlib import Path


class JsonlLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(self.path, "a", encoding="utf-8")

    def log(self, record):
        self.handle.write(json.dumps(record, sort_keys=True) + "\n")
        self.handle.flush()

    def close(self):
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
