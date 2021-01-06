from typing import Tuple, Dict, Any, List
import pandas as pd


class Dataset:
    def __init__(self, name: str, path: str, filenames: Dict[str, str]):
        self.name = name
        self.set_split_keys()

        assert list(filenames.keys()) == self._split_keys
        self.filenames = {k: path + v for k, v in filenames.items()}
        self.data = {k: None for k in self._split_keys}
        self._loaded = False
        self._split_keys = []

    def set_split_keys(self) -> None:
        self._split_keys = ["train", "val", "test"]

    def _load(self) -> None:
        for name, filename in zip(self.data.keys(), self.filenames.values()):
            self.data[name] = pd.read_csv(filename)
            print("Loaded {} set: {}".format(name, len(self.data[name])))
        print("")

    def load(self) -> Tuple[pd.DataFrame]:
        if not self._loaded:
            self._load()
            self._loaded = True
        return self.data["train"], self.data["val"], self.data["test"]

    def get_details(self, label_columns: List[str]) -> Dict[str, Any]:
        d = {
            "dataset": self.name
        }

        for split in self._split_keys:
            d[f"info_{split}"] = {label_column: self.get_split_info(split, label_column)
                                  for label_column in label_columns}
            d[f"n_samples_{split}"] = len(self.data[split])

        return d

    def get_split_info(self, split: str, label_column: str) -> Dict[str, Any]:
        vc = self.data[split][label_column].value_counts()
        return {
            "n_classes": self.data[split][label_column].nunique(),
            "n_min_samples": vc.min(),
            "n_max_samples": vc.max()
        }


