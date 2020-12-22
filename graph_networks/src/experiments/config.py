class BaseConfig:
    def to_dict(self):
        return self.__dict__


class Config(BaseConfig):
    def __init__(self, base_path: str, eval_type: str, k_per_class: int, n_iter_eval: int, col_label: str, logging: bool = False):
        self.n_iter_eval = n_iter_eval
        self.logging = logging
        self.base_path = base_path
        self.eval_type = eval_type
        self.k_per_class = k_per_class
        self.col_label = col_label
        self.process_path = None

    def set_process_path(self, path: str):
        self.process_path = path
