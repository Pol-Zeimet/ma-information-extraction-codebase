class BaseConfig:
    def to_dict(self):
        return self.__dict__


class Config(BaseConfig):
    def __init__(self, base_path: str, eval_type: str, n_iter_eval: int,
                 logging: bool = False, num_classes: int = 5, train_test_split = 0.25):
        self.n_iter_eval = n_iter_eval
        self.logging = logging
        self.base_path = base_path
        self.eval_type = eval_type
        self.process_path = None
        self.num_classes = num_classes
        self.train_test_split = train_test_split

    def set_process_path(self, path: str):
        self.process_path = path
