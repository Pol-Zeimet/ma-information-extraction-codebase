class BaseConfig:
    def to_dict(self):
        return self.__dict__


class Config(BaseConfig):
    def __init__(self, logging: bool = False, num_classes: int = 5):
        self.logging = logging
        self.process_path = None
        self.num_classes = num_classes

    def set_process_path(self, path: str):
        self.process_path = path
