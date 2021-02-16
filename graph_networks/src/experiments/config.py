class BaseConfig:
    def to_dict(self):
        return self.__dict__


class Config(BaseConfig):
    def __init__(self, model_id: str, logging: bool = False, num_classes: int = 5):
        self.logging = logging
        self.process_path = None
        self.num_classes = num_classes
        self.model_id = model_id


    def set_process_path(self, path: str):
        self.process_path = path
