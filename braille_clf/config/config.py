from dataclasses import dataclass
import yaml
import os


@dataclass
class BrailleParams:
    root_dir: str = os.path.abspath(os.path.join(".", os.pardir))
    name_source_data: str = ""
    dir_bgs: str = ""
    dir_tactile_imgs: str = ""
    dir_tactile_real: str = ""
    remove_bg_train: bool = True
    img_size: int = 64
    batch_size: int = 12
    lr: float = 1e-4
    n_epochs: int = 100
    model: str = "resnet18"
    n_froze_layers: int = 4
    fine_tune: bool = False
    checkpoint: str = ""
    p_data_fine_tune: float = 0.0

    def dump(self, cfg_file):
        with open(cfg_file, "w") as f:
            yaml.dump(self.__dict__, f)

    def load(self, cfg_file):
        with open(cfg_file, "r") as f:
            self.__dict__ = yaml.load(f, Loader=yaml.FullLoader)
