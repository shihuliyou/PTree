from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

PROJECT_ROOT = Path(r"D:\PythonProject\PTree")

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SRC_DIR = PROJECT_ROOT / "src"


@dataclass(frozen=True)
class PathsConfig:
    data_feather: Path = DATA_DIR / " "  # <- 路径
    factor_csv: Path = DATA_DIR / " "  # <- 路径

    outputs_root: Path = OUTPUT_DIR
    models_dir: Path = OUTPUT_DIR / "models"
    artifacts_dir: Path = OUTPUT_DIR / "artifacts"

    def ensure_dirs(self) -> None:
        self.outputs_root.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class DataConfig:
    firm_col: str = " "
    id_col: str = " "
    sic_col: str = " "
    exch_col: str = " "
    shrcd_col: str = " "
    industry_col: str = " "
    date_col: str = " "

    # 收益相关
    ret_col: str = " "
    xret_col: str = " "

    # 权重相关
    lag_me_col: str = " "

    # 时间切分
    start: str = " "
    split: str = " "
    end: str = " "

    # 因子文件列名
    factor_cols: Tuple[str, ...] = ("mktrf", "smb", "hml", "rmw", "cma", "rf", "mom")

    columns_order: Tuple[str, ...] = (
        """
        特征补全
        """
    )


@dataclass(frozen=True)
class FeatureConfig:
    """
    视具体数据而定
    """

    all_chars_start_pos_r: int = 10
    all_chars_end_pos_r: int = 70
    instruments_top_k: int = 5
    add_intercept_to_instruments: bool = True
    rank_prefix: str = "rank_"
    macro_prefix: str = "x_"
    max_splitting_features: Optional[int] = None


@dataclass(frozen=True)
class TreeConfig:
    min_leaf_size: int = 20
    max_depth: int = 10
    num_iter: int = 9
    num_cutpoints: int = 4
    equal_weight: bool = False
    abs_normalize: bool = True
    weighted_loss: bool = False

    use_all_features_for_first_second_split: bool = True


@dataclass(frozen=True)
class BoostingConfig:
    max_depth_boosting: int = 10
    num_iterB: int = 9
    n_boost_steps: int = 20
    eta: float = 1.0
    random_split: bool = False


@dataclass(frozen=True)
class BenchmarkConfig:
    no_H1: bool = True
    no_H: bool = False
    first_benchmark_factor: str = "mktrf"


@dataclass(frozen=True)
class LossConfig:
    early_stop: bool = False
    stop_threshold: float = 1.0
    lambda_ridge: float = 0.0
    lambda_mean: float = 0.0
    lambda_cov: float = 1e-4
    lambda_mean_factor: float = 0.0
    lambda_cov_factor: float = 1e-5


@dataclass(frozen=True)
class ExperimentConfig:
    case: str = " "  # <- 进行命名
    paths: PathsConfig = PathsConfig()
    data: DataConfig = DataConfig()
    features: FeatureConfig = FeatureConfig()
    tree: TreeConfig = TreeConfig()
    boosting: BoostingConfig = BoostingConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()
    loss: LossConfig = LossConfig()

    def to_dict(self) -> Dict[str, Any]:
        """便于保存配置快照，保证可复现。"""
        return asdict(self)


CFG = ExperimentConfig()
