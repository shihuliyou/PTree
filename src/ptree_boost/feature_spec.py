from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence
import pandas as pd
from .config import ExperimentConfig, CFG


@dataclass(frozen=True)
class FeatureSpec:
    ordered_columns: Tuple[str, ...]
    all_chars: List[str]
    instruments: List[str]
    splitting_chars: List[str]
    first_split_var: List[int]
    second_split_var: List[int]
    rank_cols: List[str]
    macro_cols: List[str]
    strategy: str = "positional"


def _normalize_columns_order(cfg: ExperimentConfig) -> List[str]:
    raw = cfg.data.columns_order
    if raw is None:
        raise ValueError("cfg.data.columns_order 未设置，请明确填写列顺序。")

    if isinstance(raw, str):
        raise ValueError(
            "cfg.data.columns_order 的类型不正确：应为字符串列表/元组，而不是单个字符串。"
        )

    cols: List[str] = []
    for i, c in enumerate(raw):
        if not isinstance(c, str):
            raise ValueError(
                f"cfg.data.columns_order 第 {i} 个元素类型错误：期望 str，"
                f"实际为 {type(c).__name__}，值为 {c!r}"
            )
        s = c.strip()
        if not s:
            raise ValueError(
                f"cfg.data.columns_order 第 {i} 个元素为空字符串或仅空白，无法识别。"
            )
        cols.append(s)

    if not cols:
        raise ValueError("cfg.data.columns_order 为空，请至少提供一个有效列名。")

    seen = set()
    ordered: List[str] = []
    for c in cols:
        if c not in seen:
            ordered.append(c)
            seen.add(c)

    return ordered


def _build_ordered_columns(df: pd.DataFrame, cfg: ExperimentConfig) -> Tuple[str, ...]:
    cols = _normalize_columns_order(cfg)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"columns_order 中存在数据未包含的列: {missing}。"
            f"请检查 config.data.columns_order 或数据字段。"
        )

    return tuple(cols)


def _build_prefix_sets(ordered_columns: Sequence[str], cfg: ExperimentConfig) -> Tuple[List[str], List[str]]:
    rank_prefix = cfg.features.rank_prefix
    macro_prefix = cfg.features.macro_prefix

    rank_cols = [c for c in ordered_columns if isinstance(c, str) and c.startswith(rank_prefix)]
    macro_cols = [c for c in ordered_columns if isinstance(c, str) and c.startswith(macro_prefix)]
    return rank_cols, macro_cols


def _slice_all_chars_by_position(ordered_columns: Sequence[str], cfg: ExperimentConfig) -> List[str]:
    start_r = cfg.features.all_chars_start_pos_r
    end_r = cfg.features.all_chars_end_pos_r
    if start_r <= 0 or end_r <= 0 or end_r < start_r:
        return []

    start_py = max(start_r - 1, 0)
    end_py = min(end_r, len(ordered_columns))

    return list(ordered_columns[start_py:end_py])


def _build_split_vars(splitting_chars: Sequence[str], cfg: ExperimentConfig) -> Tuple[List[int], List[int]]:
    max_feat: Optional[int] = cfg.features.max_splitting_features
    n_total = len(splitting_chars)
    n = n_total if max_feat is None else min(n_total, max_feat)
    idx = list(range(n))
    return idx, idx.copy()


def build_feature_spec(df: pd.DataFrame, cfg: ExperimentConfig = CFG) -> FeatureSpec:
    ordered_columns = _build_ordered_columns(df, cfg)
    rank_cols, macro_cols = _build_prefix_sets(ordered_columns, cfg)

    all_chars = _slice_all_chars_by_position(ordered_columns, cfg)
    strategy = "positional"

    if not all_chars and rank_cols:
        all_chars = rank_cols
        strategy = "prefix_rank_fallback"

    if not all_chars:
        rp = cfg.features.rank_prefix
        mp = cfg.features.macro_prefix
        raise ValueError(
            "无法构造 all_chars。"
            "请检查：\n"
            "1) config.data.columns_order 是否已按 R 逻辑完成列重排；\n"
            "2) config.features.all_chars_start_pos_r/all_chars_end_pos_r 是否合理；\n"
            f"3) 或者数据中是否存在以 {rp!r}（rank_prefix）或 {mp!r}（macro_prefix）开头的特征列。"
        )

    k = cfg.features.instruments_top_k
    instruments = all_chars[: max(k, 0)]
    splitting_chars = list(all_chars)

    first_split_var, second_split_var = _build_split_vars(splitting_chars, cfg)

    return FeatureSpec(
        ordered_columns=tuple(ordered_columns),
        all_chars=all_chars,
        instruments=instruments,
        splitting_chars=splitting_chars,
        first_split_var=first_split_var,
        second_split_var=second_split_var,
        rank_cols=rank_cols,
        macro_cols=macro_cols,
        strategy=strategy,
    )


def apply_column_order(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    cols = [c for c in spec.ordered_columns if c in df.columns]
    return df.loc[:, cols]


def build_instrument_matrix(df: pd.DataFrame, spec: FeatureSpec, cfg: ExperimentConfig = CFG,) -> pd.DataFrame:
    if not spec.instruments:
        z = pd.DataFrame(index=df.index)
    else:
        z = df.loc[:, spec.instruments].copy()

    if cfg.features.add_intercept_to_instruments:
        z.insert(0, "const", 1.0)

    return z


def build_splitting_matrix(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    if not spec.splitting_chars:
        return pd.DataFrame(index=df.index)
    return df.loc[:, spec.splitting_chars].copy()


def get_split_var_arrays(spec: FeatureSpec) -> Tuple[List[int], List[int]]:
    return spec.first_split_var, spec.second_split_var
