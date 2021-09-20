from pathlib import Path

from text_utils import SymbolIdDict

INFER_MAP_FN = "inference_map.json"


def get_infer_map_path(merge_dir: Path) -> Path:
  path = merge_dir / INFER_MAP_FN
  return path


def infer_map_exists(merge_dir: Path) -> bool:
  path = merge_dir / INFER_MAP_FN
  return path.is_file()


_merge_symbols_json = "symbols.json"


def get_merged_dir(base_dir: Path, merge_name: str) -> Path:
  return base_dir / merge_name


def load_merged_symbol_converter(merge_dir: Path) -> SymbolIdDict:
  path = merge_dir / _merge_symbols_json
  return SymbolIdDict.load_from_file(path)


def save_merged_symbol_converter(merge_dir: Path, data: SymbolIdDict) -> None:
  path = merge_dir / _merge_symbols_json
  data.save(path)
