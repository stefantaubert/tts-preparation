import os

from text_utils import SymbolIdDict
from tts_preparation.utils import get_subdir

_symbols_json = "symbols.json"
INFER_MAP_FN = "inference_map.json"


def get_pre_dir(base_dir: str, create: bool = False):
  return get_subdir(base_dir, 'pre', create)


def load_text_symbol_converter(text_dir: str) -> SymbolIdDict:
  path = os.path.join(text_dir, _symbols_json)
  return SymbolIdDict.load_from_file(path)


def save_text_symbol_converter(text_dir: str, data: SymbolIdDict):
  path = os.path.join(text_dir, _symbols_json)
  data.save(path)


def get_infer_map_path(merge_dir: str) -> str:
  path = os.path.join(merge_dir, INFER_MAP_FN)
  return path


def infer_map_exists(merge_dir: str) -> bool:
  path = os.path.join(merge_dir, INFER_MAP_FN)
  return os.path.isfile(path)
