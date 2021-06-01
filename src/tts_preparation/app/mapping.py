import os
from logging import getLogger
from typing import List, Optional

from text_utils import (SymbolsMap, create_or_update_inference_map,
                        create_or_update_weights_map)
from tts_preparation.app.inference import get_all_symbols
from tts_preparation.app.io import get_infer_map_path, infer_map_exists
from tts_preparation.app.merge_ds import (get_merged_dir,
                                          load_merged_symbol_converter)

# maybe move it to each inference text dir
INFER_MAP_SYMB_FN = "inference_map.symbols"


def save_infer_map(merge_dir: str, infer_map: SymbolsMap):
  infer_map.save(get_infer_map_path(merge_dir))


def load_infer_map(merge_dir: str) -> SymbolsMap:
  return SymbolsMap.load(get_infer_map_path(merge_dir))


def save_weights_map(merge_dir: str, orig_merge_name: str, weights_map: SymbolsMap):
  path = os.path.join(merge_dir, f"{orig_merge_name}.json")
  weights_map.save(path)


def load_weights_map(merge_dir: str, orig_merge_name: str) -> SymbolsMap:
  path = os.path.join(merge_dir, f"{orig_merge_name}.json")
  return SymbolsMap.load(path)


def weights_map_exists(merge_dir: str, orig_merge_name: str) -> bool:
  path = os.path.join(merge_dir, f"{orig_merge_name}.json")
  return os.path.isfile(path)


def try_load_symbols_map(symbols_map_path: str) -> Optional[SymbolsMap]:
  symbols_map = SymbolsMap.load(symbols_map_path) if symbols_map_path else None
  return symbols_map


def get_infer_symbols_path(merge_dir: str) -> str:
  path = os.path.join(merge_dir, INFER_MAP_SYMB_FN)
  return path


def save_infer_symbols(merge_dir: str, symbols: List[str]):
  path = get_infer_symbols_path(merge_dir)
  save_symbols(path, symbols)


def save_weights_symbols(merge_dir: str, weights_merge_name: str, symbols: List[str]):
  path = os.path.join(merge_dir, f"{weights_merge_name}.symbols")
  save_symbols(path, symbols)


def save_symbols(path: str, symbols: List[str]):
  with open(path, 'w', encoding='utf-8') as f:
    f.write('\n'.join([f"\"{x}\"" for x in symbols]))


def create_or_update_weights_map_main(base_dir: str, merge_name: str, weights_merge_name: str, template_map: Optional[str] = None):
  merge_dir = get_merged_dir(base_dir, merge_name)
  assert os.path.isdir(merge_dir)
  orig_prep_dir = get_merged_dir(base_dir, weights_merge_name)
  assert os.path.isdir(orig_prep_dir)

  logger = getLogger(__name__)
  logger.info(f"Creating/updating weights map for {weights_merge_name}...")

  if template_map is not None:
    _template_map = SymbolsMap.load(template_map)
  else:
    _template_map = None

  if weights_map_exists(merge_dir, weights_merge_name):
    existing_map = load_weights_map(merge_dir, weights_merge_name)
  else:
    existing_map = None

  weights_map, symbols = create_or_update_weights_map(
    orig=load_merged_symbol_converter(orig_prep_dir).get_all_symbols(),
    dest=load_merged_symbol_converter(merge_dir).get_all_symbols(),
    existing_map=existing_map,
    template_map=_template_map,
  )

  save_weights_map(merge_dir, weights_merge_name, weights_map)
  save_weights_symbols(merge_dir, weights_merge_name, symbols)


def create_or_update_inference_map_main(base_dir: str, merge_name: str, template_map: Optional[str] = None):
  logger = getLogger(__name__)
  logger.info("Creating/updating inference map...")
  merge_dir = get_merged_dir(base_dir, merge_name)
  assert os.path.isdir(merge_dir)

  all_symbols = get_all_symbols(merge_dir)

  if template_map is not None:
    _template_map = SymbolsMap.load(template_map)
  else:
    _template_map = None

  if infer_map_exists(merge_dir):
    existing_map = load_infer_map(merge_dir)
  else:
    existing_map = None

  infer_map, symbols = create_or_update_inference_map(
    orig=load_merged_symbol_converter(merge_dir).get_all_symbols(),
    dest=all_symbols,
    existing_map=existing_map,
    template_map=_template_map,
  )

  save_infer_map(merge_dir, infer_map)
  save_infer_symbols(merge_dir, symbols)
