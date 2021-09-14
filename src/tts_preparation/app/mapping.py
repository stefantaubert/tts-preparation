import os
from logging import getLogger
from pathlib import Path
from typing import Optional

from text_utils import (INFERENCE_ARROW_TYPE, WEIGHTS_ARROW_TYPE, Symbols,
                        SymbolsMap, create_or_update_inference_map,
                        create_or_update_weights_map, print_map, print_symbols)
from tts_preparation.app.inference import get_all_symbols
from tts_preparation.app.io import (get_infer_map_path, get_merged_dir,
                                    infer_map_exists,
                                    load_merged_symbol_converter)

# maybe move it to each inference text dir
INFER_MAP_SYMB_FN = "inference_map.symbols"


def save_infer_map(merge_dir: Path, infer_map: SymbolsMap) -> Path:
  path = get_infer_map_path(merge_dir)
  infer_map.save(path)
  return path


def load_infer_map(merge_dir: Path) -> SymbolsMap:
  return SymbolsMap.load(get_infer_map_path(merge_dir))


def save_weights_map(merge_dir: Path, orig_merge_name: str, weights_map: SymbolsMap) -> Path:
  path = merge_dir / f"{orig_merge_name}.json"
  weights_map.save(path)
  return path


def load_weights_map(merge_dir: Path, orig_merge_name: str) -> SymbolsMap:
  path = merge_dir / f"{orig_merge_name}.json"
  return SymbolsMap.load(path)


def weights_map_exists(merge_dir: Path, orig_merge_name: str) -> bool:
  path = merge_dir / f"{orig_merge_name}.json"
  return path.is_file()


def try_load_symbols_map(symbols_map_path: Path) -> Optional[SymbolsMap]:
  symbols_map = SymbolsMap.load(symbols_map_path) if symbols_map_path else None
  return symbols_map


def get_infer_symbols_path(merge_dir: Path) -> Path:
  path = merge_dir / INFER_MAP_SYMB_FN
  return path


def save_infer_symbols(merge_dir: Path, symbols: Symbols) -> Path:
  path = get_infer_symbols_path(merge_dir)
  save_symbols(path, symbols)
  return path


def save_weights_symbols(merge_dir: Path, weights_merge_name: str, symbols: Symbols) -> Path:
  path = merge_dir / f"{weights_merge_name}.symbols"
  save_symbols(path, symbols)
  return path


def save_symbols(path: Path, symbols: Symbols) -> None:
  with path.open(mode='w', encoding='utf-8') as f:
    f.write('\n'.join([f"\"{symbol}\"" for symbol in symbols]))


def create_or_update_weights_map_main(base_dir: Path, merge_name: str, weights_merge_name: str, template_map: Optional[str] = None) -> None:
  merge_dir = get_merged_dir(base_dir, merge_name)
  assert merge_dir.is_dir()
  orig_prep_dir = get_merged_dir(base_dir, weights_merge_name)
  assert orig_prep_dir.is_dir()

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

  resulting_map_path = save_weights_map(merge_dir, weights_merge_name, weights_map)
  resulting_symbols_path = save_weights_symbols(merge_dir, weights_merge_name, symbols)
  print_map(resulting_map_path, WEIGHTS_ARROW_TYPE)
  print_symbols(resulting_symbols_path)


def create_or_update_inference_map_main(base_dir: Path, merge_name: str, template_map: Optional[Path] = None) -> None:
  logger = getLogger(__name__)
  logger.info("Creating/updating inference map...")
  merge_dir = get_merged_dir(base_dir, merge_name)
  assert merge_dir.is_dir()

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

  infer_map_path = save_infer_map(merge_dir, infer_map)
  infer_symbols_path = save_infer_symbols(merge_dir, symbols)
  print_map(infer_map_path, INFERENCE_ARROW_TYPE)
  print_symbols(infer_symbols_path)
