from text_utils import StringFormat2
from pronunciation_dict_parser import (
    PublicDictType, )
from logging import getLogger
from pathlib import Path
from typing import List, Optional, Set

from accent_analyser import load_probabilities
from general_utils import get_subfolder_names, load_obj, save_obj
from text_utils import EngToIPAMode, Language, Symbol, SymbolFormat, SymbolsMap
from tts_preparation.app.io import (get_infer_map_path, get_merged_dir,
                                    infer_map_exists)
from tts_preparation.app.merge_ds import load_merged_symbol_converter
from tts_preparation.core.inference import (InferableUtterances,
                                            add_utterances_from_text,
                                            get_utterances_txt, log_utterances,
                                            utterances_apply_mapping_table,
                                            utterances_apply_symbols_map,
                                            utterances_change_ipa,
                                            utterances_change_text,
                                            utterances_convert_to_ipa, utterances_convert_to_string,
                                            utterances_normalize,
                                            utterances_split)
from tts_preparation.core.inference_transcribe_arpa import transcribe_to_arpa
from tts_preparation.globals import NOT_INFERABLE_SYMBOL_MARKER

UTTERANCES_PKL = "utterances.pkl"
UTTERANCES_TXT = "utterances.txt"


def __get_inference_text_root_dir(merged_dir: Path) -> Path:
  return merged_dir / 'inference'


def get_text_dir(merge_dir: Path, text_name: str) -> Path:
  return __get_inference_text_root_dir(merge_dir) / text_name


def get_available_texts(merge_dir: Path) -> List[str]:
  root_folder = __get_inference_text_root_dir(merge_dir)
  if not root_folder.exists():
    return []
  all_text_names = get_subfolder_names(root_folder)
  return all_text_names


def get_all_symbols(merge_dir: Path) -> Set[Symbol]:
  all_text_names = get_available_texts(merge_dir)
  all_symbols: Set[Symbol] = set()
  for text_name in all_text_names:
    text_dir = get_text_dir(merge_dir, text_name)
    utterances = load_utterances(text_dir)
    utterances_symbols = {symbol for utterance in utterances.items()
                          for symbol in utterance.symbols}
    all_symbols |= utterances_symbols

  return all_symbols


def load_utterances(text_dir: Path) -> InferableUtterances:
  path = text_dir / UTTERANCES_PKL
  return load_obj(path)


def __save_utterances(text_dir: Path, utterances: InferableUtterances) -> None:
  path = text_dir / UTTERANCES_PKL
  save_obj(utterances, path)


def __save_utterances_txt(text_dir: Path, utterances: InferableUtterances) -> None:
  path = text_dir / UTTERANCES_TXT
  txt = get_utterances_txt(utterances, marker=NOT_INFERABLE_SYMBOL_MARKER)
  path.write_text(txt)


def add_text(base_dir: Path, merge_name: str, text_name: str, text_filepath: Optional[Path], language: Language, text: Optional[str], text_format: SymbolFormat, string_format: StringFormat2) -> None:
  assert text_name is not None and text_name != ""
  logger = getLogger(__name__)
  merge_dir = get_merged_dir(base_dir, merge_name)
  if not merge_dir.is_dir():
    logger.error("Please prepare data first.")
    return

  logger.info("Adding text...")
  text_input = ""
  if text_filepath is None and text is None:
    logger.error("Text and Text path could not be both empty!")
    return
  if text_filepath is not None:
    text_input = text_filepath.read_text()
  else:
    text_input = text

  utterances = add_utterances_from_text(
    text=text_input,
    language=language,
    text_format=text_format,
    string_format=string_format,
    symbol_id_dict=load_merged_symbol_converter(merge_dir),
  )

  text_dir = get_text_dir(merge_dir, text_name)
  text_dir.mkdir(parents=True, exist_ok=True)
  __save_utterances(text_dir, utterances)
  __save_utterances_txt(text_dir, utterances)

  log_utterances(utterances, marker=NOT_INFERABLE_SYMBOL_MARKER)
  __check_for_unknown_symbols(utterances)


def split_text(base_dir: Path, merge_name: str, text_name: str) -> None:
  logger = getLogger(__name__)
  merge_dir = get_merged_dir(base_dir, merge_name)
  text_dir = get_text_dir(merge_dir, text_name)
  if not text_dir.is_dir():
    logger.error("Please add text first.")
    return

  logger.info("Splitting text...")
  utterances = utterances_split(
    utterances=load_utterances(text_dir),
    symbol_id_dict=load_merged_symbol_converter(merge_dir),
  )

  text_dir = get_text_dir(merge_dir, text_name)
  text_dir.mkdir(parents=True, exist_ok=True)
  __save_utterances(text_dir, utterances)
  __save_utterances_txt(text_dir, utterances)

  log_utterances(utterances, marker=NOT_INFERABLE_SYMBOL_MARKER)
  __check_for_unknown_symbols(utterances)


def normalize_text(base_dir: Path, merge_name: str, text_name: str) -> None:
  logger = getLogger(__name__)
  merge_dir = get_merged_dir(base_dir, merge_name)
  text_dir = get_text_dir(merge_dir, text_name)
  if not text_dir.is_dir():
    logger.error("Please add text first.")
    return

  logger.info("Normalizing text...")
  utterances = load_utterances(text_dir)
  utterances_normalize(
    utterances=utterances,
    symbol_id_dict=load_merged_symbol_converter(merge_dir),
  )

  text_dir = get_text_dir(merge_dir, text_name)
  text_dir.mkdir(parents=True, exist_ok=True)
  __save_utterances(text_dir, utterances)
  __save_utterances_txt(text_dir, utterances)

  log_utterances(utterances, marker=NOT_INFERABLE_SYMBOL_MARKER)
  __check_for_unknown_symbols(utterances)


def ipa_convert_text(base_dir: Path, merge_name: str, text_name: str, consider_annotations: bool = False, mode: Optional[EngToIPAMode] = None) -> None:
  logger = getLogger(__name__)
  merge_dir = get_merged_dir(base_dir, merge_name)
  text_dir = get_text_dir(merge_dir, text_name)
  if not text_dir.is_dir():
    logger.error("Please add text first.")
    return

  logger.info("Converting text to IPA...")
  utterances = load_utterances(text_dir)
  utterances_convert_to_ipa(
    utterances=utterances,
    symbol_id_dict=load_merged_symbol_converter(merge_dir),
    consider_annotations=consider_annotations,
    mode=mode,
  )

  text_dir = get_text_dir(merge_dir, text_name)
  text_dir.mkdir(parents=True, exist_ok=True)
  __save_utterances(text_dir, utterances)
  __save_utterances_txt(text_dir, utterances)

  log_utterances(utterances, marker=NOT_INFERABLE_SYMBOL_MARKER)
  __check_for_unknown_symbols(utterances)


def arpa_convert_text(base_dir: Path, merge_name: str, text_name: str, consider_annotations: bool = False, dictionary: PublicDictType = PublicDictType.MFA_ARPA) -> None:
  logger = getLogger(__name__)
  merge_dir = get_merged_dir(base_dir, merge_name)
  text_dir = get_text_dir(merge_dir, text_name)
  if not text_dir.is_dir():
    logger.error("Please add text first.")
    return

  logger.info("Converting text to ARPA...")
  utterances = load_utterances(text_dir)
  transcribe_to_arpa(
    utterances=utterances,
    symbol_id_dict=load_merged_symbol_converter(merge_dir),
    consider_annotations=consider_annotations,
    dictionary=dictionary,
  )

  text_dir = get_text_dir(merge_dir, text_name)
  text_dir.mkdir(parents=True, exist_ok=True)
  __save_utterances(text_dir, utterances)
  __save_utterances_txt(text_dir, utterances)

  log_utterances(utterances, marker=NOT_INFERABLE_SYMBOL_MARKER)
  __check_for_unknown_symbols(utterances)


def change_ipa_text(base_dir: Path, merge_name: str, text_name: str, ignore_tones: bool, ignore_arcs: bool, ignore_stress: bool, break_n_thongs: bool, build_n_thongs: bool) -> None:
  logger = getLogger(__name__)
  merge_dir = get_merged_dir(base_dir, merge_name)
  text_dir = get_text_dir(merge_dir, text_name)
  if not text_dir.is_dir():
    logger.error("Please add text first.")
    return

  logger.info("Changing IPA...")
  utterances = load_utterances(text_dir)
  utterances_change_ipa(
    utterances=utterances,
    symbol_id_dict=load_merged_symbol_converter(merge_dir),
    ignore_arcs=ignore_arcs,
    ignore_stress=ignore_stress,
    ignore_tones=ignore_tones,
    break_n_thongs=break_n_thongs,
    build_n_thongs=build_n_thongs,
  )

  text_dir = get_text_dir(merge_dir, text_name)
  text_dir.mkdir(parents=True, exist_ok=True)
  __save_utterances(text_dir, utterances)
  __save_utterances_txt(text_dir, utterances)

  log_utterances(utterances, marker=NOT_INFERABLE_SYMBOL_MARKER)
  __check_for_unknown_symbols(utterances)


def export_text(base_dir: Path, merge_name: str, text_name: str, string_format: StringFormat2) -> None:
  logger = getLogger(__name__)
  merge_dir = get_merged_dir(base_dir, merge_name)
  text_dir = get_text_dir(merge_dir, text_name)
  if not text_dir.is_dir():
    logger.error("Please add text first.")
    return

  logger.info("Changing IPA...")
  utterances = load_utterances(text_dir)

  text = utterances_convert_to_string(utterances, string_format)

  text_dir = get_text_dir(merge_dir, text_name)
  text_dir.mkdir(parents=True, exist_ok=True)
  out_file = text_dir / "export.txt"
  out_file.write_text(text, encoding="UTF-8")
  logger.info(f"Exported to: {out_file.absolute()}")


def change_text(base_dir: Path, merge_name: str, text_name: str, remove_space_around_punctuation: bool) -> None:
  logger = getLogger(__name__)
  merge_dir = get_merged_dir(base_dir, merge_name)
  text_dir = get_text_dir(merge_dir, text_name)
  if not text_dir.is_dir():
    logger.error("Please add text first.")
    return

  logger.info("Changing content...")
  utterances = load_utterances(text_dir)
  utterances_change_text(
    utterances=utterances,
    symbol_id_dict=load_merged_symbol_converter(merge_dir),
    remove_space_around_punctuation=remove_space_around_punctuation,
  )

  text_dir = get_text_dir(merge_dir, text_name)
  text_dir.mkdir(parents=True, exist_ok=True)
  __save_utterances(text_dir, utterances)
  __save_utterances_txt(text_dir, utterances)

  log_utterances(utterances, marker=NOT_INFERABLE_SYMBOL_MARKER)
  __check_for_unknown_symbols(utterances)


def apply_mapping_table(base_dir: Path, merge_name: str, text_name: str, mapping_table_path: Path, seed: int) -> None:
  logger = getLogger(__name__)
  merge_dir = get_merged_dir(base_dir, merge_name)
  text_dir = get_text_dir(merge_dir, text_name)
  if not text_dir.is_dir():
    logger.error("Please add text first.")
    return

  logger.info("Applying mapping table...")
  probabilities_dict = load_probabilities(mapping_table_path)
  # mapping_table_is_valid = check_probabilities_are_valid(probabilities_dict)
  # if not mapping_table_is_valid:
  #   logger.error("Mapping table is not valid!")
  #   return

  utterances = load_utterances(text_dir)
  utterances_apply_mapping_table(
    utterances=utterances,
    symbol_id_dict=load_merged_symbol_converter(merge_dir),
    probabilities_dict=probabilities_dict,
    seed=seed,
  )

  text_dir = get_text_dir(merge_dir, text_name)
  text_dir.mkdir(parents=True, exist_ok=True)
  __save_utterances(text_dir, utterances)
  __save_utterances_txt(text_dir, utterances)

  log_utterances(utterances, marker=NOT_INFERABLE_SYMBOL_MARKER)
  __check_for_unknown_symbols(utterances)


def map_text(base_dir: Path, merge_name: str, text_name: str, symbols_map_path: Optional[Path] = None) -> None:
  logger = getLogger(__name__)
  merge_dir = get_merged_dir(base_dir, merge_name)
  text_dir = get_text_dir(merge_dir, text_name)
  if not text_dir.is_dir():
    logger.error("Please add text first.")
    return

  logger.info("Applying symbols map...")
  if symbols_map_path is None:
    if not infer_map_exists(merge_dir):
      logger.error("Inference map is required if no symbols_map_path is provided!")
      return
    symbols_map_path = get_infer_map_path(merge_dir)

  utterances = load_utterances(text_dir)
  utterances_apply_symbols_map(
    utterances=utterances,
    symbol_id_dict=load_merged_symbol_converter(merge_dir),
    symbols_map=SymbolsMap.load(symbols_map_path),
  )

  text_dir = get_text_dir(merge_dir, text_name)
  text_dir.mkdir(parents=True, exist_ok=True)
  __save_utterances(text_dir, utterances)
  __save_utterances_txt(text_dir, utterances)

  log_utterances(utterances, marker=NOT_INFERABLE_SYMBOL_MARKER)
  __check_for_unknown_symbols(utterances)


def __check_for_unknown_symbols(utterances: InferableUtterances) -> None:
  logger = getLogger(__name__)

  all_can_be_synthesized = all(
    utterance.can_all_symbols_be_inferred for utterance in utterances.items())
  if all_can_be_synthesized:
    logger.info("All utterances are fully synthesizable.")
  else:
    logger.warning(
      "Some utterances contain symbols which can not be synthesized! You can create an inference map and then apply it to the symbols.")
