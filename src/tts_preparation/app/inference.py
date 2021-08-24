import os
from logging import getLogger
from pathlib import Path
from typing import List, Optional, Set

from accent_analyser import check_probabilities_are_valid, load_probabilities
from text_utils import EngToIpaMode, Language, SymbolsMap
from text_utils.ipa2symb import IPAExtractionSettings
from tts_preparation.app.io import (get_infer_map_path, infer_map_exists,
                                    load_text_symbol_converter,
                                    save_text_symbol_converter)
from tts_preparation.app.merge_ds import (get_merged_dir,
                                          load_merged_accents_ids,
                                          load_merged_symbol_converter)
from tts_preparation.core.inference import (AccentedSymbol, AccentedSymbolList,
                                            InferSentenceList, Sentence,
                                            SentenceList)
from tts_preparation.core.inference import add_text as infer_add
from tts_preparation.core.inference import (sents_accent_apply,
                                            sents_accent_template,
                                            sents_apply_mapping_table,
                                            sents_convert_to_ipa, sents_map,
                                            sents_normalize, set_accent)
from tts_preparation.globals import DEFAULT_PADDING_SYMBOL
from tts_preparation.utils import get_subdir, get_subfolder_names, read_text


def _get_inference_text_root_dir(merged_dir: str, create: bool = False):
  return get_subdir(merged_dir, 'inference', create)


def get_text_dir(merge_dir: str, text_name: str, create: bool):
  return get_subdir(_get_inference_text_root_dir(merge_dir, create=create), text_name, create)


def get_available_texts(merge_dir: str) -> List[str]:
  root_folder = _get_inference_text_root_dir(merge_dir, create=False)
  all_text_names = get_subfolder_names(root_folder)
  return all_text_names


def get_all_symbols(merge_dir: str) -> Set[str]:
  all_text_names = get_available_texts(merge_dir)
  all_symbols: Set[str] = set()
  for text_name in all_text_names:
    text_dir = get_text_dir(merge_dir, text_name, create=False)
    text_symbol_ids = load_text_symbol_converter(text_dir)
    all_symbols |= text_symbol_ids.get_all_symbols()

  return all_symbols


_text_csv = "text.csv"
_accents_csv = "accents.csv"


def load_text_csv(text_dir: str) -> SentenceList:
  path = os.path.join(text_dir, _text_csv)
  return SentenceList.load(Sentence, path)


def _save_text_csv(text_dir: str, data: SentenceList):
  path = os.path.join(text_dir, _text_csv)
  data.save(path)


def _load_accents_csv(text_dir: str) -> AccentedSymbolList:
  path = os.path.join(text_dir, _accents_csv)
  return AccentedSymbolList.load(AccentedSymbol, path)


def _save_accents_csv(text_dir: str, data: AccentedSymbolList):
  path = os.path.join(text_dir, _accents_csv)
  data.save(path)


def add_text(base_dir: str, merge_name: str, text_name: str, filepath: Optional[str], lang: Language, ignore_arcs: bool, ignore_tones: bool, text: Optional[str] = None):
  assert text_name is not None and text_name != ""
  logger = getLogger(__name__)
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  if not os.path.isdir(merge_dir):
    logger.error("Please prepare data first.")
  else:
    logger.info("Adding text...")
    text_input = ""
    if filepath is None:
      assert text is not None
      text_input = text
    else:
      text_input = read_text(filepath)

    ipa_extraction_settings = IPAExtractionSettings(
      ignore_arcs=ignore_arcs,
      ignore_tones=ignore_tones,
      replace_unknown_ipa_by=DEFAULT_PADDING_SYMBOL,
    )

    symbol_ids, data = infer_add(
      text=text_input,
      lang=lang,
      ipa_settings=ipa_extraction_settings,
      logger=logger,
    )
    print("\n" + data.get_formatted(
      symbol_id_dict=symbol_ids,
      accent_id_dict=load_merged_accents_ids(merge_dir)
    ))
    text_dir = get_text_dir(merge_dir, text_name, create=True)
    _save_text_csv(text_dir, data)
    save_text_symbol_converter(text_dir, symbol_ids)
    _accent_template(base_dir, merge_name, text_name)
    _check_for_unknown_symbols(base_dir, merge_name, text_name)


def normalize_text(base_dir: str, merge_name: str, text_name: str):
  logger = getLogger(__name__)
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  text_dir = get_text_dir(merge_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    logger.error("Please add text first.")
  else:
    logger.info("Normalizing text...")
    symbol_ids, updated_sentences = sents_normalize(
      sentences=load_text_csv(text_dir),
      text_symbols=load_text_symbol_converter(text_dir),
      logger=logger,
    )
    print("\n" + updated_sentences.get_formatted(
      symbol_id_dict=symbol_ids,
      accent_id_dict=load_merged_accents_ids(merge_dir)
    ))
    _save_text_csv(text_dir, updated_sentences)
    save_text_symbol_converter(text_dir, symbol_ids)
    _accent_template(base_dir, merge_name, text_name)
    _check_for_unknown_symbols(base_dir, merge_name, text_name)


def ipa_convert_text(base_dir: str, merge_name: str, text_name: str, ignore_tones: bool = False, ignore_arcs: bool = True, consider_ipa_annotations: bool = False, mode: Optional[EngToIpaMode] = None):
  logger = getLogger(__name__)
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  text_dir = get_text_dir(merge_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    logger.error("Please add text first.")
  else:
    logger.info("Converting text to IPA...")
    symbol_ids, updated_sentences = sents_convert_to_ipa(
      sentences=load_text_csv(text_dir),
      text_symbols=load_text_symbol_converter(text_dir),
      ignore_tones=ignore_tones,
      ignore_arcs=ignore_arcs,
      mode=mode,
      consider_ipa_annotations=consider_ipa_annotations,
      logger=logger,
    )
    print("\n" + updated_sentences.get_formatted(
      symbol_id_dict=symbol_ids,
      accent_id_dict=load_merged_accents_ids(merge_dir)
    ))
    _save_text_csv(text_dir, updated_sentences)
    save_text_symbol_converter(text_dir, symbol_ids)
    _accent_template(base_dir, merge_name, text_name)
    _check_for_unknown_symbols(base_dir, merge_name, text_name)


def apply_mapping_table(base_dir: str, merge_name: str, text_name: str, mapping_table_path: Path, seed: int):
  logger = getLogger(__name__)
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  text_dir = get_text_dir(merge_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    logger.error("Please add text first.")
  else:
    logger.info("Converting text to IPA...")
    mapping_table = load_probabilities(mapping_table_path)
    # mapping_table_is_valid = check_probabilities_are_valid(mapping_table)
    # if not mapping_table_is_valid:
    #   logger.error("Mapping table is not valid!")
    #   return

    symbol_ids, updated_sentences = sents_apply_mapping_table(
      sentences=load_text_csv(text_dir),
      text_symbols=load_text_symbol_converter(text_dir),
      mapping_table=mapping_table,
      seed=seed,
    )

    print("\n" + updated_sentences.get_formatted(
      symbol_id_dict=symbol_ids,
      accent_id_dict=load_merged_accents_ids(merge_dir)
    ))
    _save_text_csv(text_dir, updated_sentences)
    save_text_symbol_converter(text_dir, symbol_ids)
    _accent_template(base_dir, merge_name, text_name)
    _check_for_unknown_symbols(base_dir, merge_name, text_name)


def accent_set(base_dir: str, merge_name: str, text_name: str, accent: str):
  logger = getLogger(__name__)
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  text_dir = get_text_dir(merge_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    logger.error("Please add text first.")
  else:
    logger.info(f"Applying accent {accent}...")
    updated_sentences = set_accent(
      sentences=load_text_csv(text_dir),
      accent_ids=load_merged_accents_ids(merge_dir),
      accent=accent
    )
    print("\n" + updated_sentences.get_formatted(
      symbol_id_dict=load_text_symbol_converter(text_dir),
      accent_id_dict=load_merged_accents_ids(merge_dir)
    ))
    _save_text_csv(text_dir, updated_sentences)
    _accent_template(base_dir, merge_name, text_name)
    _check_for_unknown_symbols(base_dir, merge_name, text_name)


def accent_apply(base_dir: str, merge_name: str, text_name: str):
  logger = getLogger(__name__)
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  text_dir = get_text_dir(merge_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    logger.error("Please add text first.")
  else:
    logger.info("Applying accents...")
    updated_sentences = sents_accent_apply(
      sentences=load_text_csv(text_dir),
      accented_symbols=_load_accents_csv(text_dir),
      accent_ids=load_merged_accents_ids(merge_dir),
    )
    print("\n" + updated_sentences.get_formatted(
      symbol_id_dict=load_text_symbol_converter(text_dir),
      accent_id_dict=load_merged_accents_ids(merge_dir)
    ))
    _save_text_csv(text_dir, updated_sentences)
    _check_for_unknown_symbols(base_dir, merge_name, text_name)


def map_text(base_dir: str, merge_name: str, text_name: str, symbols_map_path: str, ignore_arcs: bool = True):
  logger = getLogger(__name__)
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  text_dir = get_text_dir(merge_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    logger.error("Please add text first.")
  else:
    symbol_ids, updated_sentences = sents_map(
      sentences=load_text_csv(text_dir),
      text_symbols=load_text_symbol_converter(text_dir),
      symbols_map=SymbolsMap.load(symbols_map_path),
      ignore_arcs=ignore_arcs,
      logger=logger,
    )

    print("\n" + updated_sentences.get_formatted(
      symbol_id_dict=symbol_ids,
      accent_id_dict=load_merged_accents_ids(merge_dir)
    ))
    _save_text_csv(text_dir, updated_sentences)
    save_text_symbol_converter(text_dir, symbol_ids)
    _accent_template(base_dir, merge_name, text_name)
    _check_for_unknown_symbols(base_dir, merge_name, text_name)


def map_to_prep_symbols(base_dir: str, merge_name: str, text_name: str, ignore_arcs: bool = True):
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  assert os.path.isdir(merge_dir)
  assert infer_map_exists(merge_dir)

  symb_map_path = get_infer_map_path(merge_dir)
  map_text(base_dir, merge_name, text_name, symb_map_path, ignore_arcs)


def _accent_template(base_dir: str, merge_name: str, text_name: str):
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  text_dir = get_text_dir(merge_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    print("Please add text first.")
  else:
    print("Updating accent template...")
    accented_symbol_list = sents_accent_template(
      sentences=load_text_csv(text_dir),
      text_symbols=load_text_symbol_converter(text_dir),
      accent_ids=load_merged_accents_ids(merge_dir),
    )
    _save_accents_csv(text_dir, accented_symbol_list)


def get_infer_sentences(base_dir: str, merge_name: str, text_name: str) -> InferSentenceList:
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  text_dir = get_text_dir(merge_dir, text_name, create=False)
  if not os.path.isdir(text_dir):
    print(f"The text '{text_name}' doesn't exist.")
    assert False
  result = InferSentenceList.from_sentences(
    sentences=load_text_csv(text_dir),
    accents=load_merged_accents_ids(merge_dir),
    symbols=load_text_symbol_converter(text_dir)
  )

  return result


def _check_for_unknown_symbols(base_dir: str, merge_name: str, text_name: str):
  infer_sents = get_infer_sentences(
    base_dir, merge_name, text_name)

  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  logger = getLogger(__name__)
  unknown_symbols_exist = infer_sents.replace_unknown_symbols(
    model_symbols=load_merged_symbol_converter(merge_dir),
    logger=logger
  )

  if unknown_symbols_exist:
    logger.info(
      "Some symbols are not in the prepared dataset symbolset. You need to create an inference map and then apply it to the symbols.")
  else:
    logger.info("All symbols are in the prepared dataset symbolset. You can now synthesize this text.")
