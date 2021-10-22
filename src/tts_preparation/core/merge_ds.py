from collections import Counter, OrderedDict
from copy import deepcopy
from functools import partial
from logging import getLogger
from typing import Dict, List, Optional, Set, Tuple

from speech_dataset_preprocessing import FinalDsEntry, FinalDsEntryList
from text_selection.metrics_export import get_rarity_ngrams
from text_utils.speakers_dict import SpeakersDict
from text_utils.symbol_id_dict import SymbolIdDict
from text_utils.types import Speaker, Symbol
from tts_preparation.core.data import EntryId, PreparedData, PreparedDataList
from tts_preparation.globals import DEFAULT_PADDING_SYMBOL

ALL_SPEAKERS_INDICATOR = "all"

DsName = str


def expand_speakers(speakers_to_ds_names: Dict[DsName, Set[Speaker]], ds_speakers: List[Tuple[DsName, Speaker]]) -> Dict[DsName, Set[Speaker]]:
  expanded_speakers: Dict[DsName, Set[Speaker]] = {
    ds_name: set() for ds_name in speakers_to_ds_names}
  for ds_name, speaker_name in ds_speakers:
    if ds_name not in speakers_to_ds_names:
      continue
    if speaker_name == ALL_SPEAKERS_INDICATOR:
      expanded_speakers[ds_name] |= speakers_to_ds_names[ds_name]
    else:
      if speaker_name not in speakers_to_ds_names[ds_name]:
        continue
      expanded_speakers[ds_name].add(speaker_name)
  return expanded_speakers


def get_speakers_of_final_data(final_ds_list: FinalDsEntryList) -> Set[Speaker]:
  result = {entry.speaker_name for entry in final_ds_list.items()}
  return result


def get_speakers_of_prep_data(final_ds_list: PreparedDataList) -> Set[Speaker]:
  result = {entry.speaker_name for entry in final_ds_list.items()}
  return result


def get_ds_speaker_name(ds_name: DsName, speaker: Speaker) -> Speaker:
  result = f"{speaker} ({ds_name})"
  return result


def merge(datasets: List[Tuple[DsName, FinalDsEntryList]], ds_speakers: List[Tuple[DsName, Speaker]]) -> Tuple[PreparedDataList, SymbolIdDict, SpeakersDict]:
  final_ds = merge_final_datasets(datasets, ds_speakers)
  prep_data_list = map_to_prepared_data(final_ds)

  symbol_id_dict = create_symbol_id_dict(prep_data_list)
  set_symbol_ids(prep_data_list, symbol_id_dict)

  speaker_id_dict = create_speaker_id_dict(prep_data_list)
  set_speaker_ids(prep_data_list, speaker_id_dict)

  set_rarities(prep_data_list)
  log_stats(prep_data_list, symbol_id_dict, speaker_id_dict)
  return prep_data_list, symbol_id_dict, speaker_id_dict


def remove_unwanted_symbols(data: PreparedDataList, allowed_symbols: Set[Symbol]) -> Optional[Tuple[PreparedDataList, SymbolIdDict, SpeakersDict]]:
  result = remove_unwanted_symbols_core(data, allowed_symbols)
  removed_anything = len(result) < len(data)

  if removed_anything:
    symbol_id_dict = create_symbol_id_dict(result)
    set_symbol_ids(result, symbol_id_dict)

    speaker_id_dict = create_speaker_id_dict(result)
    set_speaker_ids(result, speaker_id_dict)

    logger = getLogger(__name__)
    logger.info("Updating rarities...")
    set_rarities(result)
    log_stats(result, symbol_id_dict, speaker_id_dict)
    return result, symbol_id_dict, speaker_id_dict

  return None


def remove_unwanted_symbols_core(data: PreparedDataList, allowed_symbols: Set[Symbol]) -> PreparedDataList:
  logger = getLogger(__name__)
  all_occurring_symbols = {symbol for entry in data.items() for symbol in entry.symbols}
  keep_symbols = all_occurring_symbols & allowed_symbols
  not_keep_symbols = all_occurring_symbols - allowed_symbols
  logger.info(
    f"All occurring symbols: {' '.join(sorted(all_occurring_symbols))}")
  logger.info(
    f"Keep utterances with these symbols: {' '.join(sorted(keep_symbols))}")
  logger.info(
    f"Remove utterances with these symbols: {' '.join(sorted(not_keep_symbols))}")

  result = PreparedDataList()
  for entry in data.items():
    contains_only_allowed_symbols = len(set(entry.symbols) - keep_symbols) == 0
    if contains_only_allowed_symbols:
      result.append(entry)

  if len(result) == len(data):
    logger.info("Nothing to remove!")
  elif len(result) == 0:
    assert len(data) > 0
    logger.info("Removed all utterances!")
  else:
    logger.info(
      f"Removed {len(data) - len(result)} from {len(data)} total entries and got {len(result)} entries ({len(result)/len(data)*100:.2f}%).")

  return result


def merge_final_datasets(datasets: List[Tuple[DsName, FinalDsEntryList]], ds_speakers: List[Tuple[DsName, Speaker]]) -> FinalDsEntryList:
  speakers_to_ds_name = {ds_name: get_speakers_of_final_data(data) for ds_name, data in datasets}
  selected_ds_speakers = expand_speakers(speakers_to_ds_name, ds_speakers)
  res = FinalDsEntryList()
  for ds_name, final_ds_data in datasets:
    for entry in final_ds_data.items():
      take_entry = entry.speaker_name in selected_ds_speakers[ds_name]
      if take_entry:
        copied_entry = deepcopy(entry)
        copied_entry.speaker_name = get_ds_speaker_name(ds_name, entry.speaker_name)
        res.append(copied_entry)
  return res


def map_to_prepared_data(data: FinalDsEntryList) -> PreparedDataList:
  result = PreparedDataList(
     map_final_ds_entry_to_prepared_data_entry(entry, entry_id) for entry_id, entry in enumerate(data.items())
  )
  return result


def map_final_ds_entry_to_prepared_data_entry(entry: FinalDsEntry, entry_id: EntryId) -> PreparedData:
  prep_entry = PreparedData(
    entry_id=entry_id,
    ds_entry_id=entry.entry_id,
    basename=entry.basename,
    mel_absolute_path=entry.mel_absolute_path,
    mel_n_channels=entry.mel_n_channels,
    speaker_gender=entry.speaker_gender,
    speaker_name=entry.speaker_name,
    symbols=entry.symbols,
    symbols_format=entry.symbols_format,
    symbols_language=entry.symbols_language,
    symbols_original=entry.symbols_original,
    symbols_original_format=entry.symbols_original_format,
    wav_absolute_path=entry.wav_absolute_path,
    wav_duration=entry.wav_duration,
    wav_original_absolute_path=entry.wav_original_absolute_path,
    wav_sampling_rate=entry.wav_sampling_rate,
    symbol_ids=None,
    speaker_id=None,
    three_gram_rarity=None,
    two_gram_rarity=None,
    one_gram_rarity=None,
  )
  return prep_entry


def create_speaker_id_dict(data: PreparedDataList) -> SpeakersDict:
  all_speakers = get_speakers_of_prep_data(data)
  result = SpeakersDict.fromlist(list(sorted(all_speakers)))
  return result


def set_speaker_ids(data: PreparedDataList, speaker_id_dict: SpeakersDict) -> None:
  for entry in data.items():
    entry.speaker_id = speaker_id_dict.get_id(entry.speaker_name)


def create_symbol_id_dict(data: PreparedDataList) -> SymbolIdDict:
  all_symbols = {symbol for entry in data.items() for symbol in entry.symbols}
  result = SymbolIdDict.init_from_symbols_with_pad(all_symbols, pad_symbol=DEFAULT_PADDING_SYMBOL)
  return result


def set_symbol_ids(data: PreparedDataList, symbol_id_dict: SymbolIdDict) -> None:
  for entry in data.items():
    entry.symbol_ids = symbol_id_dict.get_ids(entry.symbols)


def set_rarities(data: PreparedDataList) -> None:
  logger = getLogger(__name__)
  logger.info("Calculating rarities...")
  corpus = OrderedDict(
      {entry.entry_id: entry.symbols for entry in data.items()})

  get_rarity_method = partial(
    get_rarity_ngrams,
    data=corpus,
    corpus=corpus,
    ignore_symbols=None,
  )

  logger.info("1-grams...")
  one_gram_rarity = get_rarity_method(n_gram=1)

  logger.info("2-grams...")
  two_gram_rarity = get_rarity_method(n_gram=2)

  logger.info("3-grams...")
  three_gram_rarity = get_rarity_method(n_gram=3)

  for entry in data.items():
    entry.one_gram_rarity = one_gram_rarity[entry.entry_id]
    entry.two_gram_rarity = two_gram_rarity[entry.entry_id]
    entry.three_gram_rarity = three_gram_rarity[entry.entry_id]

  logger.info("Done.")


def log_stats(data: PreparedDataList, symbols: SymbolIdDict, speakers: SpeakersDict) -> None:
  logger = getLogger(__name__)
  logger.info(f"Entries ({len(data)}): {data.total_duration_s/60:.2f}m")
  logger.info(f"Speakers ({len(speakers)}): {', '.join(sorted(speakers.get_all_speakers()))}")
  logger.info(f"Symbols ({len(symbols)}): {' '.join(sorted(symbols.get_all_symbols()))}")
  logger.info(f"Symbol occurrences:")
  symbol_counter = Counter([symbol for entry in data.items() for symbol in entry.symbols])
  logger.info(symbol_counter)
  # log texts and trainsets
