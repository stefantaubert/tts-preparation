from collections import OrderedDict
from logging import Logger, getLogger
from shutil import Error
from typing import Callable, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple

import pandas as pd
from ordered_set import OrderedSet
from text_utils import SymbolIdDict
from text_utils.text_selection import (cover_symbols_default,
                                       get_rarity_ngrams,
                                       greedy_kld_uniform_ngrams_seconds,
                                       greedy_ngrams_cover,
                                       greedy_ngrams_epochs,
                                       greedy_ngrams_seconds,
                                       random_ngrams_cover_seconds,
                                       random_percent, random_seconds,
                                       random_seconds_divergence_seeds)
from tts_preparation.core.data import (DatasetType, PreparedDataList,
                                       get_speaker_wise)
from tts_preparation.core.helper import (
    prep_data_list_to_dict_with_durations_s,
    prep_data_list_to_dict_with_symbols, select_enties_from_prep_data)
from tts_preparation.core.merge_ds import MergedDataset
from tts_preparation.core.stats_lvl2 import (get_one_gram_stats,
                                             get_three_gram_stats,
                                             get_two_gram_stats)


def prepare_core(merge_data: MergedDataset) -> PreparedDataList:
  # logger = getLogger(__name__)
  restset = PreparedDataList.init_from_merged_ds(merge_data)
  return restset


def add_rest(existing_set: PreparedDataList, restset: PreparedDataList, symbols: SymbolIdDict) -> Tuple[PreparedDataList, PreparedDataList]:
  # logger = getLogger(__name__)
  new_set = existing_set
  new_restset = PreparedDataList()
  new_set.extend(restset)
  return new_set, new_restset


def __add(existing_set: PreparedDataList, restset: PreparedDataList, symbols: SymbolIdDict, func: Callable[[OrderedDictType[int, List[str]], SymbolIdDict], OrderedSet[int]], **kwargs) -> Tuple[PreparedDataList, PreparedDataList]:
  logger = getLogger(__name__)
  new_set = existing_set
  new_restset = PreparedDataList()

  available_speaker_data = get_speaker_wise(restset)
  existing_speaker_data = get_speaker_wise(existing_set)

  for speaker_id, speaker_available in available_speaker_data.items():
    speaker_existing = existing_speaker_data[speaker_id] if speaker_id in existing_speaker_data else PreparedDataList(
    )

    speaker_existing_dict = prep_data_list_to_dict_with_symbols(speaker_existing, symbols)
    speaker_available_dict = prep_data_list_to_dict_with_symbols(speaker_available, symbols)

    selected_keys = func(
      speaker_available=speaker_available,
      speaker_available_dict=speaker_available_dict,
      speaker_existing=speaker_existing,
      speaker_existing_dict=speaker_existing_dict,
      **kwargs
    )

    not_selected_keys = set(speaker_available_dict.keys()).difference(selected_keys)
    selected_data = select_enties_from_prep_data(selected_keys, speaker_available)
    not_selected_data = select_enties_from_prep_data(not_selected_keys, speaker_available)
    assert len(selected_data) + len(not_selected_data) == len(speaker_available)

    if len(selected_data) == 0:
      logger.warning(
        f"The part in the destination set for speaker with id {speaker_id} is empty! There exist a total of {len(speaker_available)} entries for that speaker.")

    if len(not_selected_data) == 0:
      logger.warning(
        f"The part in rest set for speaker with id {speaker_id} is empty! There exist a total of {len(speaker_available)} entries for that speaker.")

    new_set.extend(selected_data)
    new_restset.extend(not_selected_data)

    logger.info(
      f"Took {len(selected_data)}/{len(speaker_available)} utterances from speaker {speaker_id} ({selected_data.get_total_duration_s()/60:.2f}min/{selected_data.get_total_duration_s()/60/60:.2f}h).")

  return new_set, new_restset


def get_random_seconds_divergent_seeds(restset: PreparedDataList, symbols: SymbolIdDict, seed: int, seconds: float, samples: int, n: int) -> OrderedSet[int]:
  available_speaker_data = get_speaker_wise(restset)

  if len(available_speaker_data) > 1:
    raise Error("This method is not supported for multiple speakers.")

  speaker_available = list(available_speaker_data.values())[0]
  speaker_available_dict = prep_data_list_to_dict_with_symbols(speaker_available, symbols)
  speaker_avail_durations_s = prep_data_list_to_dict_with_durations_s(speaker_available)

  selected_seeds = random_seconds_divergence_seeds(
    data=speaker_available_dict,
    seed=seed,
    durations_s=speaker_avail_durations_s,
    seconds=seconds,
    samples=samples,
    n=n,
  )

  return selected_seeds


def add_random_percent(existing_set: PreparedDataList, restset: PreparedDataList, symbols: SymbolIdDict, seed: int, percent: float) -> Tuple[PreparedDataList, PreparedDataList]:
  return __add(
    existing_set=existing_set,
    restset=restset,
    symbols=symbols,
    func=__add_random_percent,
    seed=seed,
    percent=percent,
  )


def __add_random_percent(speaker_available: PreparedDataList, speaker_available_dict: OrderedDictType[int, List[str]], speaker_existing: PreparedDataList, speaker_existing_dict: OrderedDictType[int, List[str]], seed: int, percent: float):
  return random_percent(speaker_available_dict, seed, percent)


def add_symbols(existing_set: PreparedDataList, restset: PreparedDataList, symbols: SymbolIdDict, cover_symbol_ids: Set[int]):
  cover_symbols = int_set_to_symbols(cover_symbol_ids, symbols)
  return __add(
    existing_set=existing_set,
    restset=restset,
    symbols=symbols,
    func=__add_symbols,
    cover_symbols=cover_symbols,
  )


def __add_symbols(speaker_available: PreparedDataList, speaker_available_dict: OrderedDictType[int, List[str]], speaker_existing: PreparedDataList, speaker_existing_dict: OrderedDictType[int, List[str]], cover_symbols: Set[int]):
  return cover_symbols_default(speaker_available_dict, cover_symbols)


def add_random_seconds(existing_set: PreparedDataList, restset: PreparedDataList, symbols: SymbolIdDict, seed: int, seconds: float, respect_existing: bool):
  return __add(
    existing_set=existing_set,
    restset=restset,
    symbols=symbols,
    func=__add_random_seconds,
    seed=seed,
    seconds=seconds,
    respect_existing=respect_existing,
  )


def __add_random_seconds(speaker_available: PreparedDataList, speaker_available_dict: OrderedDictType[int, List[str]], speaker_existing: PreparedDataList, speaker_existing_dict: OrderedDictType[int, List[str]], seed: int, seconds: float, respect_existing: bool):
  speaker_avail_durations_s = prep_data_list_to_dict_with_durations_s(speaker_available)
  if respect_existing:
    speaker_exist_durations_s = prep_data_list_to_dict_with_durations_s(speaker_existing)
    existing_s = sum(speaker_exist_durations_s.values())
    open_s = seconds - existing_s
    seconds = max(0, open_s)

  return random_seconds(speaker_available_dict, seed, speaker_avail_durations_s, seconds)


def add_random_ngram_cover_seconds(existing_set: PreparedDataList, restset: PreparedDataList, symbols: SymbolIdDict, n_gram: int, ignore_symbol_ids: Optional[Set[int]], seed: int, seconds: float):
  ignore_symbols = int_set_to_symbols(ignore_symbol_ids, symbols)
  return __add(
    existing_set=existing_set,
    restset=restset,
    symbols=symbols,
    func=__add_random_ngram_cover_seconds,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    seed=seed,
    seconds=seconds,
  )


def __add_random_ngram_cover_seconds(speaker_available: PreparedDataList, speaker_available_dict: OrderedDictType[int, List[str]], speaker_existing: PreparedDataList, speaker_existing_dict: OrderedDictType[int, List[str]], n_gram: int, ignore_symbols: Optional[Set[str]], seed: int, seconds: float):
  speaker_avail_durations_s = prep_data_list_to_dict_with_durations_s(speaker_available)
  return random_ngrams_cover_seconds(speaker_available_dict, n_gram, ignore_symbols, seed, speaker_avail_durations_s, seconds)


def add_ngram_cover(existing_set: PreparedDataList, restset: PreparedDataList, symbols: SymbolIdDict, n_gram: int, ignore_symbol_ids: Optional[Set[int]], top_percent: Optional[float]):
  ignore_symbols = int_set_to_symbols(ignore_symbol_ids, symbols)
  return __add(
    existing_set=existing_set,
    restset=restset,
    symbols=symbols,
    func=__add_ngram_cover,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    top_percent=top_percent,
  )


def __add_ngram_cover(speaker_available: PreparedDataList, speaker_available_dict: OrderedDictType[int, List[str]], speaker_existing: PreparedDataList, speaker_existing_dict: OrderedDictType[int, List[str]], n_gram: int, ignore_symbols: Optional[Set[str]], top_percent: Optional[float]):
  return greedy_ngrams_cover(
    already_covered=speaker_existing_dict,
    data=speaker_available_dict,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    top_percent=top_percent,
  )


def add_greedy_ngram_seconds(existing_set: PreparedDataList, restset: PreparedDataList, symbols: SymbolIdDict, n_gram: int, ignore_symbol_ids: Optional[Set[int]], seconds: float):
  ignore_symbols = int_set_to_symbols(ignore_symbol_ids, symbols)
  return __add(
    existing_set=existing_set,
    restset=restset,
    symbols=symbols,
    func=__add_greedy_ngram_seconds,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    seconds=seconds,
  )


def __add_greedy_ngram_seconds(speaker_available: PreparedDataList, speaker_available_dict: OrderedDictType[int, List[str]], speaker_existing: PreparedDataList, speaker_existing_dict: OrderedDictType[int, List[str]], n_gram: int, ignore_symbols: Optional[Set[str]], seconds: float):
  speaker_avail_durations = prep_data_list_to_dict_with_durations_s(speaker_available)
  return greedy_ngrams_seconds(speaker_available_dict, n_gram, ignore_symbols, speaker_avail_durations, seconds)


def add_greedy_ngram_epochs(existing_set: PreparedDataList, restset: PreparedDataList, symbols: SymbolIdDict, n_gram: int, ignore_symbol_ids: Optional[Set[int]], epochs: int):
  ignore_symbols = int_set_to_symbols(ignore_symbol_ids, symbols)
  return __add(
    existing_set=existing_set,
    restset=restset,
    symbols=symbols,
    func=__add_greedy_ngram_epochs,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    epochs=epochs,
  )


def __add_greedy_ngram_epochs(speaker_available: PreparedDataList, speaker_available_dict: OrderedDictType[int, List[str]], speaker_existing: PreparedDataList, speaker_existing_dict: OrderedDictType[int, List[str]], n_gram: int, ignore_symbols: Optional[Set[str]], epochs: int):
  return greedy_ngrams_epochs(speaker_available_dict, n_gram, ignore_symbols, epochs)


def add_greedy_kld_ngram_seconds(existing_set: PreparedDataList, restset: PreparedDataList, symbols: SymbolIdDict, n_gram: int, ignore_symbol_ids: Optional[Set[int]], seconds: float):
  ignore_symbols = int_set_to_symbols(ignore_symbol_ids, symbols)
  return __add(
    existing_set=existing_set,
    restset=restset,
    symbols=symbols,
    func=__add_greedy_kld_ngram_seconds,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    seconds=seconds,
  )


def __add_greedy_kld_ngram_seconds(speaker_available: PreparedDataList, speaker_available_dict: OrderedDictType[int, List[str]], speaker_existing: PreparedDataList, speaker_existing_dict: OrderedDictType[int, List[str]], n_gram: int, ignore_symbols: Optional[Set[str]], seconds: float):
  speaker_avail_durations = prep_data_list_to_dict_with_durations_s(speaker_available)
  return greedy_kld_uniform_ngrams_seconds(speaker_available_dict, n_gram, ignore_symbols, speaker_avail_durations, seconds)


def core_process_stats(onegram_stats: pd.DataFrame, twogram_stats: pd.DataFrame, threegram_stats: pd.DataFrame, speaker_stats: pd.DataFrame, ds: DatasetType, logger: Logger):
  get_one_gram_stats(onegram_stats, ds, logger)
  get_two_gram_stats(twogram_stats, ds, logger)
  get_three_gram_stats(threegram_stats, ds, logger)


def int_set_to_symbols(symbol_ids: Optional[Set[int]], symbols: SymbolIdDict) -> Optional[Set[str]]:
  if symbol_ids is None:
    return None
  ignore_symbols = set(symbols.get_symbols(list(symbol_ids)))
  return ignore_symbols
