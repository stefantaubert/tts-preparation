from logging import getLogger
from typing import Callable, List, Optional
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple

import pandas as pd
from ordered_set import OrderedSet
from text_selection import (cover_symbols_default,
                            greedy_kld_uniform_ngrams_seconds,
                            greedy_ngrams_cover, greedy_ngrams_epochs,
                            greedy_ngrams_seconds, n_divergent_random_seconds,
                            random_ngrams_cover_seconds, random_percent,
                            random_seconds)
from text_utils import Symbol, SymbolIdDict, Symbols
from tts_preparation.core.data import (DatasetType, EntryId, PreparedDataList,
                                       get_speaker_wise)
from tts_preparation.core.helper import (
    prep_data_list_to_dict_with_durations_s,
    prep_data_list_to_dict_with_symbols, select_entities_from_prep_data)
from tts_preparation.core.stats_lvl2 import (get_one_gram_stats,
                                             get_three_gram_stats,
                                             get_two_gram_stats)


def add_rest(existing_set: PreparedDataList, restset: PreparedDataList) -> Tuple[PreparedDataList, PreparedDataList]:
  new_set = existing_set + restset
  new_restset = PreparedDataList()
  return new_set, new_restset


def __add(existing_set: PreparedDataList, restset: PreparedDataList, func: Callable[[OrderedDictType[EntryId, Symbols], SymbolIdDict], OrderedSet[EntryId]], **kwargs) -> Tuple[PreparedDataList, PreparedDataList]:
  logger = getLogger(__name__)
  new_set = existing_set
  new_restset = PreparedDataList()

  available_speakers_data = get_speaker_wise(restset)
  existing_speakers_data = get_speaker_wise(existing_set)

  for speaker, available_speaker_data in available_speakers_data.items():
    existing_speaker_data = existing_speakers_data[speaker] if speaker in existing_speakers_data else PreparedDataList(
    )

    selected_entry_ids = func(
      speaker_available=available_speaker_data,
      speaker_existing=existing_speaker_data,
      **kwargs
    )

    not_selected_entry_ids = available_speaker_data.unique_entry_ids - selected_entry_ids

    #not_selected_keys = set(speaker_available_dict.keys()).difference(selected_entry_ids)
    selected_data = select_entities_from_prep_data(selected_entry_ids, available_speaker_data)
    not_selected_data = select_entities_from_prep_data(
      not_selected_entry_ids, available_speaker_data)
    assert len(selected_data) + len(not_selected_data) == len(available_speaker_data)

    if len(selected_data) == 0:
      logger.warning(
        f"The part in the destination set for speaker with id {speaker} is empty! There exist a total of {len(available_speaker_data)} entries for that speaker.")

    if len(not_selected_data) == 0:
      logger.warning(
        f"The part in rest set for speaker with id {speaker} is empty! There exist a total of {len(available_speaker_data)} entries for that speaker.")

    new_set.extend(selected_data)
    new_restset.extend(not_selected_data)

    logger.info(
      f"Took {len(selected_data)}/{len(available_speaker_data)} utterances from speaker {speaker} ({selected_data.total_duration_s/60:.2f}min/{selected_data.total_duration_s/60/60:.2f}h).")

  return new_set, new_restset


def get_random_seconds_divergent_seeds(restset: PreparedDataList, seed: int, seconds: float, n: int) -> OrderedSet[int]:
  available_speaker_data = get_speaker_wise(restset)
  has_more_than_one_speaker = len(available_speaker_data) > 1
  if has_more_than_one_speaker:
    raise ValueError("This method is not supported for multiple speakers.")

  speaker_available = list(available_speaker_data.values())[0]
  speaker_available_dict = prep_data_list_to_dict_with_symbols(speaker_available)
  speaker_avail_durations_s = prep_data_list_to_dict_with_durations_s(speaker_available)

  selected_seeds = n_divergent_random_seconds(
    data=speaker_available_dict,
    seed=seed,
    durations_s=speaker_avail_durations_s,
    seconds=seconds,
    n=n,
  )

  return selected_seeds


def add_n_divergent_random_seconds(existing_set: PreparedDataList, restset: PreparedDataList, seed: int, seconds: float, n: int) -> List[Tuple[PreparedDataList, PreparedDataList]]:
  logger = getLogger(__name__)

  new_datasets: List[Tuple[PreparedDataList, PreparedDataList]] = []

  available_speaker_data = get_speaker_wise(restset)

  for speaker_id, speaker_available in available_speaker_data.items():
    speaker_available_dict = prep_data_list_to_dict_with_symbols(speaker_available)
    speaker_avail_durations_s = prep_data_list_to_dict_with_durations_s(speaker_available)

    selected_list_of_keys = n_divergent_random_seconds(
      n=n,
      seconds=seconds,
      durations_s=speaker_avail_durations_s,
      data=speaker_available_dict,
      seed=seed,
    )

    for i, k in enumerate(selected_list_of_keys):
      not_selected_keys = set(speaker_available_dict.keys()).difference(k)
      selected_data = select_entities_from_prep_data(k, speaker_available)
      not_selected_data = select_entities_from_prep_data(not_selected_keys, speaker_available)
      assert len(selected_data) + len(not_selected_data) == len(speaker_available)

      new_set = PreparedDataList(existing_set + selected_data)
      new_restset = PreparedDataList(not_selected_data)
      new_datasets.append((new_set, new_restset))

      logger.info(
        f"{i+1}/{n}: Took {len(selected_data)}/{len(speaker_available)} utterances from speaker {speaker_id} ({selected_data.total_duration_s/60:.2f}min/{selected_data.total_duration_s/60/60:.2f}h).")

  return new_datasets


def add_random_percent(existing_set: PreparedDataList, restset: PreparedDataList, seed: int, percent: float) -> Tuple[PreparedDataList, PreparedDataList]:
  return __add(
    existing_set=existing_set,
    restset=restset,
    func=__add_random_percent,
    seed=seed,
    percent=percent,
  )


def __add_random_percent(speaker_available: PreparedDataList, speaker_existing: PreparedDataList, seed: int, percent: float) -> OrderedSet[EntryId]:
  speaker_available_dict = prep_data_list_to_dict_with_symbols(speaker_available)
  return random_percent(speaker_available_dict, seed, percent)


def add_symbols(existing_set: PreparedDataList, restset: PreparedDataList, cover_symbols: Set[Symbol]):
  return __add(
    existing_set=existing_set,
    restset=restset,
    func=__add_symbols,
    cover_symbols=cover_symbols,
  )


def __add_symbols(speaker_available: PreparedDataList, speaker_existing: PreparedDataList, cover_symbols: Set[int]) -> OrderedSet[EntryId]:
  speaker_available_dict = prep_data_list_to_dict_with_symbols(speaker_available)
  return cover_symbols_default(speaker_available_dict, cover_symbols)


def add_random_seconds(existing_set: PreparedDataList, restset: PreparedDataList, symbols: SymbolIdDict, seed: int, seconds: float, respect_existing: bool):
  return __add(
    existing_set=existing_set,
    restset=restset,
    func=__add_random_seconds,
    seed=seed,
    seconds=seconds,
    respect_existing=respect_existing,
  )


def __add_random_seconds(speaker_available: PreparedDataList, speaker_existing: PreparedDataList, seed: int, seconds: float, respect_existing: bool) -> OrderedSet[EntryId]:
  speaker_avail_durations_s = prep_data_list_to_dict_with_durations_s(speaker_available)
  if respect_existing:
    speaker_exist_durations_s = prep_data_list_to_dict_with_durations_s(speaker_existing)
    existing_s = sum(speaker_exist_durations_s.values())
    open_s = seconds - existing_s
    seconds = max(0, open_s)

  speaker_available_dict = prep_data_list_to_dict_with_symbols(speaker_available)
  return random_seconds(speaker_available_dict, seed, speaker_avail_durations_s, seconds)


def add_random_ngram_cover_seconds(existing_set: PreparedDataList, restset: PreparedDataList, n_gram: int, ignore_symbols: Optional[Set[Symbol]], seed: int, seconds: float):
  return __add(
    existing_set=existing_set,
    restset=restset,
    func=__add_random_ngram_cover_seconds,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    seed=seed,
    seconds=seconds,
  )


def __add_random_ngram_cover_seconds(speaker_available: PreparedDataList, speaker_existing: PreparedDataList, n_gram: int, ignore_symbols: Optional[Set[str]], seed: int, seconds: float) -> OrderedSet[EntryId]:
  speaker_available_dict = prep_data_list_to_dict_with_symbols(speaker_available)
  speaker_avail_durations_s = prep_data_list_to_dict_with_durations_s(speaker_available)
  return random_ngrams_cover_seconds(speaker_available_dict, n_gram, ignore_symbols, seed, speaker_avail_durations_s, seconds)


def add_ngram_cover(existing_set: PreparedDataList, restset: PreparedDataList, n_gram: int, ignore_symbols: Optional[Set[Symbol]], top_percent: Optional[float]):
  return __add(
    existing_set=existing_set,
    restset=restset,
    func=__add_ngram_cover,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    top_percent=top_percent,
  )


def __add_ngram_cover(speaker_available: PreparedDataList, speaker_existing: PreparedDataList, n_gram: int, ignore_symbols: Optional[Set[str]], top_percent: Optional[float]) -> OrderedSet[EntryId]:
  speaker_available_dict = prep_data_list_to_dict_with_symbols(speaker_available)
  speaker_existing_dict = prep_data_list_to_dict_with_symbols(speaker_existing)
  return greedy_ngrams_cover(
    already_covered=speaker_existing_dict,
    data=speaker_available_dict,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    top_percent=top_percent,
  )


def add_greedy_ngram_seconds(existing_set: PreparedDataList, restset: PreparedDataList, n_gram: int, ignore_symbols: Optional[Set[Symbol]], seconds: float) -> Tuple[PreparedDataList, PreparedDataList]:
  return __add(
    existing_set=existing_set,
    restset=restset,
    func=__add_greedy_ngram_seconds,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    seconds=seconds,
  )


def __add_greedy_ngram_seconds(speaker_available: PreparedDataList, speaker_existing: PreparedDataList, n_gram: int, ignore_symbols: Optional[Set[str]], seconds: float) -> OrderedSet[EntryId]:
  speaker_available_dict = prep_data_list_to_dict_with_symbols(speaker_available)
  speaker_avail_durations = prep_data_list_to_dict_with_durations_s(speaker_available)
  return greedy_ngrams_seconds(speaker_available_dict, n_gram, ignore_symbols, speaker_avail_durations, seconds)


def add_greedy_ngram_epochs(existing_set: PreparedDataList, restset: PreparedDataList, n_gram: int, ignore_symbols: Optional[Set[Symbol]], epochs: int) -> Tuple[PreparedDataList, PreparedDataList]:
  return __add(
    existing_set=existing_set,
    restset=restset,
    func=__add_greedy_ngram_epochs,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    epochs=epochs,
  )


def __add_greedy_ngram_epochs(speaker_available: PreparedDataList, speaker_existing: PreparedDataList, n_gram: int, ignore_symbols: Optional[Set[str]], epochs: int) -> OrderedSet[EntryId]:
  speaker_available_dict = prep_data_list_to_dict_with_symbols(speaker_available)
  return greedy_ngrams_epochs(speaker_available_dict, n_gram, ignore_symbols, epochs)


def add_greedy_kld_ngram_seconds(existing_set: PreparedDataList, restset: PreparedDataList, n_gram: int, ignore_symbols: Optional[Set[Symbol]], seconds: float):
  return __add(
    existing_set=existing_set,
    restset=restset,
    func=__add_greedy_kld_ngram_seconds,
    n_gram=n_gram,
    ignore_symbols=ignore_symbols,
    seconds=seconds,
  )


def __add_greedy_kld_ngram_seconds(speaker_available: PreparedDataList, speaker_existing: PreparedDataList, n_gram: int, ignore_symbols: Optional[Set[str]], seconds: float) -> OrderedSet[EntryId]:
  speaker_available_dict = prep_data_list_to_dict_with_symbols(speaker_available)
  speaker_avail_durations = prep_data_list_to_dict_with_durations_s(speaker_available)
  return greedy_kld_uniform_ngrams_seconds(speaker_available_dict, n_gram, ignore_symbols, speaker_avail_durations, seconds)


def core_process_stats(onegram_stats: pd.DataFrame, twogram_stats: pd.DataFrame, threegram_stats: pd.DataFrame, speaker_stats: pd.DataFrame, ds: DatasetType) -> None:
  get_one_gram_stats(onegram_stats, ds)
  get_two_gram_stats(twogram_stats, ds)
  get_three_gram_stats(threegram_stats, ds)


# def int_set_to_symbols(symbol_ids: Optional[Set[int]], symbols: SymbolIdDict) -> Optional[Set[str]]:
#   if symbol_ids is None:
#     return None
#   ignore_symbols = set(symbols.get_symbols(list(symbol_ids)))
#   return ignore_symbols
