from logging import Logger
from math import ceil
from typing import Counter, Dict, List, Optional, Set, Tuple

from text_utils import SymbolIdDict
from text_utils.text import get_ngrams
from tts_preparation.core.data import PreparedDataList, get_speaker_wise
from tts_preparation.core.extraction_methods import (
    get_greedy_kld_utterances_count, get_greedy_utterances_count,
    get_random_utterances_count)
from tts_preparation.core.helper import (
    dict_with_symbols_to_prep_data_list, get_shard_size, merge_prep_data_lists,
    prep_data_list_to_dict_with_symbol_ids, select_enties_from_dict,
    select_enties_from_prep_data)


def add_random_count_core(symbols: SymbolIdDict, existing_set: PreparedDataList, restset: PreparedDataList, shards_per_speaker: int, min_count_symbol: int, seed: int, ignore_already_added: bool, logger: Logger) -> Tuple[PreparedDataList, PreparedDataList]:
  new_set = PreparedDataList(existing_set)
  new_restset = PreparedDataList()
  shard_size = get_shard_size(symbols)
  test_size = ceil(shards_per_speaker * shard_size)
  existing_speaker_data = get_speaker_wise(existing_set)
  restset_speaker_data = get_speaker_wise(restset)

  for speaker_id, speaker_available in restset_speaker_data.items():
    existing_data = existing_speaker_data.get(speaker_id, PreparedDataList())
    speaker_total_set = merge_prep_data_lists(existing_data, speaker_available)

    total_symbol_ids = prep_data_list_to_dict_with_symbol_ids(speaker_total_set)
    existing_d = prep_data_list_to_dict_with_symbol_ids(existing_data)
    available_d = prep_data_list_to_dict_with_symbol_ids(speaker_available)

    dest_part, rest_part = get_random_utterances_count(
      available=available_d,
      existing=existing_d,
      symbol_ids_total_set=total_symbol_ids,
      total_symbol_count=test_size,
      min_count_symbol=min_count_symbol,
      seed=seed,
      ignore_already_added=ignore_already_added,
      max_iters=None,
    )

    if len(dest_part) == 0:
      logger.warn(
        f"The part in the destination set for speaker with id {speaker_id} is empty! There exist a total of {len(speaker_available)} entries for that speaker.")

    if len(rest_part) == 0:
      logger.warn(
        f"The part in rest set for speaker with id {speaker_id} is empty! There exist a total of {len(speaker_available)} entries for that speaker.")

    selected_data = dict_with_symbols_to_prep_data_list(dest_part, speaker_available)
    rest_data = dict_with_symbols_to_prep_data_list(rest_part, speaker_available)

    new_set.extend(selected_data)
    new_restset.extend(rest_data)

  return new_set, new_restset


def add_ngram_kld_core(symbols: SymbolIdDict, existing_set: PreparedDataList, restset: PreparedDataList, total_set: PreparedDataList, shards_per_speaker: Optional[int], n_gram: int, n_its: Optional[int], min_count_symbol: int, ignore_already_added: bool, top_percent: float, logger: Logger) -> Tuple[PreparedDataList, PreparedDataList]:
  new_set = PreparedDataList(existing_set)
  new_restset = PreparedDataList()
  shard_size = get_shard_size(symbols)
  test_size = ceil(shards_per_speaker * shard_size) if shards_per_speaker is not None else None
  existing_speaker_data = get_speaker_wise(existing_set)
  restset_speaker_data = get_speaker_wise(restset)
  totalset_speaker_data = get_speaker_wise(total_set)

  for speaker_id, speaker_available in restset_speaker_data.items():
    existing_data = existing_speaker_data.get(speaker_id, PreparedDataList())
    speaker_rest_and_available_set = merge_prep_data_lists(existing_data, speaker_available)
    # symbols that occur only in e.g. trainset or other sets can not be selected therefore only existing + rest
    rest_and_available_possible_symbol_ids = prep_data_list_to_dict_with_symbol_ids(
      speaker_rest_and_available_set)

    assert speaker_id in totalset_speaker_data
    speaker_totalset_symbol_ids = prep_data_list_to_dict_with_symbol_ids(
      totalset_speaker_data[speaker_id])

    speaker_totalset_ngrams: Dict[int, List[Tuple]] = {
      k: get_ngrams(v, n_gram) for k, v in speaker_totalset_symbol_ids.items()
    }
    all_ngrams_list: List[Tuple] = [y for x in speaker_totalset_ngrams.values() for y in x]
    all_ngrams_set: Set[Tuple] = set(all_ngrams_list)
    counter_ngrams = Counter(all_ngrams_list)
    ngrams_ordered_by_occurrence = [ngram for ngram, _ in counter_ngrams.most_common()]
    top_ngrams_count = ceil(top_percent / 100 * len(all_ngrams_set))
    top_ngrams = set(ngrams_ordered_by_occurrence[:top_ngrams_count])
    final_ngrams_to_cover_set = top_ngrams
    if len(final_ngrams_to_cover_set) != len(all_ngrams_set):
      logger.info(
          f"Selected top {len(final_ngrams_to_cover_set)}/{len(all_ngrams_set)} ({top_percent}%) {n_gram}-grams for speaker {speaker_id}: {' '.join(str(x) for x in top_ngrams)}")
    else:
      logger.info(f"Selected all occurring {n_gram}-grams ({len(final_ngrams_to_cover_set)}).")

    logger.info(
      f"The theoretically maximum amount of {n_gram}-grams is: {shard_size ** n_gram}. Occurring amount ({len(final_ngrams_to_cover_set)}) equals to: {len(final_ngrams_to_cover_set)/(shard_size ** n_gram)*100:.2f}%.")

    uniform_distribution = {ngram: 1 / len(final_ngrams_to_cover_set)
                            for ngram in final_ngrams_to_cover_set}

    existing_ngrams_for_speaker_dict = select_enties_from_dict(
      existing_data.get_entry_ids(), speaker_totalset_ngrams)
    available_ngrams_for_speaker_dict = select_enties_from_dict(
      speaker_available.get_entry_ids(), speaker_totalset_ngrams)

    dest_part, rest_part = get_greedy_kld_utterances_count(
      existing=existing_ngrams_for_speaker_dict,
      available=available_ngrams_for_speaker_dict,
      symbol_ids_total_set=rest_and_available_possible_symbol_ids,
      total_symbol_count=test_size,
      min_count_symbol=min_count_symbol,
      ignore_already_added=ignore_already_added,
      target_dist=uniform_distribution,
      max_iters=n_its,
      cover_set=final_ngrams_to_cover_set,
    )

    selected_data = select_enties_from_prep_data(set(dest_part.keys()), speaker_available)
    rest_data = select_enties_from_prep_data(set(rest_part.keys()), speaker_available)

    new_set.extend(selected_data)
    new_restset.extend(rest_data)

    logger.info(
      f"Took {len(dest_part)}/{len(speaker_available)} utterances from speaker {speaker_id}.")

  return new_set, new_restset


def filter_ngrams(ngrams: List[Tuple], ignore_symbol_ids: Set[int]) -> List[Tuple]:
  res = [x for x in ngrams if len(set(x).intersection(ignore_symbol_ids)) == 0]
  return res


def add_ngram_core(symbols: SymbolIdDict, existing_set: PreparedDataList, restset: PreparedDataList, total_set: PreparedDataList, shards_per_speaker: Optional[int], n_gram: int, n_its: Optional[int], min_count_symbol: int, ignore_already_added: bool, top_percent: float, logger: Logger) -> Tuple[PreparedDataList, PreparedDataList]:
  new_set = PreparedDataList(existing_set)
  new_restset = PreparedDataList()
  shard_size = get_shard_size(symbols)
  test_size = ceil(shards_per_speaker * shard_size) if shards_per_speaker is not None else None
  existing_speaker_data = get_speaker_wise(existing_set)
  restset_speaker_data = get_speaker_wise(restset)
  totalset_speaker_data = get_speaker_wise(total_set)

  for speaker_id, speaker_available in restset_speaker_data.items():
    existing_data = existing_speaker_data.get(speaker_id, PreparedDataList())
    speaker_rest_and_available_set = merge_prep_data_lists(existing_data, speaker_available)
    # symbols that occur only in e.g. trainset or other sets can not be selected therefore only existing + rest
    rest_and_available_possible_symbol_ids = prep_data_list_to_dict_with_symbol_ids(
      speaker_rest_and_available_set)

    assert speaker_id in totalset_speaker_data
    speaker_totalset_symbol_ids = prep_data_list_to_dict_with_symbol_ids(
      totalset_speaker_data[speaker_id])

    speaker_totalset_ngrams: Dict[int, List[Tuple]] = {
      k: get_ngrams(v, n_gram) for k, v in speaker_totalset_symbol_ids.items()
    }
    all_ngrams_list: List[Tuple] = [y for x in speaker_totalset_ngrams.values() for y in x]
    all_ngrams_set: Set[Tuple] = set(all_ngrams_list)
    counter_ngrams = Counter(all_ngrams_list)
    ngrams_ordered_by_occurrence = [ngram for ngram, _ in counter_ngrams.most_common()]
    top_ngrams_count = ceil(top_percent / 100 * len(all_ngrams_set))
    top_ngrams = set(ngrams_ordered_by_occurrence[:top_ngrams_count])
    final_ngrams_to_cover_set = top_ngrams
    if len(final_ngrams_to_cover_set) != len(all_ngrams_set):
      logger.info(
          f"Selected top {len(final_ngrams_to_cover_set)}/{len(all_ngrams_set)} ({top_percent}%) {n_gram}-grams for speaker {speaker_id}: {' '.join(str(x) for x in top_ngrams)}")
    else:
      logger.info(f"Selected all occurring {n_gram}-grams ({len(final_ngrams_to_cover_set)}).")

    logger.info(
      f"The theoretically maximum amount of {n_gram}-grams is: {shard_size ** n_gram}. Occurring amount ({len(final_ngrams_to_cover_set)}) equals to: {len(final_ngrams_to_cover_set)/(shard_size ** n_gram)*100:.2f}%.")

    existing_ngrams_for_speaker_dict = select_enties_from_dict(
      existing_data.get_entry_ids(), speaker_totalset_ngrams)
    available_ngrams_for_speaker_dict = select_enties_from_dict(
      speaker_available.get_entry_ids(), speaker_totalset_ngrams)

    dest_part, rest_part = get_greedy_utterances_count(
      existing=existing_ngrams_for_speaker_dict,
      available=available_ngrams_for_speaker_dict,
      symbol_ids_total_set=rest_and_available_possible_symbol_ids,
      total_symbol_count=test_size,
      min_count_symbol=min_count_symbol,
      ignore_already_added=ignore_already_added,
      max_iters=n_its,
      cover_set=final_ngrams_to_cover_set,
    )

    selected_data = select_enties_from_prep_data(set(dest_part.keys()), speaker_available)
    rest_data = select_enties_from_prep_data(set(rest_part.keys()), speaker_available)

    new_set.extend(selected_data)
    new_restset.extend(rest_data)

    logger.info(
      f"Took {len(dest_part)}/{len(speaker_available)} utterances from speaker {speaker_id}.")

  return new_set, new_restset
