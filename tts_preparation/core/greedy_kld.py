from collections import Counter
from typing import Dict, List, Set, Tuple, TypeVar

import numpy as np

_T = TypeVar("_T")


def find_sentence_with_min_div(selection_list: Dict[int, List[_T]], current_cover: List[_T], target_dist: Dict[_T, float], cover_set: Set[_T]) -> Tuple[int, List[_T]]:
  c = Counter(current_cover)
  best_key, best_sentence = min(selection_list.items(
  ), key=lambda kv_utterance: get_divergence_for_sentence(
    c=c,
    utterance=kv_utterance[1],
    target_dist=target_dist,
    cover_set=cover_set,
  ))
  return best_key, best_sentence


def get_divergence_for_sentence(c: Counter, utterance: List[Tuple], target_dist: Dict[_T, float], cover_set: Set[_T]) -> float:
  c_utterance = Counter(utterance)
  counts: Dict[Tuple, int] = combine_counters(c, c_utterance)
  # counts = {k: counts[k] if k in counts else 0 for k in target_dist.keys()}
  # remove entries not to cover
  counts_containing_only_cover_items = {k: v for k, v in counts.items() if k in cover_set}

  distr = get_distribution(counts_containing_only_cover_items)
  kld = kullback_leibler_div(distr, target_dist)
  return kld


def combine_counters(c1: Counter, c2: Counter) -> Dict[Tuple, int]:
  counts = {k: v for k, v in c1.items()}
  for k, v in c2.items():
    if k in counts:
      counts[k] += v
    else:
      counts[k] = v
  return counts


def kullback_leibler_div(dist_1: Dict[_T, float], dist_2: Dict[_T, float]) -> float:
  for value in dist_1.values():
    assert value > 0
  for value in dist_2.values():
    assert value > 0
  for k in dist_1.keys():
    assert k in dist_2

  if len(dist_1) == 0 or len(dist_2) == 0:
    return float('inf')
  # what with non existing keys in dist_1?
  divergence = [dist_1[key] * (np.log(dist_1[key]) - np.log(dist_2[key]))
                for key in dist_1.keys()]
  res = sum(divergence)
  return res


def get_distribution(counts: Dict[Tuple, int]) -> Dict[_T, float]:
  total_number_of_single_units = sum(counts.values())
  new_dist = {key: counts[key] / total_number_of_single_units for key in counts.keys()}
  return new_dist
