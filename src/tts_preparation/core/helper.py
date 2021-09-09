from collections import OrderedDict
from typing import OrderedDict as OrderedDictType
from typing import Set

from text_utils import SymbolIdDict
from text_utils.types import Symbols
from tts_preparation.core.data import EntryId, PreparedDataList
from tts_preparation.globals import DEFAULT_PADDING_SYMBOL


def get_percent(l: int, total: int):
  return l / total * 100 if total > 0 else 0


def get_percent_str(l: int, total: int):
  p = get_percent(l, total)
  res = f"{p:.2f}%"
  return res


def get_shard_size(symbols: SymbolIdDict) -> int:
  all_symbols = symbols.get_all_symbols()
  count = len(all_symbols)
  if DEFAULT_PADDING_SYMBOL in all_symbols:
    count -= 1
  return count


def symbol_count_to_shards(symbol_count: int, shard_size: float) -> float:
  assert shard_size > 0
  return symbol_count / shard_size


def get_total_set(trainset: PreparedDataList, valset: PreparedDataList, testset: PreparedDataList, restset: PreparedDataList) -> PreparedDataList:
  total_set = []
  total_set.extend(trainset)
  total_set.extend(testset)
  total_set.extend(valset)
  total_set.extend(restset)
  total_set = PreparedDataList(total_set)
  return total_set


# def merge_prep_data_lists(l1: PreparedDataList, l2: PreparedDataList) -> PreparedDataList:
#   res = PreparedDataList(l1)
#   res.extend(l2)
#   return res


# def prep_data_list_to_dict_with_symbol_ids(l: PreparedDataList) -> OrderedDictType[int, List[int]]:
#   res = OrderedDict({x.entry_id: deserialize_list(x.serialized_symbol_ids) for x in l.items()})
#   return res


def prep_data_list_to_dict_with_symbols(data: PreparedDataList) -> OrderedDictType[EntryId, Symbols]:
  res = OrderedDict({entry.entry_id: entry.symbols for entry in data.items()})
  return res


def prep_data_list_to_dict_with_durations_s(data: PreparedDataList) -> OrderedDictType[EntryId, float]:
  res = OrderedDict({entry.entry_id: entry.duration_s for entry in data.items()})
  return res


# def dict_with_symbols_to_prep_data_list(d: Dict[int, List[int]], select_from: PreparedDataList):
#   res = [x for x in select_from.items() if x.entry_id in d]
#   return res


# def select_entities_from_dict(keys: Set[int], select_from: Dict) -> Dict:
#   keys_exist = len(keys.difference(select_from.keys())) == 0
#   assert keys_exist
#   res = {k: v for k, v in select_from.items() if k in keys}
#   return res


def select_entities_from_prep_data(entry_ids: Set[EntryId], select_from: PreparedDataList) -> PreparedDataList:
  all_entry_ids_exist = len(entry_ids - select_from.unique_entry_ids) == 0
  assert all_entry_ids_exist
  res = PreparedDataList(entry for entry in select_from.items() if entry.entry_id in entry_ids)
  return res
