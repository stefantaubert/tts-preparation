import random
from functools import partial
from typing import Callable, Dict, List, Optional, Set, Tuple, TypeVar

from sklearn.model_selection import train_test_split
from tts_preparation.core.greedy_kld import find_sentence_with_min_div
from tqdm import tqdm

_T = TypeVar("_T")


def all_symbols_are_covered(d: Dict[int, int]) -> bool:
  res = sum(d.values()) == 0
  return res


def _get_utterances(already_added_entries: Dict[int, List[_T]], available_entries: Dict[int, List[_T]], symbol_ids_total_set: Dict[int, List[int]], total_symbol_count: Optional[int], min_count_symbol: int, method: Callable[[Dict[int, List[_T]]], int], ignore_already_added: bool, max_iters: Optional[int]) -> Tuple[Dict[int, List[_T]], Dict[int, List[_T]], bool, int]:
  # todo add mincount for tuples

  assert len(already_added_entries) + len(available_entries) == len(symbol_ids_total_set)

  break_on_total_symbols_reached = total_symbol_count is not None
  break_on_total_its_reached = max_iters is not None
  break_after_all_symbols_count_covered = not break_on_total_its_reached and not break_on_total_symbols_reached

  have_break_condition = break_on_total_symbols_reached or break_on_total_its_reached or break_after_all_symbols_count_covered
  assert have_break_condition

  symbol_ids_total_set_final = symbol_ids_total_set.copy()
  old_all_symbol_ids: Set[int] = {x for y in symbol_ids_total_set_final.values() for x in y}
  if ignore_already_added:
    # to prevent having symbols that can not be added because the did not exist in available entries
    for k in already_added_entries:
      symbol_ids_total_set_final.pop(k)
    already_added_entries = dict()

  all_symbol_ids: Set[int] = {x for y in symbol_ids_total_set_final.values() for x in y}
  removed_symbols = old_all_symbol_ids.difference(all_symbol_ids)
  if len(removed_symbols) > 0:
    print(
      f"Removed symbols which cannot be covert because the do not exist in the available entries: {removed_symbols}")
  covered_symbols: Dict[int, int] = {x: min_count_symbol for x in all_symbol_ids}
  # maybe add min(total_occing, mincountsymbol)
  current_symbol_count = 0
  covered: List[_T] = []

  for k, v in already_added_entries.items():
    assert k in symbol_ids_total_set_final
    symbs = symbol_ids_total_set_final[k]
    covered.extend(v)
    current_symbol_count += len(symbs)
    for s in symbs:
      assert s in covered_symbols
      covered_symbols[s] = max(0, covered_symbols[s] - 1)

  rest_entries = available_entries.copy()
  selected_entries = already_added_entries.copy()
  new_selected_entries = dict()

  if break_on_total_symbols_reached:
    t = tqdm(total=total_symbol_count, initial=current_symbol_count)
  if break_on_total_its_reached:
    t = tqdm(total=max_iters, initial=0)
  if break_after_all_symbols_count_covered:
    t = tqdm(total=len(all_symbol_ids) * min_count_symbol, initial=len(all_symbol_ids)
             * min_count_symbol - sum(covered_symbols.values()))

  rest_iterations = max_iters
  while True:
    no_available_entries_left = len(rest_entries) == 0
    if no_available_entries_left:
      break
    no_iterations_left = rest_iterations == 0
    if break_on_total_its_reached and no_iterations_left:
      break
    if break_after_all_symbols_count_covered and all_symbols_are_covered(covered_symbols):
      break

    dest_sel_list = get_available_entries(
      covered_symbols,
      rest_entries,
      symbol_ids_total_set_final
    )
    no_utterances_with_open_symbols_exist_anymore = len(dest_sel_list) == 0
    if no_utterances_with_open_symbols_exist_anymore:
      break

    selected_id = method(dest_sel_list, covered)
    selected_entry = dest_sel_list[selected_id]
    selected_symbol_ids = symbol_ids_total_set_final[selected_id]
    symbol_count = len(selected_symbol_ids)

    if break_on_total_symbols_reached and current_symbol_count + symbol_count > total_symbol_count:
      break

    rest_entries.pop(selected_id)
    assert selected_id not in selected_entries
    selected_entries[selected_id] = selected_entry
    new_selected_entries[selected_id] = selected_entry
    current_symbol_count += symbol_count
    covered.extend(selected_entry)
    for x in selected_symbol_ids:
      covered_symbols[x] = max(0, covered_symbols[x] - 1)
    if rest_iterations is not None:
      rest_iterations -= 1
    if break_on_total_symbols_reached:
      t.update(symbol_count)
    if break_on_total_its_reached:
      t.update(1)
    if break_after_all_symbols_count_covered:
      t.update(len(all_symbol_ids) * min_count_symbol - sum(covered_symbols.values()) - t.n)
  t.close()
  covered_all = all_symbols_are_covered(covered_symbols)

  # not because you can define a number of symbols that is not existent: assert not break_after_all_symbols_count_covered or covered_all
  assert len(new_selected_entries) + len(rest_entries) == len(available_entries)
  # return split of available entries
  return new_selected_entries, rest_entries, covered_all, current_symbol_count


def get_available_entries(covered_symbols, rest_entries, symbol_ids_total_set_final):
  open_symbols = {x for x, open_count in covered_symbols.items() if open_count > 0}
  some_symbols_missing = len(open_symbols) > 0
  dest_sel_list = rest_entries
  if some_symbols_missing:
    dest_sel_list = {k: v for k, v in dest_sel_list.items() if len(
      set(symbol_ids_total_set_final[k]).intersection(open_symbols)) > 0}
  return dest_sel_list


def _sel_rand_utt(available: Dict[int, List[_T]], covered: List[_T]) -> int:
  assert len(available) > 0
  key = random.choice(list(available.keys()))
  return key


def get_random_utterances_percent(available: Dict[int, List[_T]], percent: float, seed: int) -> Tuple[Set[int], Set[int]]:
  rest_entries, selected_entries = train_test_split(
    list(available.keys()),
    test_size=percent,
    random_state=seed,
    shuffle=True,
  )

  return selected_entries, rest_entries


def get_random_utterances_count(existing: Dict[int, List[_T]], available: Dict[int, List[_T]], symbol_ids_total_set: Dict[int, List[int]], total_symbol_count: Optional[int], min_count_symbol: int, seed: int, ignore_already_added: bool, max_iters: Optional[int]) -> Tuple[Dict[int, List[_T]], Dict[int, List[_T]]]:
  random.seed(seed)

  new_selected_entries, rest_entries, covered_all, current_symbol_count = _get_utterances(
    already_added_entries=existing,
    available_entries=available,
    ignore_already_added=ignore_already_added,
    method=_sel_rand_utt,
    min_count_symbol=min_count_symbol,
    symbol_ids_total_set=symbol_ids_total_set,
    total_symbol_count=total_symbol_count,
    max_iters=max_iters,
  )

  return new_selected_entries, rest_entries


def sel_greedy_kld(l: Dict[int, List[_T]], covered: List[_T], target_dist: Dict[_T, float], cover_set: Set[_T]) -> int:
  k, _ = find_sentence_with_min_div(
    selection_list=l,
    current_cover=covered,
    target_dist=target_dist,
    cover_set=cover_set,
  )
  return k


def sel_greedy(l: Dict[int, List[_T]], covered: List[_T], cover_set: Set[_T]) -> int:
  already_covered = set(covered)
  k, _ = max(l.items(), key=lambda x: _get_greedy_units_score(
    cover_items=cover_set,
    subset=set(x[1]),
    already_covered=already_covered)
  )
  return k


def _get_greedy_units_score(cover_items: Optional[Set[_T]], subset: Set[_T], already_covered: Set[_T]) -> int:
  relevant_items_in_subset = subset
  if cover_items is not None:
    relevant_items_in_subset = relevant_items_in_subset.intersection(cover_items)
  difference = relevant_items_in_subset - already_covered
  res = len(difference)
  return res


def get_greedy_kld_utterances_count(existing: Dict[int, List[_T]], available: Dict[int, List[_T]], symbol_ids_total_set: Dict[int, List[int]], total_symbol_count: Optional[int], min_count_symbol: int, target_dist: Dict[_T, float], ignore_already_added: bool, max_iters: Optional[int], cover_set: Set[_T]) -> Tuple[Dict[int, List[_T]], Dict[int, List[_T]]]:
  method = partial(sel_greedy_kld, target_dist=target_dist, cover_set=cover_set)

  selected_entries, rest_entries, covered_all, current_symbol_count = _get_utterances(
    already_added_entries=existing,
    available_entries=available,
    ignore_already_added=ignore_already_added,
    method=method,
    min_count_symbol=min_count_symbol,
    symbol_ids_total_set=symbol_ids_total_set,
    total_symbol_count=total_symbol_count,
    max_iters=max_iters,
  )

  return selected_entries, rest_entries


def get_greedy_utterances_count(existing: Dict[int, List[_T]], available: Dict[int, List[_T]], symbol_ids_total_set: Dict[int, List[int]], total_symbol_count: Optional[int], min_count_symbol: int, ignore_already_added: bool, max_iters: Optional[int], cover_set: Set[_T]) -> Tuple[Dict[int, List[_T]], Dict[int, List[_T]]]:
  method = partial(sel_greedy, cover_set=cover_set)

  selected_entries, rest_entries, covered_all, current_symbol_count = _get_utterances(
    already_added_entries=existing,
    available_entries=available,
    ignore_already_added=ignore_already_added,
    method=method,
    min_count_symbol=min_count_symbol,
    symbol_ids_total_set=symbol_ids_total_set,
    total_symbol_count=total_symbol_count,
    max_iters=max_iters,
  )

  return selected_entries, rest_entries
