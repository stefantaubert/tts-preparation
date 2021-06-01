from typing import Dict, List, TypeVar

import numpy as np
import pandas as pd
from numpy.core.fromnumeric import mean
from numpy.core.numeric import Infinity, NaN

FIRST_COL = "SPEAKER"
_T = TypeVar('_T')
NOT_EXISTING = "-"


def get_duration_stats(speakers: List[str], data_trn: Dict[str, List[_T]], data_val: Dict[str, List[_T]], data_tst: Dict[str, List[_T]], data_rst: Dict[str, List[_T]], data_total: Dict[str, List[_T]]) -> pd.DataFrame:
  meta_dataset = get_meta_dict(speakers, data_trn, data_val, data_tst, data_rst, data_total)
  duration_df = get_duration_df(speakers, meta_dataset)
  rel_duration_df = get_rel_duration_df(duration_df)
  dist_df = get_dist_df(duration_df)
  min_df = get_min_df(speakers, meta_dataset)
  max_df = get_max_df(speakers, meta_dataset)
  mean_df = get_mean_df(speakers, meta_dataset)
  full_df = pd.concat([
      duration_df,
      rel_duration_df.loc[:, rel_duration_df.columns != FIRST_COL],
      dist_df.loc[:, dist_df.columns != FIRST_COL],
      min_df.loc[:, min_df.columns != FIRST_COL],
      max_df.loc[:, max_df.columns != FIRST_COL],
      mean_df.loc[:, mean_df.columns != FIRST_COL]
  ],
      axis=1,
      join='inner')
  return full_df


def get_mean_df(speakers: List[str], meta_dataset) -> pd.DataFrame:
  lines_of_df = get_mean_durations_for_every_speaker_for_all_sets(speakers, meta_dataset)
  df = pd.DataFrame(lines_of_df, columns=['SPEAKER', 'MEAN TRN', 'VAL', 'TST', 'RST', 'TOTAL'])
  last_line = mean_of_df(df)
  last_line.replace(0, NOT_EXISTING, inplace=True)
  df = df.append(last_line, ignore_index=True)
  df.iloc[-1, 0] = "all"
  return df


def mean_of_df(data: pd.DataFrame) -> pd.Series:
  data_without_hyphen = data.replace(NOT_EXISTING, NaN)
  means = data_without_hyphen.mean()
  means.replace(NaN, NOT_EXISTING, inplace=True)
  return means


def get_mean_durations_for_every_speaker_for_all_sets(speakers: List[str], dataset: Dict[str, List[List[_T]]]) -> List[List]:
  all_means = [get_mean_durations_for_one_speaker_for_all_sets(
    speaker, dataset[speaker]) for speaker in speakers]
  return all_means


def get_mean_durations_for_one_speaker_for_all_sets(speaker, durations_list: List[List[_T]]) -> List:
  means = [mean(durations) if durations != [0] else NOT_EXISTING for durations in durations_list]
  means.insert(0, speaker)
  return means


def get_max_df(speakers: List[str], meta_dataset) -> pd.DataFrame:
  lines_of_df = get_maximum_durations_for_every_speaker_for_all_sets(speakers, meta_dataset)
  df = pd.DataFrame(lines_of_df, columns=['SPEAKER', 'MAX TRN', 'VAL', 'TST', 'RST', 'TOTAL'])
  last_line = maximum_of_df(df)
  last_line.replace(0, NOT_EXISTING, inplace=True)
  df = df.append(last_line, ignore_index=True)
  df.iloc[-1, 0] = "all"
  return df


def maximum_of_df(data: pd.DataFrame) -> pd.Series:
  data_without_hyphen = data.replace(NOT_EXISTING, 0)
  maxs = data_without_hyphen.max()
  maxs.replace(0, NOT_EXISTING, inplace=True)
  return maxs


def get_maximum_durations_for_every_speaker_for_all_sets(speakers: List[str], dataset: Dict[str, List[List[_T]]]) -> List[List]:
  all_maxima = [get_maximum_durations_for_one_speaker_for_all_sets(
    speaker, dataset[speaker]) for speaker in speakers]
  return all_maxima


def get_maximum_durations_for_one_speaker_for_all_sets(speaker, durations_list: List[List[_T]]) -> List:
  maxs = [max(durations) if durations != [0] else NOT_EXISTING for durations in durations_list]
  maxs.insert(0, speaker)
  return maxs


def get_min_df(speakers: List[str], meta_dataset) -> pd.DataFrame:
  lines_of_df = get_minimum_durations_for_every_speaker_for_all_sets(speakers, meta_dataset)
  df = pd.DataFrame(lines_of_df, columns=['SPEAKER', 'MIN TRN', 'VAL', 'TST', 'RST', 'TOTAL'])
  last_line = minimum_of_df(df)
  df = df.append(last_line, ignore_index=True)
  df.iloc[-1, 0] = "all"
  return df


def minimum_of_df(data: pd.DataFrame) -> pd.Series:
  data_without_hyphen = data.replace(NOT_EXISTING, Infinity)
  mins = data_without_hyphen.min()
  mins.replace(Infinity, NOT_EXISTING, inplace=True)
  return mins


def get_minimum_durations_for_every_speaker_for_all_sets(speakers: List[str], dataset: Dict[str, List[List[_T]]]) -> List[List]:
  all_minima = [get_minimum_durations_for_one_speaker_for_all_sets(
    speaker, dataset[speaker]) for speaker in speakers]
  return all_minima


def get_minimum_durations_for_one_speaker_for_all_sets(speaker, durations_list: List[List[_T]]) -> List:
  mins = [min(durations) if durations != [0] else NOT_EXISTING for durations in durations_list]
  mins.insert(0, speaker)
  return mins


def get_dist_df(durations_df: pd.DataFrame) -> pd.DataFrame:
  durations_df.replace(NOT_EXISTING, 0, inplace=True)
  df = durations_df.iloc[:-1, 1:].copy()
  df.columns = ['DIST TRN', 'VAL', 'TST', 'RST', 'TOTAL']
  dataset_lengths = df.sum()
  df = 100 * df.div(dataset_lengths)
  df.insert(loc=0, column=FIRST_COL, value=durations_df.iloc[:-1, 0])
  last_line = df.sum()
  df = df.append(last_line, ignore_index=True)
  df.iloc[-1, 0] = "all"
  df.replace(0, NOT_EXISTING, inplace=True)
  df.replace(NaN, NOT_EXISTING, inplace=True)
  durations_df.replace(0, NOT_EXISTING, inplace=True)
  return df


def get_whole_dataset_duration(dataset: Dict[str, List[List[_T]]]) -> _T:
  duration_for_each_speaker = [sum(durations) for durations in list(dataset.values())]
  return sum(duration_for_each_speaker)


def get_rel_duration_df(durations_df: pd.DataFrame) -> pd.DataFrame:
  durations_df.replace(NOT_EXISTING, 0, inplace=True)
  df_as_row_wise_array = durations_df.to_numpy()
  df_lines = []
  for row in df_as_row_wise_array:
    rel_durations_list = get_relative_durations_for_all_sets(row[1:])
    rel_durations_list.insert(0, row[0])
    df_lines.append(rel_durations_list)
  df = pd.DataFrame(df_lines, columns=['SPEAKER', 'REL_DUR TRN', 'VAL', 'TST', 'RST'])
  df.replace(0, NOT_EXISTING, inplace=True)
  durations_df.replace(0, NOT_EXISTING, inplace=True)
  return df


def get_relative_durations_for_all_sets(duration_list: List[_T]) -> List:
  if duration_list[-1] == 0:  # falls ein Sprecher in keinem Set vorkommt
    return [0] * (len(duration_list) - 1)
  rel_durations = 100 * np.array(duration_list[:-1]) / duration_list[-1]
  return rel_durations.tolist()


def get_duration_df(speakers: List[str], meta_dataset: Dict[str, List[List[_T]]]) -> pd.DataFrame:
  lines_of_df = get_duration_sums_for_every_speaker_for_all_sets(speakers, meta_dataset)
  df = pd.DataFrame(lines_of_df, columns=['SPEAKER', 'DUR TRN', 'VAL', 'TST', 'RST', 'TOTAL'])
  last_line = df.sum()
  df = df.append(last_line, ignore_index=True)
  df.iloc[-1, 0] = "all"
  df.replace(0, NOT_EXISTING, inplace=True)
  return df


def get_duration_sums_for_every_speaker_for_all_sets(speakers: List[str], dataset: Dict[str, List[List[_T]]]) -> List[List]:
  all_duration_sums = [get_duration_sums_for_one_speaker_for_all_sets(
    speaker, dataset[speaker]) for speaker in speakers]
  return all_duration_sums


def get_duration_sums_for_one_speaker_for_all_sets(speaker, durations_list: List[List[_T]]) -> List:
  duration_sums = [sum(durations) for durations in durations_list]
  duration_sums.insert(0, speaker)
  return duration_sums


def get_meta_dict(speakers: List[str], data_trn: Dict[str, List[_T]], data_val: Dict[str, List[_T]], data_tst: Dict[str, List[_T]], data_rst: Dict[str, List[_T]], data_total: Dict[str, List[_T]]) -> Dict[str, List[List[_T]]]:
  meta_dict = {speaker: get_duration_values_for_key(
    speaker, data_trn, data_val, data_tst, data_rst, data_total) for speaker in speakers}
  return meta_dict


def get_duration_values_for_key(speaker: str, data_trn: Dict[str, List[_T]], data_val: Dict[str, List[_T]], data_tst: Dict[str, List[_T]], data_rst: Dict[str, List[_T]], data_total: Dict[str, List[_T]]) -> List[_T]:
  values = [duration_or_zero(speaker, data)
            for data in [data_trn, data_val, data_tst, data_rst, data_total]]
  return values


def duration_or_zero(speaker: str, data: Dict[str, List[_T]]) -> List[_T]:
  if speaker in data.keys():
    return data[speaker]
  return [0]


_T = TypeVar('_T')


def get_ngram_stats(symbols: List[_T], data_trn: List[List[_T]], data_val: List[List[_T]], data_tst: List[List[_T]], data_rst: List[List[_T]]) -> pd.DataFrame:
  occ_df = get_occ_df_of_all_symbols(symbols, data_trn, data_val, data_tst, data_rst)
  rel_occ_df = get_rel_occ_df_of_all_symbols(occ_df)
  dist_df = get_dist_among_other_symbols_df_of_all_symbols(
    occ_df, data_trn, data_val, data_tst, data_rst)
  utter_occ_df = get_utter_occ_df_of_all_symbols(symbols, data_trn, data_val, data_tst, data_rst)
  rel_utter_occ_df = get_rel_utter_occ_df_of_all_symbols(utter_occ_df)
  uni_dist_df = get_uniform_distr_df_for_occs(symbols, occ_df)
  rel_uni_dist_df = get_rel_uniform_distr_df_for_occs(symbols)
  full_df = pd.concat([
      occ_df,
      rel_occ_df.loc[:, rel_occ_df.columns != "SYMB"],
      dist_df.loc[:, dist_df.columns != "SYMB"],
      utter_occ_df.loc[:, utter_occ_df.columns != "SYMB"],
      rel_utter_occ_df.loc[:, rel_utter_occ_df.columns != "SYMB"],
      uni_dist_df.loc[:, uni_dist_df.columns != "SYMB"],
      rel_uni_dist_df.loc[:, rel_uni_dist_df.columns != "SYMB"]
  ],
      axis=1,
      join='inner')
  return full_df


def get_rel_uniform_distr_df_for_occs(symbols) -> pd.DataFrame:
  percentage = 100 / len(symbols)
  lines_of_df = [[symb, percentage] for symb in symbols]
  lines_of_df.append(['all', 100])
  df = pd.DataFrame(lines_of_df, columns=['SYMB', 'UNI_DISTR_%'])
  return df


def get_uniform_distr_df_for_occs(symbols: List[_T], occ_df: pd.DataFrame) -> pd.DataFrame:
  uni_distr_df = df_with_uni_distr(symbols, occ_df)
  uni_distr_df.columns = ['SYMB', 'OCC_UNI_DISTR TRN', 'VAL', 'TST', 'RST', 'TOTAL']
  return uni_distr_df


def df_with_uni_distr(symbols: List[_T], df: pd.DataFrame) -> pd.DataFrame:
  number_of_all_possible_symbols = len(symbols)
  assert number_of_all_possible_symbols != 0
  last_line_of_df = df.iloc[-1, 1:]
  uni_distr_df = df.copy()
  uniform_distr_line = last_line_of_df / number_of_all_possible_symbols
  uni_distr_df.iloc[:-1, 1:] = [uniform_distr_line] * number_of_all_possible_symbols
  return uni_distr_df


def get_rel_utter_occ_df_of_all_symbols(utter_occs_df: pd.DataFrame) -> pd.DataFrame:
  df_as_row_wise_array = utter_occs_df.to_numpy()
  df_lines = []
  for row in df_as_row_wise_array:
    rel_utter_occ_list = get_relative_utter_occs_for_all_sets(row[1:])
    rel_utter_occ_list.insert(0, row[0])
    df_lines.append(rel_utter_occ_list)
  df = pd.DataFrame(df_lines, columns=['SYMB', 'REL_UTT TRN', 'VAL', 'TST', 'RST'])
  return df


def get_relative_utter_occs_for_all_sets(utter_occs_list: List[int]) -> List[float]:
  if utter_occs_list[-1] == 0:
    return [0] * (len(utter_occs_list) - 1)
  relative_utter_occs = 100 * np.array(utter_occs_list[:-1]) / utter_occs_list[-1]
  return relative_utter_occs.tolist()


def get_utter_occ_df_of_all_symbols(symbols: List[_T], data_trn: List[List[_T]], data_val: List[List[_T]], data_tst: List[List[_T]], data_rst: List[List[_T]]) -> pd.DataFrame:
  df_lines = []
  for symb in symbols:
    utter_occ_list = get_utter_occs_for_all_sets(symb, data_trn, data_val, data_tst, data_rst)
    utter_occ_list.insert(0, symb)
    df_lines.append(utter_occ_list)
  df = pd.DataFrame(df_lines, columns=['SYMB', 'UTT TRN', 'VAL', 'TST', 'RST', 'TOTAL'])
  df = add_all_line_as_sum_of_previous_lines(df)
  return df


def get_utter_occs_for_all_sets(symb: _T, data_trn: List[List[_T]], data_val: List[List[_T]], data_tst: List[List[_T]], data_rst: List[List[_T]]) -> List[int]:
  utter_occs = [get_utter_occs_of_symbol_in_one_set(symb, dataset) for dataset in [
      data_trn, data_val, data_tst, data_rst]]
  utter_occs.append(sum(utter_occs))
  return utter_occs


def get_utter_occs_of_symbol_in_one_set(symb: _T, dataset: List[List[_T]]) -> int:
  symb_is_in_sublist = [symb in sublist for sublist in dataset]
  return sum(symb_is_in_sublist)


def get_dist_among_other_symbols_df_of_all_symbols(occs_df: pd.DataFrame, data_trn: List[List[_T]], data_val: List[List[_T]], data_tst: List[List[_T]], data_rst: List[List[_T]]) -> pd.DataFrame:
  df_as_row_wise_array = occs_df.to_numpy()
  df_lines = []
  total_symb_numbers = get_total_numbers_of_symbols_for_all_sets(
    data_trn, data_val, data_tst, data_rst)
  for row in df_as_row_wise_array:
    dist_list = get_dists_among_other_symbols(row[1:], total_symb_numbers)
    dist_list.insert(0, row[0])
    df_lines.append(dist_list)
  df = pd.DataFrame(df_lines, columns=['SYMB', 'DIST TRN', 'VAL', 'TST', 'RST', 'TOTAL'])
  return df


def get_dists_among_other_symbols(occs_of_symb: List[int], total_numb_of_symbs: List[int]) -> List[float]:
  division_possible = [number != 0 for number in total_numb_of_symbs]
  dists = np.divide(np.array(occs_of_symb), np.array(total_numb_of_symbs), where=division_possible)
  dist_is_None = [not value for value in division_possible]
  dists[dist_is_None] = 0
  dists *= 100
  return dists.tolist()


def get_total_numbers_of_symbols_for_all_sets(data_trn: List[List[_T]], data_val: List[List[_T]], data_tst: List[List[_T]], data_rst: List[List[_T]]) -> List[int]:
  total_symb_numbers = [total_number_of_symbols_in_dataset(
    dataset) for dataset in [data_trn, data_val, data_tst, data_rst]]
  total_symb_numbers.append(sum(total_symb_numbers))
  return total_symb_numbers


def total_number_of_symbols_in_dataset(dataset: List[List[_T]]) -> int:
  lens_of_sublists = [len(sublist) for sublist in dataset]
  return sum(lens_of_sublists)


def get_rel_occ_df_of_all_symbols(occs_df: pd.DataFrame) -> pd.DataFrame:
  df_as_row_wise_array = occs_df.to_numpy()
  df_lines = []
  for row in df_as_row_wise_array:
    rel_occ_list = get_relative_occs_for_all_sets(row[1:])
    rel_occ_list.insert(0, row[0])
    df_lines.append(rel_occ_list)
  df = pd.DataFrame(df_lines, columns=['SYMB', 'REL_OCC TRN', 'VAL', 'TST', 'RST'])
  return df


def get_relative_occs_for_all_sets(occs_list: np.array) -> List[float]:
  if occs_list[-1] == 0:
    return [0] * (len(occs_list) - 1)
  relative_occs = 100 * np.array(occs_list[:-1]) / occs_list[-1]
  return relative_occs.tolist()


def get_occ_df_of_all_symbols(symbols: List[_T], data_trn: List[List[_T]], data_val: List[List[_T]], data_tst: List[List[_T]], data_rst: List[List[_T]]) -> pd.DataFrame:
  df_lines = []
  for symb in symbols:
    occ_list = get_occs_for_all_sets(symb, data_trn, data_val, data_tst, data_rst)
    occ_list.insert(0, symb)
    df_lines.append(occ_list)
  df = pd.DataFrame(df_lines, columns=['SYMB', 'OCC TRN', 'VAL', 'TST', 'RST', 'TOTAL'])
  df = add_all_line_as_sum_of_previous_lines(df)
  return df


def add_all_line_as_sum_of_previous_lines(df: pd.DataFrame) -> pd.DataFrame:
  last_line = df.sum()
  df = df.append(last_line, ignore_index=True)
  df.iloc[-1, 0] = "all"
  return df


def get_occs_for_all_sets(symb: _T, data_trn: List[List[_T]], data_val: List[List[_T]], data_tst: List[List[_T]], data_rst: List[List[_T]]) -> List[int]:
  occs = [get_occs_of_symb_in_one_set(symb, dataset)
          for dataset in [data_trn, data_val, data_tst, data_rst]]
  occs.append(sum(occs))
  return occs


def get_occs_of_symb_in_one_set(symb: _T, dataset: List[List[_T]]) -> int:
  occs_of_symb_in_sublists = [sublist.count(symb) for sublist in dataset]
  return sum(occs_of_symb_in_sublists)
