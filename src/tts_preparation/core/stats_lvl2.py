from logging import getLogger
from math import ceil

import numpy as np
import pandas as pd
from tts_preparation.core.data import DatasetType


def get_one_gram_stats(stats: pd.DataFrame, ds: DatasetType):
  logger = getLogger(__name__)
  logger.info(f"Stats {str(ds)} onegrams")
  get_stats(stats, ds)


def get_two_gram_stats(stats: pd.DataFrame, ds: DatasetType):
  logger = getLogger(__name__)
  logger.info(f"Stats {str(ds)} twograms")
  get_stats(stats, ds)


def get_three_gram_stats(stats: pd.DataFrame, ds: DatasetType):
  logger = getLogger(__name__)
  logger.info(f"Stats {str(ds)} threegrams")
  get_stats(stats, ds)


def column_title(ds: DatasetType) -> str:
  if ds == DatasetType.TEST:
    return "TEST"
  if ds == DatasetType.TRAINING:
    return "TRAIN"
  if ds == DatasetType.VALIDATION:
    return "VAL"


def get_stats(stats: pd.DataFrame, ds: DatasetType):
  logger = getLogger(__name__)
  title = column_title(ds)
  top_n_range = range(10, 110, 10)
  stats.sort_values(by=['TOTAL_OCCURRENCES_COUNT'], inplace=True, ascending=False)
  ngrams_ordered_by_occ = list(stats["SYMBOL"])
  ngrams_ordered_by_occ.remove("all")

  filtered_dfs = {}
  for top_n in top_n_range:
    top_ngrams_count = ceil(top_n / 100 * len(ngrams_ordered_by_occ))
    top_ngrams = set(ngrams_ordered_by_occ[:top_ngrams_count])
    filt = stats[stats["SYMBOL"].isin(top_ngrams)]
    filtered_dfs[top_n] = filt

  for top_n, filt in filtered_dfs.items():
    test_symbol_coverage_dict = {}
    for i, row in filt.iterrows():
      symbol = row["SYMBOL"]
      test_occ = row[f"{title}_OCCURRENCES_COUNT"]
      test_symbol_coverage_dict[symbol] = 1 if test_occ > 0 else 0

    test_symbol_cov = sum(test_symbol_coverage_dict.values()) / len(test_symbol_coverage_dict) * 100
    logger.info(f"Coverage top {top_n}%: {test_symbol_cov:.2f}%")

  for top_n, filt in filtered_dfs.items():
    x = abs(filt[f"{title}_OCCURRENCES_DISTRIBUTION_PERCENT"] -
            filt["TOTAL_OCCURRENCES_DISTRIBUTION_PERCENT"])
    deviation_to_total_set = np.mean(x)
    logger.info(f"Mean deviation to total distribution top {top_n}%: {deviation_to_total_set:.2f}%")

  for top_n, filt in filtered_dfs.items():
    x = abs(filt[f"{title}_OCCURRENCES_DISTRIBUTION_PERCENT"] -
            filt["UNIFORM_OCCURRENCES_PERCENT"])
    uniform_deviation_mean = np.mean(x)
    logger.info(
      f"Mean deviation to uniform distribution top {top_n}%: {uniform_deviation_mean:.2f}%")
