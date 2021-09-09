from logging import getLogger
from typing import List, Tuple

import pandas as pd
from text_utils import get_ngrams
from text_utils.types import Symbols
from tts_preparation.core.data import PreparedDataList
from tts_preparation.core.helper import get_total_set
from tts_preparation.core.stats import (
    get_dist_among_other_symbols_df_of_all_symbols, get_occ_df_of_all_symbols,
    get_rel_occ_df_of_all_symbols, get_rel_uniform_distr_df_for_occs,
    get_rel_utter_occ_df_of_all_symbols, get_uniform_distr_df_for_occs,
    get_utter_occ_df_of_all_symbols)

FIRST_COL_NAME = "SYMBOL"


def get_ngram_stats_df(trainset: PreparedDataList, valset: PreparedDataList, testset: PreparedDataList, restset: PreparedDataList, n: int) -> pd.DataFrame:
  total_set = get_total_set(trainset, valset, testset, restset)
  logger = getLogger(__name__)
  logger.info(f"Getting all {n}-gram stats...")
  tot_symbols_list = [entry.symbols for entry in total_set.items()]
  tot_symbols_one_gram_list = [get_ngrams(symbols, n=n) for symbols in tot_symbols_list]
  symbol_order = list(
    sorted({one_gram for one_grams in tot_symbols_one_gram_list for one_gram in one_grams}))

  ngram_stats = _get_ngram_stats_df_core(
    symbol_order=symbol_order,
    trainset=trainset,
    valset=valset,
    testset=testset,
    restset=restset,
    n=n,
  )
  occurences_count_df, occurrences_percent_df, occurrences_distribution_percent_df, utterance_occurrences_count_df, utterance_occurrences_percent_df, uniform_occurrences_count_df, uniform_occurrences_percent_df = ngram_stats

  symbol_dfs = []
  symbol_dfs.append(occurences_count_df)
  symbol_dfs.append(occurrences_percent_df)
  symbol_dfs.append(occurrences_distribution_percent_df)
  symbol_dfs.append(utterance_occurrences_count_df)
  symbol_dfs.append(utterance_occurrences_percent_df)
  symbol_dfs.append(uniform_occurrences_count_df)
  symbol_dfs.append(uniform_occurrences_percent_df)

  for i in range(1, len(symbol_dfs)):
    symbol_dfs[i] = symbol_dfs[i].loc[:, symbol_dfs[i].columns != FIRST_COL_NAME]

  symbol_stats = pd.concat(
    symbol_dfs,
    axis=1,
    join='inner',
  )

  # symbol_stats = symbol_stats.round(decimals=2)
  symbol_stats = symbol_stats.sort_values(by='TOTAL_OCCURRENCES_COUNT', ascending=False)
  logger.info(symbol_stats)
  return symbol_stats


def _get_ngram_stats_df_core(symbol_order: List[Symbols], trainset: PreparedDataList, valset: PreparedDataList, testset: PreparedDataList, restset: PreparedDataList, n: int) -> Tuple[pd.DataFrame, ...]:
  logger = getLogger(__name__)
  logger.info(f"Get {n}-grams...")
  trn_symbols_list = [entry.symbols for entry in trainset.items()]
  val_symbols_list = [entry.symbols for entry in valset.items()]
  tst_symbols_list = [entry.symbols for entry in testset.items()]
  rst_symbols_list = [entry.symbols for entry in restset.items()]

  trn_symbols_one_gram = [get_ngrams(symbols, n=n) for symbols in trn_symbols_list]
  val_symbols_one_gram = [get_ngrams(symbols, n=n) for symbols in val_symbols_list]
  tst_symbols_one_gram = [get_ngrams(symbols, n=n) for symbols in tst_symbols_list]
  rst_symbols_one_gram = [get_ngrams(symbols, n=n) for symbols in rst_symbols_list]
  logger.info("Get stats...")

  occurences_count_df = get_occ_df_of_all_symbols(
    symbols=symbol_order,
    data_trn=trn_symbols_one_gram,
    data_val=val_symbols_one_gram,
    data_tst=tst_symbols_one_gram,
    data_rst=rst_symbols_one_gram,
  )
  occurences_count_df.columns = [FIRST_COL_NAME, 'TRAIN_OCCURRENCES_COUNT', 'VAL_OCCURRENCES_COUNT',
                                 'TEST_OCCURRENCES_COUNT', 'REST_OCCURRENCES_COUNT', 'TOTAL_OCCURRENCES_COUNT']
  print(occurences_count_df)

  occurrences_percent_df = get_rel_occ_df_of_all_symbols(occurences_count_df)
  occurrences_percent_df.columns = [FIRST_COL_NAME, 'TRAIN_OCCURRENCES_PERCENT', 'VAL_OCCURRENCES_PERCENT',
                                    'TEST_OCCURRENCES_PERCENT', 'REST_OCCURRENCES_PERCENT']
  print(occurrences_percent_df)

  occurrences_distribution_percent_df = get_dist_among_other_symbols_df_of_all_symbols(
    occs_df=occurences_count_df,
    data_trn=trn_symbols_one_gram,
    data_val=val_symbols_one_gram,
    data_tst=tst_symbols_one_gram,
    data_rst=rst_symbols_one_gram,
  )
  occurrences_distribution_percent_df.columns = [FIRST_COL_NAME, 'TRAIN_OCCURRENCES_DISTRIBUTION_PERCENT', 'VAL_OCCURRENCES_DISTRIBUTION_PERCENT',
                                                 'TEST_OCCURRENCES_DISTRIBUTION_PERCENT', 'REST_OCCURRENCES_DISTRIBUTION_PERCENT', 'TOTAL_OCCURRENCES_DISTRIBUTION_PERCENT']
  print(occurrences_distribution_percent_df)

  utterance_occurrences_count_df = get_utter_occ_df_of_all_symbols(
    symbols=symbol_order,
    data_trn=trn_symbols_one_gram,
    data_val=val_symbols_one_gram,
    data_tst=tst_symbols_one_gram,
    data_rst=rst_symbols_one_gram,
  )
  utterance_occurrences_count_df.columns = [FIRST_COL_NAME, 'TRAIN_UTTERANCE_OCCURRENCES_COUNT', 'VAL_UTTERANCE_OCCURRENCES_COUNT',
                                            'TEST_UTTERANCE_OCCURRENCES_COUNT', 'REST_UTTERANCE_OCCURRENCES_COUNT', 'TOTAL_UTTERANCE_OCCURRENCES_COUNT']
  print(utterance_occurrences_count_df)

  utterance_occurrences_percent_df = get_rel_utter_occ_df_of_all_symbols(
    utterance_occurrences_count_df)
  utterance_occurrences_percent_df.columns = [FIRST_COL_NAME, 'TRAIN_UTTERANCE_OCCURRENCES_PERCENT', 'VAL_UTTERANCE_OCCURRENCES_PERCENT',
                                              'TEST_UTTERANCE_OCCURRENCES_PERCENT', 'REST_UTTERANCE_OCCURRENCES_PERCENT']
  print(utterance_occurrences_percent_df)

  uniform_occurrences_count_df = get_uniform_distr_df_for_occs(
    symbols=symbol_order,
    occ_df=occurences_count_df,
  )
  uniform_occurrences_count_df.columns = [FIRST_COL_NAME, 'TRAIN_UNIFORM_OCCURRENCES_COUNT', 'VAL_UNIFORM_OCCURRENCES_COUNT',
                                          'TEST_UNIFORM_OCCURRENCES_COUNT', 'REST_UNIFORM_OCCURRENCES_COUNT', 'TOTAL_UNIFORM_OCCURRENCES_COUNT']
  print(uniform_occurrences_count_df)

  uniform_occurrences_percent_df = get_rel_uniform_distr_df_for_occs(
    symbols=symbol_order,
  )
  uniform_occurrences_percent_df.columns = [FIRST_COL_NAME, 'UNIFORM_OCCURRENCES_PERCENT']
  print(uniform_occurrences_percent_df)

  return occurences_count_df, occurrences_percent_df, occurrences_distribution_percent_df, utterance_occurrences_count_df, utterance_occurrences_percent_df, uniform_occurrences_count_df, uniform_occurrences_percent_df
