from logging import getLogger
from typing import Dict, List, Tuple

import pandas as pd
from text_utils import SymbolIdDict
from text_utils.speakers_dict import SpeakersDict
from text_utils.types import Speaker, Speakers
from tts_preparation.core.data import PreparedDataList, get_speaker_wise
from tts_preparation.core.helper import (get_percent_str, get_shard_size,
                                         get_total_set)
from tts_preparation.core.stats import (
    get_dist_among_other_symbols_df_of_all_symbols, get_dist_df,
    get_duration_df, get_max_df, get_mean_df, get_meta_dict, get_min_df,
    get_occ_df_of_all_symbols, get_rel_duration_df,
    get_rel_occ_df_of_all_symbols)


def log_general_stats(trainset: PreparedDataList, valset: PreparedDataList, testset: PreparedDataList, restset: PreparedDataList, data: PreparedDataList):
  logger = getLogger(__name__)
  total_set = get_total_set(trainset, valset, testset, restset)
  len_except_rest = len(trainset) + len(testset) + len(valset)
  logger.info(
    f"Size training set: {len(trainset)} ({trainset.total_duration_s/60:.2f}m) --> {get_percent_str(len(trainset), len(total_set))} ({get_percent_str(len(trainset), len_except_rest)})")
  logger.info(
    f"Size validation set: {len(valset)} ({valset.total_duration_s/60:.2f}m) --> {get_percent_str(len(valset), len(total_set))} ({get_percent_str(len(valset), len_except_rest)})")
  logger.info(
    f"Size test set: {len(testset)} ({testset.total_duration_s/60:.2f}m) --> {get_percent_str(len(testset), len(total_set))} ({get_percent_str(len(testset), len_except_rest)})")
  logger.info(
    f"Size rest set: {len(restset)} ({restset.total_duration_s/60:.2f}m) --> {get_percent_str(len(restset), len(total_set))}")
  logger.info(f"Total: {len(total_set)} ({total_set.total_duration_s/60:.2f}m)")
  logger.info(f"Original set: {len(data)} ({data.total_duration_s/60:.2f}m)")
  logger.info(f"Something lost: {'no' if len(total_set) == len(data) else 'yes!'}")


def get_speaker_stats(symbols: SymbolIdDict, speakers: SpeakersDict, trainset: PreparedDataList, valset: PreparedDataList, testset: PreparedDataList, restset: PreparedDataList):
  speaker_order = list(sorted(speakers.get_all_speakers()))
  chars_dfs, chars = _get_chars_stats(
    speaker_order=speaker_order,
    trainset=trainset,
    valset=valset,
    testset=testset,
    restset=restset,
  )
  chars_sum_count_df, chars_sum_percent_df, chars_sum_distribution_percent_df, chars_min_count_df, chars_max_count_df, chars_mean_count_df = chars_dfs

  shards_dfs = _get_shards_stats(
    speaker_order=speaker_order,
    symbols=symbols,
    chars=chars,
  )
  shards_sum_count_df, shards_sum_percent_df, shards_sum_distribution_percent_df, shards_min_count_df, shards_max_count_df, shards_mean_count_df = shards_dfs

  utt_dfs = _get_speaker_occ_stats(
    speaker_order=speaker_order,
    trainset=trainset,
    valset=valset,
    testset=testset,
    restset=restset,
  )
  utterances_count_df, utterances_percent_df, utterances_distribution_percent_df = utt_dfs

  duration_stats = _get_speaker_duration_stats(
    speaker_order=speaker_order,
    trainset=trainset,
    valset=valset,
    testset=testset,
    restset=restset,
  )
  duration_sum_s_df, duration_sum_percent_df, duration_sum_distribution_percent_df, duration_min_s_df, duration_max_s_df, duration_mean_s_df = duration_stats

  speaker_dfs = []
  speaker_dfs.append(chars_sum_count_df)
  speaker_dfs.append(chars_sum_percent_df)
  speaker_dfs.append(chars_sum_distribution_percent_df)
  speaker_dfs.append(chars_min_count_df)
  speaker_dfs.append(chars_max_count_df)
  speaker_dfs.append(chars_mean_count_df)

  speaker_dfs.append(shards_sum_count_df)
  speaker_dfs.append(shards_sum_percent_df)
  speaker_dfs.append(shards_sum_distribution_percent_df)
  speaker_dfs.append(shards_min_count_df)
  speaker_dfs.append(shards_max_count_df)
  speaker_dfs.append(shards_mean_count_df)

  speaker_dfs.append(utterances_count_df)
  speaker_dfs.append(utterances_percent_df)
  speaker_dfs.append(utterances_distribution_percent_df)

  speaker_dfs.append(duration_sum_s_df)
  speaker_dfs.append(duration_sum_percent_df)
  speaker_dfs.append(duration_sum_distribution_percent_df)
  speaker_dfs.append(duration_min_s_df)
  speaker_dfs.append(duration_max_s_df)
  speaker_dfs.append(duration_mean_s_df)

  for i in range(1, len(speaker_dfs)):
    speaker_dfs[i] = speaker_dfs[i].loc[:, speaker_dfs[i].columns != "SPEAKER_NAME"]

  speaker_stats = pd.concat(
    speaker_dfs,
    axis=1,
    join='inner',
  )

  #speaker_stats = speaker_stats.round(decimals=2)
  print(speaker_stats)
  return speaker_stats


def _get_chars_stats(speaker_order: Speakers, trainset: PreparedDataList, valset: PreparedDataList, testset: PreparedDataList, restset: PreparedDataList) -> Tuple[Tuple[pd.DataFrame, ...], Tuple[Dict[Speaker, List[int]], ...]]:
  total_set = get_total_set(trainset, valset, testset, restset)
  trn_speaker = get_speaker_wise(trainset)
  val_speaker = get_speaker_wise(valset)
  tst_speaker = get_speaker_wise(testset)
  rst_speaker = get_speaker_wise(restset)
  tot_speaker = get_speaker_wise(total_set)

  trn_chars = {}
  val_chars = {}
  tst_chars = {}
  rst_chars = {}
  tot_chars = {}

  for speaker in speaker_order:
    if speaker in trn_speaker:
      trn_chars[speaker] = [len(entry.symbols)
                            for entry in trn_speaker[speaker].items()]
    if speaker in val_speaker:
      val_chars[speaker] = [len(entry.symbols)
                            for entry in val_speaker[speaker].items()]
    if speaker in tst_speaker:
      tst_chars[speaker] = [len(entry.symbols)
                            for entry in tst_speaker[speaker].items()]
    if speaker in rst_speaker:
      rst_chars[speaker] = [len(entry.symbols)
                            for entry in rst_speaker[speaker].items()]
    if speaker in tot_speaker:
      tot_chars[speaker] = [len(entry.symbols)
                            for entry in tot_speaker[speaker].items()]

  meta_dataset = get_meta_dict(
    speakers=speaker_order,
    data_trn=trn_chars,
    data_val=val_chars,
    data_rst=rst_chars,
    data_tst=tst_chars,
    data_total=tot_chars,
  )

  chars_sum_count_df = get_duration_df(
    speakers=speaker_order,
    meta_dataset=meta_dataset,
  )
  chars_sum_count_df.columns = ['SPEAKER_NAME', 'TRAIN_CHARS_SUM_COUNT', 'VAL_CHARS_SUM_COUNT',
                                'TEST_CHARS_SUM_COUNT', 'REST_CHARS_SUM_COUNT', 'TOTAL_CHARS_SUM_COUNT']
  print(chars_sum_count_df)

  chars_sum_percent_df = get_rel_duration_df(chars_sum_count_df)
  chars_sum_percent_df.columns = ['SPEAKER_NAME', 'TRAIN_CHARS_SUM_PERCENT', 'VAL_CHARS_SUM_PERCENT',
                                  'TEST_CHARS_SUM_PERCENT', 'REST_CHARS_SUM_PERCENT']
  print(chars_sum_percent_df)

  chars_sum_distribution_percent_df = get_dist_df(
    durations_df=chars_sum_count_df,
  )
  chars_sum_distribution_percent_df.columns = ['SPEAKER_NAME', 'TRAIN_CHARS_SUM_DISTRIBUTION_PERCENT', 'VAL_CHARS_DISTRIBUTION_PERCENT',
                                               'TEST_CHARS_DISTRIBUTION_PERCENT', 'REST_CHARS_DISTRIBUTION_PERCENT', 'TOTAL_CHARS_DISTRIBUTION_PERCENT']
  print(chars_sum_distribution_percent_df)

  chars_min_count_df = get_min_df(
    speakers=speaker_order,
    meta_dataset=meta_dataset,
  )
  chars_min_count_df.columns = ['SPEAKER_NAME', 'TRAIN_CHARS_MIN_COUNT', 'VAL_CHARS_MIN_COUNT',
                                'TEST_CHARS_MIN_COUNT', 'REST_CHARS_MIN_COUNT', 'TOTAL_CHARS_MIN_COUNT']
  print(chars_min_count_df)

  chars_max_count_df = get_max_df(
    speakers=speaker_order,
    meta_dataset=meta_dataset,
  )
  chars_max_count_df.columns = ['SPEAKER_NAME', 'TRAIN_CHARS_MAX_COUNT', 'VAL_CHARS_MAX_COUNT',
                                'TEST_CHARS_MAX_COUNT', 'REST_CHARS_MAX_COUNT', 'TOTAL_CHARS_MAX_COUNT']
  print(chars_max_count_df)

  chars_mean_count_df = get_mean_df(
    speakers=speaker_order,
    meta_dataset=meta_dataset,
  )
  chars_mean_count_df.columns = ['SPEAKER_NAME', 'TRAIN_CHARS_MEAN_COUNT', 'VAL_CHARS_MEAN_COUNT',
                                 'TEST_CHARS_MEAN_COUNT', 'REST_CHARS_MEAN_COUNT', 'TOTAL_CHARS_MEAN_COUNT']
  print(chars_mean_count_df)

  dfs = (chars_sum_count_df, chars_sum_percent_df, chars_sum_distribution_percent_df,
         chars_min_count_df, chars_max_count_df, chars_mean_count_df)
  chars = (trn_chars, val_chars, tst_chars, rst_chars, tot_chars)
  return dfs, chars


def _get_shards_stats(speaker_order: List[str], symbols: SymbolIdDict, chars: Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]]]) -> Tuple[pd.DataFrame, ...]:
  shard_size = get_shard_size(symbols)
  trn_chars, val_chars, tst_chars, rst_chars, tot_chars = chars
  trn_shards = {k: [x / shard_size for x in v] for k, v in trn_chars.items()}
  val_shards = {k: [x / shard_size for x in v] for k, v in val_chars.items()}
  tst_shards = {k: [x / shard_size for x in v] for k, v in tst_chars.items()}
  rst_shards = {k: [x / shard_size for x in v] for k, v in rst_chars.items()}
  tot_shards = {k: [x / shard_size for x in v] for k, v in tot_chars.items()}

  meta_dataset = get_meta_dict(
    speakers=speaker_order,
    data_trn=trn_shards,
    data_val=val_shards,
    data_rst=rst_shards,
    data_tst=tst_shards,
    data_total=tot_shards,
  )

  shards_sum_count_df = get_duration_df(
    speakers=speaker_order,
    meta_dataset=meta_dataset,
  )
  shards_sum_count_df.columns = ['SPEAKER_NAME', 'TRAIN_SHARDS_SUM_COUNT', 'VAL_SHARDS_SUM_COUNT',
                                 'TEST_SHARDS_SUM_COUNT', 'REST_SHARDS_SUM_COUNT', 'TOTAL_SHARDS_SUM_COUNT']
  print(shards_sum_count_df)

  shards_sum_percent_df = get_rel_duration_df(shards_sum_count_df)
  shards_sum_percent_df.columns = ['SPEAKER_NAME', 'TRAIN_SHARDS_SUM_PERCENT', 'VAL_SHARDS_SUM_PERCENT',
                                   'TEST_SHARDS_SUM_PERCENT', 'REST_SHARDS_SUM_PERCENT']
  print(shards_sum_percent_df)

  shards_sum_distribution_percent_df = get_dist_df(
    durations_df=shards_sum_count_df,
  )
  shards_sum_distribution_percent_df.columns = ['SPEAKER_NAME', 'TRAIN_SHARDS_SUM_DISTRIBUTION_PERCENT', 'VAL_SHARDS_DISTRIBUTION_PERCENT',
                                                'TEST_SHARDS_DISTRIBUTION_PERCENT', 'REST_SHARDS_DISTRIBUTION_PERCENT', 'TOTAL_SHARDS_DISTRIBUTION_PERCENT']
  print(shards_sum_distribution_percent_df)

  shards_min_count_df = get_min_df(
    speakers=speaker_order,
    meta_dataset=meta_dataset,
  )
  shards_min_count_df.columns = ['SPEAKER_NAME', 'TRAIN_SHARDS_MIN_COUNT', 'VAL_SHARDS_MIN_COUNT',
                                 'TEST_SHARDS_MIN_COUNT', 'REST_SHARDS_MIN_COUNT', 'TOTAL_SHARDS_MIN_COUNT']
  print(shards_min_count_df)

  shards_max_count_df = get_max_df(
    speakers=speaker_order,
    meta_dataset=meta_dataset,
  )
  shards_max_count_df.columns = ['SPEAKER_NAME', 'TRAIN_SHARDS_MAX_COUNT', 'VAL_SHARDS_MAX_COUNT',
                                 'TEST_SHARDS_MAX_COUNT', 'REST_SHARDS_MAX_COUNT', 'TOTAL_SHARDS_MAX_COUNT']
  print(shards_max_count_df)

  shards_mean_count_df = get_mean_df(
    speakers=speaker_order,
    meta_dataset=meta_dataset,
  )
  shards_mean_count_df.columns = ['SPEAKER_NAME', 'TRAIN_SHARDS_MEAN_COUNT', 'VAL_SHARDS_MEAN_COUNT',
                                  'TEST_SHARDS_MEAN_COUNT', 'REST_SHARDS_MEAN_COUNT', 'TOTAL_SHARDS_MEAN_COUNT']
  print(shards_mean_count_df)

  return shards_sum_count_df, shards_sum_percent_df, shards_sum_distribution_percent_df, shards_min_count_df, shards_max_count_df, shards_mean_count_df


def _get_speaker_occ_stats(speaker_order: Speakers, trainset: PreparedDataList, valset: PreparedDataList, testset: PreparedDataList, restset: PreparedDataList) -> Tuple[pd.DataFrame, ...]:
  trn_speakers = [[x.speaker_name] for x in trainset.items()]
  val_speakers = [[x.speaker_name] for x in valset.items()]
  tst_speakers = [[x.speaker_name] for x in testset.items()]
  rst_speakers = [[x.speaker_name] for x in restset.items()]

  utterances_count_df = get_occ_df_of_all_symbols(
    symbols=speaker_order,
    data_trn=trn_speakers,
    data_val=val_speakers,
    data_tst=tst_speakers,
    data_rst=rst_speakers,
  )
  utterances_count_df.columns = ['SPEAKER_NAME', 'TRAIN_UTTERANCES_COUNT', 'VAL_UTTERANCES_COUNT',
                                 'TEST_UTTERANCES_COUNT', 'REST_UTTERANCES_COUNT', 'TOTAL_UTTERANCES_COUNT']
  print(utterances_count_df)

  utterances_percent_df = get_rel_occ_df_of_all_symbols(utterances_count_df)
  utterances_percent_df.columns = ['SPEAKER_NAME', 'TRAIN_UTTERANCES_PERCENT', 'VAL_UTTERANCES_PERCENT',
                                   'TEST_UTTERANCES_PERCENT', 'REST_UTTERANCES_PERCENT']
  print(utterances_percent_df)

  utterances_distribution_percent_df = get_dist_among_other_symbols_df_of_all_symbols(
    occs_df=utterances_count_df,
    data_trn=trn_speakers,
    data_val=val_speakers,
    data_tst=tst_speakers,
    data_rst=rst_speakers,
  )
  utterances_distribution_percent_df.columns = ['SPEAKER_NAME', 'TRAIN_UTTERANCES_DISTRIBUTION_PERCENT', 'VAL_UTTERANCES_DISTRIBUTION_PERCENT',
                                                'TEST_UTTERANCES_DISTRIBUTION_PERCENT', 'REST_UTTERANCES_DISTRIBUTION_PERCENT', 'TOTAL_UTTERANCES_DISTRIBUTION_PERCENT']
  print(utterances_distribution_percent_df)

  return utterances_count_df, utterances_percent_df, utterances_distribution_percent_df


def _get_speaker_duration_stats(speaker_order: Speakers, trainset: PreparedDataList, valset: PreparedDataList, testset: PreparedDataList, restset: PreparedDataList) -> Tuple[pd.DataFrame, ...]:
  total_set = get_total_set(trainset, valset, testset, restset)
  trn_speaker = get_speaker_wise(trainset)
  val_speaker = get_speaker_wise(valset)
  tst_speaker = get_speaker_wise(testset)
  rst_speaker = get_speaker_wise(restset)
  tot_speaker = get_speaker_wise(total_set)
  trn_durations = {}
  val_durations = {}
  tst_durations = {}
  rst_durations = {}
  tot_durations = {}

  for speaker in speaker_order:
    if speaker in trn_speaker:
      trn_durations[speaker] = [entry.duration_s for entry in trn_speaker[speaker].items()]
    if speaker in val_speaker:
      val_durations[speaker] = [entry.duration_s for entry in val_speaker[speaker].items()]
    if speaker in tst_speaker:
      tst_durations[speaker] = [entry.duration_s for entry in tst_speaker[speaker].items()]
    if speaker in rst_speaker:
      rst_durations[speaker] = [entry.duration_s for entry in rst_speaker[speaker].items()]
    if speaker in tot_speaker:
      tot_durations[speaker] = [entry.duration_s for entry in tot_speaker[speaker].items()]

  meta_dataset = get_meta_dict(
    speakers=speaker_order,
    data_trn=trn_durations,
    data_val=val_durations,
    data_rst=rst_durations,
    data_tst=tst_durations,
    data_total=tot_durations,
  )

  duration_sum_s_df = get_duration_df(
    speakers=speaker_order,
    meta_dataset=meta_dataset,
  )
  duration_sum_s_df.columns = ['SPEAKER_NAME', 'TRAIN_DURATION_SUM_S', 'VAL_DURATION_SUM_S',
                               'TEST_DURATION_SUM_S', 'REST_DURATION_SUM_S', 'TOTAL_DURATION_SUM_S']
  print(duration_sum_s_df)

  duration_sum_percent_df = get_rel_duration_df(duration_sum_s_df)
  duration_sum_percent_df.columns = ['SPEAKER_NAME', 'TRAIN_DURATION_SUM_PERCENT', 'VAL_DURATION_SUM_PERCENT',
                                     'TEST_DURATION_SUM_PERCENT', 'REST_DURATION_SUM_PERCENT']
  print(duration_sum_percent_df)

  duration_sum_distribution_percent_df = get_dist_df(
    durations_df=duration_sum_s_df,
  )
  duration_sum_distribution_percent_df.columns = ['SPEAKER_NAME', 'TRAIN_DURATION_SUM_DISTRIBUTION_PERCENT', 'VAL_DURATION_DISTRIBUTION_PERCENT',
                                                  'TEST_DURATION_DISTRIBUTION_PERCENT', 'REST_DURATION_DISTRIBUTION_PERCENT', 'TOTAL_DURATION_DISTRIBUTION_PERCENT']
  print(duration_sum_distribution_percent_df)

  duration_min_s_df = get_min_df(
    speakers=speaker_order,
    meta_dataset=meta_dataset,
  )
  duration_min_s_df.columns = ['SPEAKER_NAME', 'TRAIN_DURATION_MIN_S', 'VAL_DURATION_MIN_S',
                               'TEST_DURATION_MIN_S', 'REST_DURATION_MIN_S', 'TOTAL_DURATION_MIN_S']
  print(duration_min_s_df)

  duration_max_s_df = get_max_df(
    speakers=speaker_order,
    meta_dataset=meta_dataset,
  )
  duration_max_s_df.columns = ['SPEAKER_NAME', 'TRAIN_DURATION_MAX_S', 'VAL_DURATION_MAX_S',
                               'TEST_DURATION_MAX_S', 'REST_DURATION_MAX_S', 'TOTAL_DURATION_MAX_S']
  print(duration_max_s_df)

  duration_mean_s_df = get_mean_df(
    speakers=speaker_order,
    meta_dataset=meta_dataset,
  )
  duration_mean_s_df.columns = ['SPEAKER_NAME', 'TRAIN_DURATION_MEAN_S', 'VAL_DURATION_MEAN_S',
                                'TEST_DURATION_MEAN_S', 'REST_DURATION_MEAN_S', 'TOTAL_DURATION_MEAN_S']
  print(duration_mean_s_df)

  return duration_sum_s_df, duration_sum_percent_df, duration_sum_distribution_percent_df, duration_min_s_df, duration_max_s_df, duration_mean_s_df
