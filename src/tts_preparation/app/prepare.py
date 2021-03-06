import os
import shutil
from logging import Logger, getLogger
from statistics import mean
from typing import Callable, Optional, Set, Tuple

import pandas as pd
from text_selection import get_common_durations
from text_utils import SymbolIdDict
from tts_preparation.app.merge_ds import (get_merged_dir, load_merged_data,
                                          load_merged_speakers_json,
                                          load_merged_symbol_converter)
from tts_preparation.core.data import (DatasetType, PreparedData,
                                       PreparedDataList)
from tts_preparation.core.helper import prep_data_list_to_dict_with_durations_s
from tts_preparation.core.prepare import (add_greedy_kld_ngram_seconds,
                                          add_greedy_ngram_epochs,
                                          add_greedy_ngram_seconds,
                                          add_n_divergent_random_seconds,
                                          add_ngram_cover,
                                          add_random_ngram_cover_seconds,
                                          add_random_percent,
                                          add_random_seconds, add_rest,
                                          add_symbols, core_process_stats,
                                          get_random_seconds_divergent_seeds,
                                          prepare_core)
from tts_preparation.core.stats_speaker import (get_speaker_stats,
                                                log_general_stats)
from tts_preparation.core.stats_symbols import get_ngram_stats_df
from tts_preparation.globals import DEFAULT_CSV_SEPERATOR
from tts_preparation.utils import get_subdir


def _get_prep_root_dir(merged_dir: str, create: bool = False):
  return get_subdir(merged_dir, 'training', create)


def get_prep_dir(merged_dir: str, prep_name: str, create: bool = False):
  return get_subdir(_get_prep_root_dir(merged_dir, create=create), prep_name, create)


def save_trainset(prep_dir: str, dataset: PreparedDataList):
  path = get_trainset_path(prep_dir)
  dataset.sort_after_entry_id()
  dataset.save(path)


def get_trainset_path(prep_dir: str):
  path = os.path.join(prep_dir, "training.csv")
  return path


def load_trainset(prep_dir: str) -> PreparedDataList:
  path = get_trainset_path(prep_dir)
  return PreparedDataList.load(PreparedData, path)


def get_testset_path(prep_dir: str):
  path = os.path.join(prep_dir, "test.csv")
  return path


def save_testset(prep_dir: str, dataset: PreparedDataList):
  path = get_testset_path(prep_dir)
  dataset.sort_after_entry_id()
  dataset.save(path)


def load_testset(prep_dir: str) -> PreparedDataList:
  path = get_testset_path(prep_dir)
  return PreparedDataList.load(PreparedData, path)


def get_valset_path(prep_dir: str):
  path = os.path.join(prep_dir, "validation.csv")
  return path


def save_valset(prep_dir: str, dataset: PreparedDataList):
  path = get_valset_path(prep_dir)
  dataset.sort_after_entry_id()
  dataset.save(path)


def load_valset(prep_dir: str) -> PreparedDataList:
  path = get_valset_path(prep_dir)
  return PreparedDataList.load(PreparedData, path)


def get_restset_path(prep_dir: str):
  path = os.path.join(prep_dir, "rest.csv")
  return path


def save_restset(prep_dir: str, dataset: PreparedDataList):
  path = get_restset_path(prep_dir)
  dataset.sort_after_entry_id()
  dataset.save(path)


def load_restset(prep_dir: str) -> PreparedDataList:
  path = get_restset_path(prep_dir)
  assert os.path.isfile(path)
  return PreparedDataList.load(PreparedData, path)


def get_totalset_path(prep_dir: str):
  path = os.path.join(prep_dir, "total.csv")
  return path


def save_totalset(prep_dir: str, dataset: PreparedDataList):
  path = get_totalset_path(prep_dir)
  dataset.sort_after_entry_id()
  dataset.save(path)


def load_totalset(prep_dir: str) -> PreparedDataList:
  path = get_totalset_path(prep_dir)
  assert os.path.isfile(path)
  return PreparedDataList.load(PreparedData, path)


def _save_speaker_stats(prep_dir: str, stats: pd.DataFrame):
  path = os.path.join(prep_dir, "stats_speaker.csv")
  stats.to_csv(path, sep=DEFAULT_CSV_SEPERATOR)


def _save_onegram_stats(prep_dir: str, stats: pd.DataFrame):
  path = os.path.join(prep_dir, "stats_onegram.csv")
  stats.to_csv(path, sep=DEFAULT_CSV_SEPERATOR)


def _load_onegram_stats(prep_dir: str) -> pd.DataFrame:
  path = os.path.join(prep_dir, "stats_onegram.csv")
  data = pd.read_csv(path, sep=DEFAULT_CSV_SEPERATOR)
  return data


def _save_twogram_stats(prep_dir: str, stats: pd.DataFrame):
  path = os.path.join(prep_dir, "stats_twogram.csv")
  stats.to_csv(path, sep=DEFAULT_CSV_SEPERATOR)


def _load_twogram_stats(prep_dir: str) -> pd.DataFrame:
  path = os.path.join(prep_dir, "stats_twogram.csv")
  data = pd.read_csv(path, sep=DEFAULT_CSV_SEPERATOR)
  return data


def _save_threegram_stats(prep_dir: str, stats: pd.DataFrame):
  path = os.path.join(prep_dir, "stats_threegram.csv")
  stats.to_csv(path, sep=DEFAULT_CSV_SEPERATOR)


def _load_threegram_stats(prep_dir: str) -> pd.DataFrame:
  path = os.path.join(prep_dir, "stats_threegram.csv")
  data = pd.read_csv(path, sep=DEFAULT_CSV_SEPERATOR)
  return data


def print_and_save_stats(base_dir: str, merge_name: str, prep_name: str):
  logger = getLogger(__name__)
  _print_quick_stats(base_dir, merge_name, prep_name, logger=logger)

  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  merge_data = load_merged_data(merge_dir)
  prep_dir = get_prep_dir(merge_dir, prep_name, create=False)
  trainset = load_trainset(prep_dir) if os.path.isfile(
    get_trainset_path(prep_dir)) else PreparedDataList()
  testset = load_testset(prep_dir) if os.path.isfile(
    get_testset_path(prep_dir)) else PreparedDataList()
  valset = load_valset(prep_dir) if os.path.isfile(
    get_valset_path(prep_dir)) else PreparedDataList()
  restset = load_restset(prep_dir) if os.path.isfile(
    get_restset_path(prep_dir)) else PreparedDataList()
  symbols = load_merged_symbol_converter(merge_dir)
  speakers = load_merged_speakers_json(merge_dir)

  log_general_stats(
    trainset=trainset,
    valset=valset,
    testset=testset,
    restset=restset,
    merge_data=merge_data,
    logger=logger,
  )

  logger.info("Calculating speaker stats...")
  speaker_stats = get_speaker_stats(
    speakers=speakers,
    symbols=symbols,
    trainset=trainset,
    valset=valset,
    testset=testset,
    restset=restset,
  )
  _save_speaker_stats(prep_dir, speaker_stats)

  logger.info("Calculating onegram stats...")
  onegram_stats = get_ngram_stats_df(
    symbols=symbols,
    trainset=trainset,
    valset=valset,
    testset=testset,
    restset=restset,
    n=1,
    logger=logger,
  )
  _save_onegram_stats(prep_dir, onegram_stats)

  logger.info("Calculating twogram stats...")
  twogram_stats = get_ngram_stats_df(
    symbols=symbols,
    trainset=trainset,
    valset=valset,
    testset=testset,
    restset=restset,
    n=2,
    logger=logger,
  )
  _save_twogram_stats(prep_dir, twogram_stats)

  logger.info("Calculating threegram stats...")
  threegram_stats = get_ngram_stats_df(
    symbols=symbols,
    trainset=trainset,
    valset=valset,
    testset=testset,
    restset=restset,
    n=3,
    logger=logger,
  )
  _save_threegram_stats(prep_dir, threegram_stats)

  logger.info("Done.")


def process_stats(base_dir: str, merge_name: str, prep_name: str, ds: DatasetType):
  logger = getLogger(__name__)
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  #merge_data = load_merged_data(merge_dir)
  prep_dir = get_prep_dir(merge_dir, prep_name, create=False)
  onegram_stats = _load_onegram_stats(prep_dir)
  twogram_stats = _load_twogram_stats(prep_dir)
  threegram_stats = _load_threegram_stats(prep_dir)
  core_process_stats(
    onegram_stats=onegram_stats,
    twogram_stats=twogram_stats,
    threegram_stats=threegram_stats,
    speaker_stats=None,
    ds=ds,
    logger=logger,
  )


def _print_quick_stats(base_dir: str, merge_name: str, prep_name: str, logger: Logger):
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  merge_data = load_merged_data(merge_dir)
  prep_dir = get_prep_dir(merge_dir, prep_name, create=False)
  trainset = load_trainset(prep_dir) if os.path.isfile(
    get_trainset_path(prep_dir)) else PreparedDataList()
  testset = load_testset(prep_dir) if os.path.isfile(
    get_testset_path(prep_dir)) else PreparedDataList()
  valset = load_valset(prep_dir) if os.path.isfile(
    get_valset_path(prep_dir)) else PreparedDataList()
  restset = load_restset(prep_dir) if os.path.isfile(
    get_restset_path(prep_dir)) else PreparedDataList()

  log_general_stats(
    trainset=trainset,
    valset=valset,
    testset=testset,
    restset=restset,
    merge_data=merge_data,
    logger=logger,
  )


def app_prepare(base_dir: str, merge_name: str, prep_name: str, overwrite: bool = True):
  logger = getLogger(__name__)
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  prep_dir = get_prep_dir(merge_dir, prep_name, create=False)
  if os.path.isdir(prep_dir):
    if overwrite:
      logger.info("Removing existing...")
      shutil.rmtree(prep_dir)
    else:
      logger.info("Already created.")
      return
  merge_data = load_merged_data(merge_dir)
  totalset = prepare_core(merge_data)
  os.makedirs(prep_dir)
  save_restset(prep_dir, totalset)
  save_totalset(prep_dir, totalset)
  logger.info("Done.")
  _print_quick_stats(base_dir, merge_name, prep_name, logger)


def load_set(prep_dir: str, dataset: DatasetType) -> PreparedDataList:
  if dataset == DatasetType.TRAINING:
    return load_trainset(prep_dir) if os.path.isfile(
        get_trainset_path(prep_dir)) else PreparedDataList()
  if dataset == DatasetType.VALIDATION:
    return load_valset(prep_dir) if os.path.isfile(
        get_valset_path(prep_dir)) else PreparedDataList()
  if dataset == DatasetType.TEST:
    return load_testset(prep_dir) if os.path.isfile(
        get_testset_path(prep_dir)) else PreparedDataList()
  raise Exception()


def _save_results(dest_prep_dir: str, new_set: PreparedDataList, new_restset: PreparedDataList, dataset: DatasetType):
  save_restset(dest_prep_dir, new_restset)
  if dataset == DatasetType.TRAINING:
    save_trainset(dest_prep_dir, new_set)
  elif dataset == DatasetType.VALIDATION:
    save_valset(dest_prep_dir, new_set)
  elif dataset == DatasetType.TEST:
    save_testset(dest_prep_dir, new_set)


def copy_orig_to_dest_dir(orig_prep_dir: str, dest_prep_dir: str):
  os.makedirs(dest_prep_dir, exist_ok=True)
  if os.path.isfile(get_trainset_path(orig_prep_dir)):
    save_trainset(dest_prep_dir, load_trainset(orig_prep_dir))
  if os.path.isfile(get_valset_path(orig_prep_dir)):
    save_valset(dest_prep_dir, load_valset(orig_prep_dir))
  if os.path.isfile(get_testset_path(orig_prep_dir)):
    save_testset(dest_prep_dir, load_testset(orig_prep_dir))
  if os.path.isfile(get_restset_path(orig_prep_dir)):
    save_restset(dest_prep_dir, load_restset(orig_prep_dir))
  if os.path.isfile(get_totalset_path(orig_prep_dir)):
    save_totalset(dest_prep_dir, load_totalset(orig_prep_dir))


def app_add_rest(base_dir: str, merge_name: str, orig_prep_name: str, dest_prep_name: str, dataset: DatasetType, overwrite: bool = True):
  __add(
    base_dir=base_dir,
    merge_name=merge_name,
    orig_prep_name=orig_prep_name,
    dest_prep_name=dest_prep_name,
    dataset=dataset,
    overwrite=overwrite,
    func=add_rest,
  )


def app_add_ngram_cover(base_dir: str, merge_name: str, orig_prep_name: str, dest_prep_name: str, dataset: DatasetType, n_gram: int, ignore_symbol_ids: Optional[Set[int]] = None, top_percent: Optional[float] = None, overwrite: bool = True):
  __add(
    base_dir=base_dir,
    merge_name=merge_name,
    orig_prep_name=orig_prep_name,
    dest_prep_name=dest_prep_name,
    dataset=dataset,
    overwrite=overwrite,
    func=add_ngram_cover,
    n_gram=n_gram,
    ignore_symbol_ids=ignore_symbol_ids,
    top_percent=top_percent,
  )


def app_add_random_minutes(base_dir: str, merge_name: str, orig_prep_name: str, dest_prep_name: str, minutes: float, seed: int, dataset: DatasetType, respect_existing: bool = False, overwrite: bool = True):
  __add(
    base_dir=base_dir,
    merge_name=merge_name,
    orig_prep_name=orig_prep_name,
    dest_prep_name=dest_prep_name,
    dataset=dataset,
    overwrite=overwrite,
    func=add_random_seconds,
    seconds=minutes * 60,
    seed=seed,
    respect_existing=respect_existing,
  )


def app_add_random_percent(base_dir: str, merge_name: str, orig_prep_name: str, dest_prep_name: str, percent: float, seed: int, dataset: DatasetType, overwrite: bool = True):
  __add(
    base_dir=base_dir,
    merge_name=merge_name,
    orig_prep_name=orig_prep_name,
    dest_prep_name=dest_prep_name,
    dataset=dataset,
    overwrite=overwrite,
    func=add_random_percent,
    percent=percent,
    seed=seed,
  )


def app_add_symbols(base_dir: str, merge_name: str, orig_prep_name: str, dest_prep_name: str, dataset: DatasetType, cover_symbol_ids: Set[int], overwrite: bool = True):
  __add(
    base_dir=base_dir,
    merge_name=merge_name,
    orig_prep_name=orig_prep_name,
    dest_prep_name=dest_prep_name,
    dataset=dataset,
    overwrite=overwrite,
    func=add_symbols,
    cover_symbol_ids=cover_symbol_ids,
  )


def app_add_random_ngram_cover_minutes(base_dir: str, merge_name: str, orig_prep_name: str, dest_prep_name: str, dataset: DatasetType, n_gram: int, seed: int, minutes: float, ignore_symbol_ids: Optional[Set[int]] = None, overwrite: bool = True):
  __add(
    base_dir=base_dir,
    merge_name=merge_name,
    orig_prep_name=orig_prep_name,
    dest_prep_name=dest_prep_name,
    dataset=dataset,
    overwrite=overwrite,
    func=add_random_ngram_cover_seconds,
    ignore_symbol_ids=ignore_symbol_ids,
    seconds=minutes * 60,
    n_gram=n_gram,
    seed=seed,
  )


def app_add_greedy_ngram_minutes(base_dir: str, merge_name: str, orig_prep_name: str, dest_prep_name: str, dataset: DatasetType, n_gram: int, minutes: float, ignore_symbol_ids: Optional[Set[int]] = None, overwrite: bool = True):
  __add(
    base_dir=base_dir,
    merge_name=merge_name,
    orig_prep_name=orig_prep_name,
    dest_prep_name=dest_prep_name,
    dataset=dataset,
    overwrite=overwrite,
    func=add_greedy_ngram_seconds,
    ignore_symbol_ids=ignore_symbol_ids,
    seconds=minutes * 60,
    n_gram=n_gram,
  )


def app_add_n_diverse_random_minutes(base_dir: str, merge_name: str, orig_prep_name: str, dest_prep_name: str, dataset: DatasetType, overwrite: bool, seed: int, minutes: float, n: int):
  logger = getLogger(__name__)
  logger.info(f"Adding utterances speaker-wise to {str(dataset)}...")
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)

  orig_prep_dir = get_prep_dir(merge_dir, orig_prep_name, create=False)
  symbols = load_merged_symbol_converter(merge_dir)

  new_datasets = add_n_divergent_random_seconds(
    existing_set=load_set(orig_prep_dir, dataset),
    restset=load_restset(orig_prep_dir),
    symbols=symbols,
    seed=seed,
    n=n,
    seconds=minutes * 60,
  )

  dest_names = []
  for i, (new_set, new_restset) in enumerate(new_datasets):
    logger.info(f"Saving {i+1}/{len(new_datasets)}...")
    dest_name = f"{dest_prep_name}_{i+1}"
    dest_prep_dir = get_prep_dir(merge_dir, dest_name, create=False)
    if not overwrite and os.path.isdir(dest_prep_dir):
      logger.info(f"{dest_name} already exists. Skipping...")
      continue
    if dest_prep_dir != orig_prep_dir:
      copy_orig_to_dest_dir(orig_prep_dir, dest_prep_dir)
    _save_results(dest_prep_dir, new_set, new_restset, dataset)
    logger.info(f"Saved to: {dest_name}")
    _print_quick_stats(base_dir, merge_name, dest_name, logger)
    dest_names.append(dest_name)

  new_sets = [
   {x.entry_id for x in load_trainset(
    get_prep_dir(merge_dir, dest_name, create=False)).items()}
      for dest_name in dest_names
  ]

  total_set = load_totalset(orig_prep_dir)
  durations_s = prep_data_list_to_dict_with_durations_s(total_set)

  common_durations = get_common_durations(new_sets, durations_s)
  logger.info("Overlapping")
  for set_combi, common_duration_s in common_durations.items():
    logger.info(
      f"{set_combi}: {common_duration_s:.2f}s / {common_duration_s / 60:.2f}min / {common_duration_s / 60 / 60:.2f}h")

  mean_s = mean(common_durations.values())
  logger.info(
    f"Average duration: {mean_s:.2f}s / {mean_s / 60:.2f}min / {mean_s / 60 / 60:.2f}h")
  mean_dur = mean(durations_s.values())

  #common_elements = get_total_number_of_common_elements(new_sets)
  #logger.info(f"Entries: {common_elements}")
  logger.info(f"Avg Entry Dur: {mean_dur:.2f}s")
  #avg_entries = common_elements / len(common_durations)
  #logger.info(f"Avg Entries: {avg_entries:.0f} = {avg_entries * mean_dur:.2f}s")
  logger.info("Done.")


def app_add_greedy_ngram_epochs(base_dir: str, merge_name: str, orig_prep_name: str, dest_prep_name: str, dataset: DatasetType, n_gram: int, epochs: int, ignore_symbol_ids: Optional[Set[int]] = None, overwrite: bool = True):
  __add(
    base_dir=base_dir,
    merge_name=merge_name,
    orig_prep_name=orig_prep_name,
    dest_prep_name=dest_prep_name,
    dataset=dataset,
    overwrite=overwrite,
    func=add_greedy_ngram_epochs,
    ignore_symbol_ids=ignore_symbol_ids,
    n_gram=n_gram,
    epochs=epochs,
  )


def app_add_greedy_kld_ngram_minutes(base_dir: str, merge_name: str, orig_prep_name: str, dest_prep_name: str, dataset: DatasetType, n_gram: int, minutes: float, ignore_symbol_ids: Optional[Set[int]] = None, overwrite: bool = True):
  __add(
    base_dir=base_dir,
    merge_name=merge_name,
    orig_prep_name=orig_prep_name,
    dest_prep_name=dest_prep_name,
    dataset=dataset,
    overwrite=overwrite,
    func=add_greedy_kld_ngram_seconds,
    ignore_symbol_ids=ignore_symbol_ids,
    seconds=minutes * 60,
    n_gram=n_gram,
  )


def __add(base_dir: str, merge_name: str, orig_prep_name: str, dest_prep_name: str, dataset: DatasetType, overwrite: bool, func: Callable[[PreparedDataList, PreparedDataList, SymbolIdDict], Tuple[PreparedDataList, PreparedDataList]], **kwargs):
  logger = getLogger(__name__)
  logger.info(f"Adding utterances speaker-wise to {str(dataset)}...")
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  dest_prep_dir = get_prep_dir(merge_dir, dest_prep_name, create=False)
  if not overwrite and os.path.isdir(dest_prep_dir):
    logger.info("Already exists.")
  orig_prep_dir = get_prep_dir(merge_dir, orig_prep_name, create=False)
  symbols = load_merged_symbol_converter(merge_dir)

  new_set, new_restset = func(
    existing_set=load_set(orig_prep_dir, dataset),
    restset=load_restset(orig_prep_dir),
    symbols=symbols,
    **kwargs,
  )

  if dest_prep_dir != orig_prep_dir:
    copy_orig_to_dest_dir(orig_prep_dir, dest_prep_dir)
  _save_results(dest_prep_dir, new_set, new_restset, dataset)
  logger.info("Done.")
  _print_quick_stats(base_dir, merge_name, dest_prep_name, logger)


def app_get_random_seconds_divergent_seeds(base_dir: str, merge_name: str, prep_name: str, minutes: float, seed: int, samples: int, n: int):
  logger = getLogger(__name__)
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  symbols = load_merged_symbol_converter(merge_dir)
  orig_prep_dir = get_prep_dir(merge_dir, prep_name, create=False)
  rest_set = load_restset(orig_prep_dir)

  selected_seeds, selected_sets = get_random_seconds_divergent_seeds(
    restset=rest_set,
    symbols=symbols,
    seed=seed,
    seconds=minutes * 60,
    n=n,
    samples=samples,
  )

  logger.info("The most divergent seeds are:")
  show_n = 10
  for selected_seed, selected_set in zip(selected_seeds, selected_sets):
    selected_entry_ids = list(sorted(selected_set))
    first_entries = list(map(str, selected_entry_ids[:show_n]))
    last_entries = list(map(str, selected_entry_ids[-show_n:]))
    logger.info(
      f"{selected_seed}: {', '.join(first_entries)}, ..., {', '.join(last_entries)} ({len(selected_entry_ids)})")
