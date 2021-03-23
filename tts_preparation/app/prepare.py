from logging import getLogger
import os
from typing import Optional

from tts_preparation.app.merge_ds import get_merged_dir, load_merged_symbol_converter
from tts_preparation.utils import get_subdir
from tts_preparation.core.data import DatasetType, PreparedData, PreparedDataList
from tts_preparation.core.main import (add_ngram_core, add_ngram_kld_core,
                                    add_random_count_core)


def _get_prep_root_dir(merged_dir: str, create: bool = False):
  return get_subdir(merged_dir, 'training', create)


def get_prep_dir(merged_dir: str, prep_name: str, create: bool = False):
  return get_subdir(_get_prep_root_dir(merged_dir, create=create), prep_name, create)


def save_trainset(prep_dir: str, dataset: PreparedDataList):
  path = get_trainset_path(prep_dir)
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
  dataset.save(path)


def load_testset(prep_dir: str) -> PreparedDataList:
  path = get_testset_path(prep_dir)
  return PreparedDataList.load(PreparedData, path)


def get_valset_path(prep_dir: str):
  path = os.path.join(prep_dir, "validation.csv")
  return path


def save_valset(prep_dir: str, dataset: PreparedDataList):
  path = get_valset_path(prep_dir)
  dataset.save(path)


def load_valset(prep_dir: str) -> PreparedDataList:
  path = get_valset_path(prep_dir)
  return PreparedDataList.load(PreparedData, path)


def get_restset_path(prep_dir: str):
  path = os.path.join(prep_dir, "rest.csv")
  return path


def save_restset(prep_dir: str, dataset: PreparedDataList):
  path = get_restset_path(prep_dir)
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
  dataset.save(path)


def load_totalset(prep_dir: str) -> PreparedDataList:
  path = get_totalset_path(prep_dir)
  assert os.path.isfile(path)
  return PreparedDataList.load(PreparedData, path)


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


def add_random_count(base_dir: str, merge_name: str, orig_prep_name: str, dest_prep_name: str, shards_per_speaker: int, seed: int, ignore_already_added: bool, dataset: DatasetType, min_count_symbol: int = 0, overwrite: bool = True):
  logger = getLogger(__name__)
  logger.info(
    f"Adding {shards_per_speaker} shards out of random utterances per speaker to {str(dataset)}...")
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  orig_prep_dir = get_prep_dir(merge_dir, orig_prep_name, create=False)
  dest_prep_dir = get_prep_dir(merge_dir, dest_prep_name, create=False)
  if not overwrite and os.path.isdir(dest_prep_dir):
    logger.info("Already exists.")
  symbols = load_merged_symbol_converter(merge_dir)

  new_set, new_restset = add_random_count_core(
    symbols=symbols,
    existing_set=load_set(orig_prep_dir, dataset),
    restset=load_restset(orig_prep_dir),
    shards_per_speaker=shards_per_speaker,
    min_count_symbol=min_count_symbol,
    seed=seed,
    ignore_already_added=ignore_already_added,
    logger=logger,
  )

  if dest_prep_dir != orig_prep_dir:
    copy_orig_to_dest_dir(orig_prep_dir, dest_prep_dir)
  _save_results(dest_prep_dir, new_set, new_restset, dataset)
  logger.info("Done.")


def add_ngram_kld(base_dir: str, merge_name: str, orig_prep_name: str, dest_prep_name: str, shards_per_speaker: Optional[int], n_gram: int, n_its: Optional[int], dataset: DatasetType, ignore_already_added: bool, min_count_symbol: int = 0, overwrite: bool = True, top_percent: float = 100):
  logger = getLogger(__name__)
  logger.info(f"Adding {n_gram}-grams kld to {str(dataset)}...")
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  orig_prep_dir = get_prep_dir(merge_dir, orig_prep_name, create=False)
  dest_prep_dir = get_prep_dir(merge_dir, dest_prep_name, create=False)
  if not overwrite and os.path.isdir(dest_prep_dir):
    logger.info("Already exists.")
  symbols = load_merged_symbol_converter(merge_dir)
  totalset = load_totalset(orig_prep_dir)

  new_set, new_restset = add_ngram_kld_core(
    existing_set=load_set(orig_prep_dir, dataset),
    restset=load_restset(orig_prep_dir),
    n_gram=n_gram,
    n_its=n_its,
    ignore_already_added=ignore_already_added,
    shards_per_speaker=shards_per_speaker,
    symbols=symbols,
    total_set=totalset,
    min_count_symbol=min_count_symbol,
    logger=logger,
    top_percent=top_percent,
  )

  if dest_prep_dir != orig_prep_dir:
    copy_orig_to_dest_dir(orig_prep_dir, dest_prep_dir)
  _save_results(dest_prep_dir, new_set, new_restset, dataset)
  logger.info("Done.")


def add_ngram(base_dir: str, merge_name: str, orig_prep_name: str, dest_prep_name: str, shards_per_speaker: Optional[int], n_gram: int, n_its: Optional[int], dataset: DatasetType, ignore_already_added: bool, min_count_symbol: int = 0, overwrite: bool = True, top_percent: float = 100):
  logger = getLogger(__name__)
  logger.info(f"Adding {n_gram}-grams to {str(dataset)}...")
  merge_dir = get_merged_dir(base_dir, merge_name, create=False)
  orig_prep_dir = get_prep_dir(merge_dir, orig_prep_name, create=False)
  dest_prep_dir = get_prep_dir(merge_dir, dest_prep_name, create=False)
  if not overwrite and os.path.isdir(dest_prep_dir):
    logger.info("Already exists.")
  symbols = load_merged_symbol_converter(merge_dir)
  totalset = load_totalset(orig_prep_dir)

  new_set, new_restset = add_ngram_core(
    existing_set=load_set(orig_prep_dir, dataset),
    restset=load_restset(orig_prep_dir),
    n_gram=n_gram,
    n_its=n_its,
    ignore_already_added=ignore_already_added,
    shards_per_speaker=shards_per_speaker,
    symbols=symbols,
    total_set=totalset,
    min_count_symbol=min_count_symbol,
    top_percent=top_percent,
    logger=logger,
  )

  if dest_prep_dir != orig_prep_dir:
    copy_orig_to_dest_dir(orig_prep_dir, dest_prep_dir)
  _save_results(dest_prep_dir, new_set, new_restset, dataset)
  logger.info("Done.")
