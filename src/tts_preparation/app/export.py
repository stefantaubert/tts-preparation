
from logging import getLogger
from pathlib import Path
from shutil import copy2, rmtree

from ordered_set import OrderedSet
from text_selection_api import Dataset, create, get_selection
from text_utils import StringFormat2
from tts_preparation.app.io import get_merged_dir
from tts_preparation.app.prepare import (get_prep_dir, load_restset, load_set,
                                         load_testset, load_totalset,
                                         load_trainset, load_valset,
                                         save_restset, save_testset,
                                         save_trainset, save_valset)
from tts_preparation.core.data import DatasetType, get_entry_ids
from tts_preparation.core.helper import select_entities_from_prep_data


def export_audios(base_dir: Path, merge_name: str, prep_name: str, dataset: DatasetType, output_directory: Path, overwrite: bool) -> None:
  logger = getLogger(__name__)
  logger.info(f"Exporting audio files {str(dataset)}...")
  merge_dir = get_merged_dir(base_dir, merge_name)
  prep_dir = get_prep_dir(merge_dir, prep_name)

  if output_directory.exists():
    if overwrite:
      rmtree(output_directory)
      logger.info("Removed existing output directory.")
    else:
      logger.error("Output directory already exists!")
      return

  output_directory.mkdir(parents=True, exist_ok=False)

  ds = load_set(prep_dir, dataset)
  for entry in ds.items_tqdm():
    assert entry.wav_absolute_path.is_file()
    output_wav_path = output_directory / f"{entry.entry_id}.wav"
    output_text_path = output_directory / f"{entry.entry_id}.txt"
    text = StringFormat2.SPACED.convert_symbols_to_string(entry.symbols)

    copy2(entry.wav_absolute_path, output_wav_path, follow_symlinks=False)
    output_text_path.write_text(text, encoding="UTF-8")


def export_for_text_selection(base_dir: Path, merge_name: str, prep_name: str, output_directory: Path, overwrite: bool) -> None:
  logger = getLogger(__name__)
  logger.info(f"Exporting for text selection...")
  merge_dir = get_merged_dir(base_dir, merge_name)
  prep_dir = get_prep_dir(merge_dir, prep_name)

  if output_directory.exists():
    if overwrite:
      rmtree(output_directory)
      logger.info("Removed existing output directory.")
    else:
      logger.error("Output directory already exists!")
      return

  output_directory.mkdir(parents=True, exist_ok=False)
  totalset = load_totalset(prep_dir)
  ids = get_entry_ids(totalset)

  all_ids = OrderedSet(ids)
  trn_ids = OrderedSet(get_entry_ids(load_trainset(prep_dir)))
  tst_ids = OrderedSet(get_entry_ids(load_testset(prep_dir)))
  val_ids = OrderedSet(get_entry_ids(load_valset(prep_dir)))
  rst_ids = OrderedSet(get_entry_ids(load_restset(prep_dir)))

  assert set(all_ids) == set(trn_ids.union(tst_ids, val_ids, rst_ids))
  assert trn_ids.issubset(all_ids)
  assert tst_ids.issubset(all_ids)
  assert val_ids.issubset(all_ids)
  assert rst_ids.issubset(all_ids)

  dataset = Dataset(all_ids)
  dataset.subsets["trn"] = trn_ids
  dataset.subsets["tst"] = tst_ids
  dataset.subsets["val"] = val_ids
  dataset.subsets["rst"] = rst_ids

  durations = ((entry.entry_id, entry.wav_duration) for entry in totalset.items())
  weights = {
    "durations": dict(durations)
  }

  symbols = dict(
    (entry.entry_id, StringFormat2.SPACED.convert_symbols_to_string(entry.symbols))
    for entry in totalset.items()
  )

  try:
    create(output_directory, dataset, weights, symbols, False)
  except ValueError as error:
    logger.error(error)
    return


def import_from_selection(base_dir: Path, merge_name: str, prep_name: str, import_directory: Path) -> None:
  logger = getLogger(__name__)
  logger.info("Importing from text selection...")
  merge_dir = get_merged_dir(base_dir, merge_name)
  prep_dir = get_prep_dir(merge_dir, prep_name)

  if not import_directory.exists():
    logger.error("Import directory does not exist!")
    return

  try:
    trn_selection = get_selection(import_directory, "trn")
    tst_selection = get_selection(import_directory, "tst")
    val_selection = get_selection(import_directory, "val")
    rst_selection = get_selection(import_directory, "rst")
  except ValueError as error:
    logger.error(error)
    return
  totalset = load_totalset(prep_dir)
  all_ids = set(get_entry_ids(totalset))

  # TODO maybe also ensure that sets are exclusive
  if all_ids != trn_selection.union(tst_selection, val_selection, rst_selection):
    logger.error("Not all ids are included in the sets!")
    return

  save_trainset(prep_dir, select_entities_from_prep_data(trn_selection, totalset))
  save_testset(prep_dir, select_entities_from_prep_data(tst_selection, totalset))
  save_valset(prep_dir, select_entities_from_prep_data(val_selection, totalset))
  save_restset(prep_dir, select_entities_from_prep_data(rst_selection, totalset))
