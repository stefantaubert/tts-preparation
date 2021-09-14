import shutil
from logging import getLogger
from pathlib import Path
from typing import List, Set, Tuple

from speech_dataset_preprocessing import get_final_ds
from speech_dataset_preprocessing.core.final import FinalDsEntryList
from text_utils import SpeakersDict
from text_utils.types import Speaker, Symbol
from tts_preparation.app.io import (get_merged_dir,
                                    load_merged_symbol_converter,
                                    save_merged_symbol_converter)
from tts_preparation.core.merge_ds import (DsName, PreparedDataList, merge,
                                           remove_unwanted_symbols)
from tts_preparation.utils import load_obj, save_obj

_merge_data_csv = "data.csv"
_merge_speakers_json = "speakers.json"


def load_merged_data(merge_dir: Path) -> PreparedDataList:
  path = merge_dir / _merge_data_csv
  return load_obj(path)


def save_merged_data(merge_dir: Path, result: PreparedDataList) -> None:
  path = merge_dir / _merge_data_csv
  save_obj(result, path)


def load_merged_speakers_json(merge_dir: Path) -> SpeakersDict:
  path = merge_dir / _merge_speakers_json
  return SpeakersDict.load(path)


def save_merged_speakers_json(merge_dir: Path, speakers: SpeakersDict):
  path = merge_dir / _merge_speakers_json
  speakers.save(path)


def merge_ds(base_dir: Path, sdp_dir: Path, merge_name: str, ds_speakers: List[Tuple[DsName, Speaker]], ds_text_audio: List[Tuple[DsName, str, str]], overwrite: bool = True) -> None:
  logger = getLogger(__name__)
  logger.info(f"Merging dataset: {merge_name}...")
  dest_merge_dir = get_merged_dir(base_dir, merge_name)

  if dest_merge_dir.is_dir() and dest_merge_dir.exists() and not overwrite:
    logger.info("Already created.")
    return

  datasets: List[Tuple[DsName, FinalDsEntryList]] = []
  for ds_name, text_name, audio_name in set(ds_text_audio):
    final_data_list = get_final_ds(
      base_dir=sdp_dir,
      ds_name=ds_name,
      text_name=text_name,
      wav_name=audio_name,
    )

    datasets.append((ds_name, final_data_list))

  dest_data, dest_symbol_ids_dict, dest_speaker_ids_dict = merge(
    datasets=datasets,
    ds_speakers=ds_speakers,
  )

  assert overwrite
  if dest_merge_dir.is_dir():
    shutil.rmtree(dest_merge_dir)
  dest_merge_dir.mkdir(parents=True, exist_ok=False)

  save_merged_data(dest_merge_dir, dest_data)
  save_merged_symbol_converter(dest_merge_dir, dest_symbol_ids_dict)
  save_merged_speakers_json(dest_merge_dir, dest_speaker_ids_dict)
  logger.info("Done.")


def ds_filter_symbols(base_dir: str, orig_merge_name: str, dest_merge_name: str, allowed_symbols: Set[Symbol], overwrite: bool = True):
  logger = getLogger(__name__)
  dest_merge_dir = get_merged_dir(base_dir, dest_merge_name)

  if dest_merge_dir.is_dir() and dest_merge_dir.exists() and not overwrite:
    logger.info("Already created.")
    return

  orig_merge_dir = get_merged_dir(base_dir, orig_merge_name)
  orig_data = load_merged_data(orig_merge_dir)

  result = remove_unwanted_symbols(
    data=orig_data,
    allowed_symbols=allowed_symbols,
  )

  if result is None:
    dest_data = orig_data
    dest_symbol_ids_dict = load_merged_symbol_converter(orig_merge_dir)
    dest_speaker_ids_dict = load_merged_symbol_converter(orig_merge_dir)
  else:
    dest_data, dest_symbol_ids_dict, dest_speaker_ids_dict = result

  assert overwrite
  if dest_merge_dir.is_dir():
    shutil.rmtree(dest_merge_dir)
  dest_merge_dir.mkdir(parents=True, exist_ok=False)

  save_merged_data(dest_merge_dir, dest_data)
  save_merged_symbol_converter(dest_merge_dir, dest_symbol_ids_dict)
  save_merged_speakers_json(dest_merge_dir, dest_speaker_ids_dict)
  logger.info("Done.")
