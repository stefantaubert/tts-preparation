import os
import shutil
from logging import getLogger
from typing import List, Set, Tuple

from speech_dataset_preprocessing import (get_ds_dir, get_mel_dir,
                                          get_text_dir, get_wav_dir,
                                          load_ds_accents_json, load_ds_csv,
                                          load_ds_speaker_json, load_mel_csv,
                                          load_text_csv,
                                          load_text_symbol_converter,
                                          load_wav_csv)
from speech_dataset_preprocessing.app.final import get_final_ds
from speech_dataset_preprocessing.core.final import FinalDsEntryList
from text_utils import AccentsDict, SpeakersDict, SymbolIdDict
from text_utils.types import Speaker
from tts_preparation.core.merge_ds import (DsDataset, DsDatasetList,
                                           MergedDataset, MergedDatasetEntry,
                                           filter_symbols, merge)
from tts_preparation.utils import get_subdir

_merge_data_csv = "data.csv"
_merge_speakers_json = "speakers.json"
_merge_symbols_json = "symbols.json"
_merge_accents_json = "accents.json"


def get_merged_dir(base_dir: str, merge_name: str, create: bool = False):
  return get_subdir(base_dir, merge_name, create)


def load_merged_data(merge_dir: str) -> MergedDataset:
  path = os.path.join(merge_dir, _merge_data_csv)
  return MergedDataset.load(MergedDatasetEntry, path)


def save_merged_data(merge_dir: str, result: MergedDataset):
  path = os.path.join(merge_dir, _merge_data_csv)
  result.save(path)


def load_merged_speakers_json(merge_dir: str) -> SpeakersDict:
  path = os.path.join(merge_dir, _merge_speakers_json)
  return SpeakersDict.load(path)


def save_merged_speakers_json(merge_dir: str, speakers: SpeakersDict):
  path = os.path.join(merge_dir, _merge_speakers_json)
  speakers.save(path)


def load_merged_symbol_converter(merge_dir: str) -> SymbolIdDict:
  path = os.path.join(merge_dir, _merge_symbols_json)
  return SymbolIdDict.load_from_file(path)


def save_merged_symbol_converter(merge_dir: str, data: SymbolIdDict):
  path = os.path.join(merge_dir, _merge_symbols_json)
  data.save(path)


def load_merged_accents_ids(merge_dir: str) -> AccentsDict:
  path = os.path.join(merge_dir, _merge_accents_json)
  return AccentsDict.load(path)


def save_merged_accents_ids(merge_dir: str, data: AccentsDict):
  path = os.path.join(merge_dir, _merge_accents_json)
  data.save(path)


def merge_ds(base_dir: str, sdp_dir: str, merge_name: str, ds_speakers: List[Tuple[str, Speaker]], ds_text_audio: List[Tuple[str, str, str]], delete_existing: bool = True):
  logger = getLogger(__name__)
  logger.info(f"Merging dataset: {merge_name}...")
  merge_dir = get_merged_dir(base_dir, merge_name)

  if os.path.isdir(merge_dir) and not delete_existing:
    logger.info("Already created.")
    return

  datasets: List[Tuple[str, FinalDsEntryList]] = []
  for ds_name, text_name, audio_name in ds_text_audio:
    # multiple uses of one ds are not valid
    final_data_list = get_final_ds(
      base_dir=base_dir,
      ds_name=ds_name,
      text_name=text_name,
      wav_name=audio_name,
    )

    datasets.append((ds_name, final_data_list))

  merged_data = merge(
    datasets=datasets,
    ds_speakers=ds_speakers,
  )

  if os.path.isdir(merge_dir):
    shutil.rmtree(merge_dir)
  os.makedirs(merge_dir)
  save_merged_data(merge_dir, merged_data.data)
  save_merged_symbol_converter(merge_dir, merged_data.symbol_ids)
  save_merged_accents_ids(merge_dir, merged_data.accent_ids)
  save_merged_speakers_json(merge_dir, merged_data.speaker_ids)


def ds_filter_symbols(base_dir: str, orig_merge_name: str, dest_merge_name: str, allowed_symbol_ids: Set[int], overwrite: bool = True):
  logger = getLogger(__name__)
  dest_merge_dir = get_merged_dir(base_dir, dest_merge_name)

  if os.path.isdir(dest_merge_dir) and not overwrite:
    logger.info("Already created.")
    return

  merge_dir = get_merged_dir(base_dir, orig_merge_name)
  accents = load_merged_accents_ids(merge_dir)
  symbols = load_merged_symbol_converter(merge_dir)
  speakers = load_merged_speakers_json(merge_dir)
  data = load_merged_data(merge_dir)

  resulting_data = filter_symbols(
    data=data,
    symbols=symbols,
    accent_ids=accents,
    speakers=speakers,
    allowed_symbol_ids=allowed_symbol_ids,
    logger=logger,
  )

  if os.path.isdir(dest_merge_dir):
    shutil.rmtree(dest_merge_dir)
  os.makedirs(dest_merge_dir)
  save_merged_data(dest_merge_dir, resulting_data.data)
  save_merged_symbol_converter(dest_merge_dir, resulting_data.symbol_ids)
  save_merged_accents_ids(dest_merge_dir, resulting_data.accent_ids)
  save_merged_speakers_json(dest_merge_dir, resulting_data.speaker_ids)
