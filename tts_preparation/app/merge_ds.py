from logging import getLogger
import os
import shutil
from typing import List, Set, Tuple

from speech_dataset_preprocessing import (get_ds_dir, get_mel_dir,
                                          get_text_dir, get_wav_dir,
                                          load_ds_accents_json, load_ds_csv,
                                          load_ds_speaker_json, load_mel_csv,
                                          load_text_csv,
                                          load_text_symbol_converter,
                                          load_wav_csv)
from text_utils import AccentsDict, SpeakersDict, SymbolIdDict
from tts_preparation.app.io import get_pre_dir
from tts_preparation.core.merge_ds import (DsDataset, DsDatasetList,
                                           MergedDataset, MergedDatasetEntry,
                                           filter_symbols, preprocess)
from tts_preparation.utils import get_subdir


def _get_merged_root_dir(base_dir: str, create: bool = False):
  return get_subdir(get_pre_dir(base_dir, create), 'merged', create)


_merge_data_csv = "data.csv"
_merge_speakers_json = "speakers.json"
_merge_symbols_json = "symbols.json"
_merge_accents_json = "accents.json"


def get_merged_dir(base_dir: str, merge_name: str, create: bool = False):
  return get_subdir(_get_merged_root_dir(base_dir, create), merge_name, create)


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


def merge_ds(base_dir: str, merge_name: str, ds_speakers: List[Tuple[str, str]], ds_text_audio: List[Tuple[str, str, str]], delete_existing: bool = True):
  logger = getLogger(__name__)
  logger.info(f"Merging dataset: {merge_name}...")
  merge_dir = get_merged_dir(base_dir, merge_name)

  if os.path.isdir(merge_dir) and not delete_existing:
    logger.info("Already created.")
    return

  datasets = DsDatasetList()
  for ds_name, text_name, audio_name in ds_text_audio:
    # multiple uses of one ds are not valid

    ds_dir = get_ds_dir(base_dir, ds_name)
    text_dir = get_text_dir(ds_dir, text_name)
    wav_dir = get_wav_dir(ds_dir, audio_name)
    mel_dir = get_mel_dir(ds_dir, audio_name)

    ds_dataset = DsDataset(
      name=ds_name,
      data=load_ds_csv(ds_dir),
      texts=load_text_csv(text_dir),
      wavs=load_wav_csv(wav_dir),
      mels=load_mel_csv(mel_dir),
      speakers=load_ds_speaker_json(ds_dir),
      symbol_ids=load_text_symbol_converter(text_dir),
      accent_ids=load_ds_accents_json(ds_dir),
      absolute_wav_dir=wav_dir,
      absolute_mel_dir=mel_dir,
    )

    datasets.append(ds_dataset)

  merged_data = preprocess(
    datasets=datasets,
    ds_speakers=ds_speakers,
    logger=logger,
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
