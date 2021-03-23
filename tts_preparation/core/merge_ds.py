import os
from dataclasses import dataclass
from logging import Logger
from typing import Dict, List, Optional, OrderedDict, Set, Tuple

from speech_dataset_preprocessing import (DsDataList, MelDataList,
                                          TextDataList, WavDataList)
from text_utils import (AccentsDict, Gender, Language, SpeakersDict,
                        SymbolIdDict, deserialize_list)
from tts_preparation.globals import (DEFAULT_PADDING_ACCENT,
                                     DEFAULT_PADDING_SYMBOL)
from tts_preparation.utils import (GenericList, contains_only_allowed_symbols,
                                   get_counter)

ALL_SPEAKERS_INDICATOR = "all"


@dataclass
class DsDataset():
  name: str
  data: DsDataList
  texts: TextDataList
  wavs: WavDataList
  mels: MelDataList
  speakers: SpeakersDict
  symbol_ids: SymbolIdDict
  accent_ids: AccentsDict
  absolute_mel_dir: str
  absolute_wav_dir: str


@dataclass
class DsDatasetList(GenericList[DsDataset]):
  pass


@dataclass
class MergedDatasetEntry():
  entry_id: int
  basename: str
  speaker_id: int
  text_original: str
  text: str
  serialized_symbol_ids: str
  serialized_accent_ids: str
  gender: Gender
  lang: Language
  absolute_wav_path: str
  duration: float
  sampling_rate: int
  absolute_mel_path: str
  n_mel_channels: int

  def load_init(self):
    self.lang = Language(self.lang)
    self.gender = Gender(self.gender)


class MergedDataset(GenericList[MergedDatasetEntry]):
  def load_init(self):
    for item in self.items():
      item.load_init()

  @classmethod
  def init_from_ds_dataset(cls, ds: DsDataset):
    res = cls()
    for ds_data, text_data, wav_data, mel_data in zip(ds.data.items(), ds.texts.items(), ds.wavs.items(), ds.mels.items()):
      absolute_wav_path = os.path.join(ds.absolute_wav_dir, wav_data.relative_wav_path)
      absolute_mel_path = os.path.join(ds.absolute_mel_dir, mel_data.relative_mel_path)

      new_entry = MergedDatasetEntry(
        entry_id=ds_data.entry_id,
        gender=ds_data.gender,
        basename=ds_data.basename,
        speaker_id=ds_data.speaker_id,
        lang=text_data.lang,
        text_original=ds_data.text,
        text=text_data.text,
        serialized_accent_ids=text_data.serialized_accent_ids,
        serialized_symbol_ids=text_data.serialized_symbol_ids,
        absolute_wav_path=absolute_wav_path,
        duration=wav_data.duration,
        sampling_rate=wav_data.sr,
        absolute_mel_path=absolute_mel_path,
        n_mel_channels=mel_data.n_mel_channels,
      )
      res.append(new_entry)
    return res

  def get_total_duration_s(self):
    durations = [x.duration for x in self.items()]
    total_duration = sum(durations)
    return total_duration


class MergedDatasetContainer():
  def __init__(self, data: MergedDataset, name: Optional[str], speaker_ids: SpeakersDict, symbol_ids: SymbolIdDict, accent_ids: AccentsDict) -> None:
    self.data = data
    self.name = name
    self.symbol_ids = symbol_ids
    self.accent_ids = accent_ids
    self.speaker_ids = speaker_ids

  @classmethod
  def init_from_ds_dataset(cls, ds: DsDataset):
    data = MergedDataset.init_from_ds_dataset(ds)
    res = cls(
      data=data,
      name=ds.name,
      speaker_ids=ds.speakers,
      symbol_ids=ds.symbol_ids,
      accent_ids=ds.accent_ids,
    )
    return res

  def remove_speakers(self, speakers: Set[int]) -> None:
    res = MergedDataset()
    for entry in self.data.items():
      if entry.speaker_id not in speakers:
        res.append(entry)
    self.data = res
    self._remove_unused_speakers()
    self._remove_unused_symbols()
    self._remove_unused_accents()

  def _remove_unused_symbols(self) -> None:
    all_symbol_ids: Set[int] = set()
    for entry in self.data.items():
      all_symbol_ids |= set(deserialize_list(entry.serialized_symbol_ids))
    unused_symbol_ids = self.symbol_ids.get_all_symbol_ids().difference(all_symbol_ids)
    # unused_symbols = unused_symbols.difference({PADDING_SYMBOL})
    self.symbol_ids.remove_ids(unused_symbol_ids)

  def _remove_unused_accents(self) -> None:
    all_accent_ids: Set[int] = set()
    for entry in self.data.items():
      all_accent_ids |= set(deserialize_list(entry.serialized_accent_ids))
    unused_accent_ids = self.accent_ids.get_all_ids().difference(all_accent_ids)
    # unused_symbols = unused_symbols.difference({PADDING_SYMBOL})
    self.accent_ids.remove_ids(unused_accent_ids)

  def _remove_unused_speakers(self) -> None:
    all_speaker_ids: Set[int] = set()
    for entry in self.data.items():
      all_speaker_ids |= {entry.speaker_id}
    unused_speaker_ids = set(self.speaker_ids.get_all_speaker_ids()).difference(all_speaker_ids)
    # unused_symbols = unused_symbols.difference({PADDING_SYMBOL})
    self.speaker_ids.remove_ids(unused_speaker_ids)


class MergedDatasetContainerList():
  def __init__(self, data: List[MergedDatasetContainer]) -> None:
    self.data = data

  def merge(self) -> MergedDatasetContainer:
    accent_ids = self.make_common_accent_ids()
    speaker_ids = self.make_common_speaker_ids()
    symbol_ids = self.make_common_symbol_ids()

    new_ds = MergedDataset()
    for dataset in self.data:
      for entry in dataset.data.items():
        new_ds.append(entry)

    # TODO: maybe sorting after speakerid and then entry_id

    # # Set new entry_id
    # for i, entry in enumerate(new_ds.items()):
    #   entry.entry_id = i

    res = MergedDatasetContainer(
      data=new_ds,
      name=None,
      accent_ids=accent_ids,
      speaker_ids=speaker_ids,
      symbol_ids=symbol_ids,
    )

    return res

  def make_common_symbol_ids(self) -> SymbolIdDict:
    all_symbols: Set[str] = set()
    for ds in self.data:
      all_symbols |= ds.symbol_ids.get_all_symbols()
    new_symbol_ids = SymbolIdDict.init_from_symbols_with_pad(
      all_symbols, pad_symbol=DEFAULT_PADDING_SYMBOL)

    for ds in self.data:
      for entry in ds.data.items():
        original_symbols = ds.symbol_ids.get_symbols(entry.serialized_symbol_ids)
        entry.serialized_symbol_ids = new_symbol_ids.get_serialized_ids(original_symbols)
      ds.symbol_ids = new_symbol_ids

    return new_symbol_ids

  def make_common_accent_ids(self) -> AccentsDict:
    all_accents: Set[str] = set()
    for ds in self.data:
      all_accents |= ds.accent_ids.get_all_accents()
    new_accent_ids = AccentsDict.init_from_accents_with_pad(
      all_accents, pad_accent=DEFAULT_PADDING_ACCENT)

    for ds in self.data:
      for entry in ds.data.items():
        original_accents = ds.accent_ids.get_accents(entry.serialized_accent_ids)
        entry.serialized_accent_ids = new_accent_ids.get_serialized_ids(original_accents)
      ds.accent_ids = new_accent_ids

    return new_accent_ids

  @staticmethod
  def get_new_speaker_name(ds_name: str, speaker_name: str) -> str:
    return f"{ds_name},{speaker_name}"

  def make_common_speaker_ids(self) -> SpeakersDict:
    all_speakers: List[str] = []
    for ds in self.data:
      old_speaker_names = ds.speaker_ids.get_all_speakers()
      new_speaker_names = [MergedDatasetContainerList.get_new_speaker_name(
        ds.name, old_name) for old_name in old_speaker_names]
      all_speakers.extend(new_speaker_names)

    all_speakers_have_unique_names = len(all_speakers) == len(set(all_speakers))
    assert all_speakers_have_unique_names

    new_speaker_ids = SpeakersDict.fromlist(all_speakers)

    for ds in self.data:
      for entry in ds.data.items():
        old_speaker_name = ds.speaker_ids.get_speaker(entry.speaker_id)
        new_speaker_name = MergedDatasetContainerList.get_new_speaker_name(
          ds.name, old_speaker_name)
        entry.speaker_id = new_speaker_ids.get_id(new_speaker_name)
      ds.speaker_ids = new_speaker_ids

    return new_speaker_ids


def log_stats(data: MergedDataset, symbols: SymbolIdDict, accent_ids: AccentsDict, speakers: SpeakersDict, logger: Logger):
  logger.info(f"Speakers ({len(speakers)}): {', '.join(sorted(speakers.get_all_speakers()))}")
  logger.info(f"Symbols ({len(symbols)}): {' '.join(sorted(symbols.get_all_symbols()))}")
  logger.info(f"Accents ({len(accent_ids)}): {', '.join(sorted(accent_ids.get_all_accents()))}")
  logger.info(f"Entries ({len(data)}): {data.get_total_duration_s()/60:.2f}m")
  symbol_counter = get_counter([symbols.get_symbols(x.serialized_symbol_ids) for x in data.items()])
  logger.info(symbol_counter)
  # log texts and trainsets


def _get_ds_speaker_ids(datasets: DsDatasetList, ds_speakers: List[Tuple[str, str]]) -> Dict[str, Set[int]]:
  speakers_dict = {ds.name: ds.speakers.get_all_speakers() for ds in datasets.items()}
  expanded_ds_speakers = expand_speakers(speakers_dict, ds_speakers)

  result: Dict[str, Set[int]] = {}
  for ds_name, speaker_name in expanded_ds_speakers:
    for ds in datasets.items():
      if ds.name == ds_name:
        ds_speaker_id = ds.speakers.get_id(speaker_name)
        if ds_name not in result:
          result[ds_name] = set()
        result[ds_name] |= {ds_speaker_id}
        break

  return result


def preprocess(datasets: DsDatasetList, ds_speakers: List[Tuple[str, str]], logger: Logger) -> MergedDatasetContainer:
  ds_sepaker_ids = _get_ds_speaker_ids(datasets, ds_speakers)

  merged_datasets: List[MergedDatasetContainer] = []
  for ds in datasets.items():
    if ds.name in ds_sepaker_ids.keys():
      merged_ds_container = MergedDatasetContainer.init_from_ds_dataset(ds)
      merged_datasets.append(merged_ds_container)

  for ds in merged_datasets:
    not_included_speakers = set(ds.speaker_ids.get_all_speaker_ids(
      )).difference(ds_sepaker_ids[ds.name])
    ds.remove_speakers(not_included_speakers)

  merged_dataset_container_list = MergedDatasetContainerList(
    data=merged_datasets
  )

  result = merged_dataset_container_list.merge()

  log_stats(
    data=result.data,
    symbols=result.symbol_ids,
    accent_ids=result.accent_ids,
    speakers=result.speaker_ids,
    logger=logger,
  )

  return result


def expand_speakers(speakers_dict: OrderedDict[str, List[str]], ds_speakers: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
  # expand all
  expanded_speakers: List[Tuple[str, str]] = []
  for ds_name, speaker_name in ds_speakers:
    if ds_name not in speakers_dict:
      continue
    if speaker_name == ALL_SPEAKERS_INDICATOR:
      expanded_speakers.extend([(ds_name, speaker) for speaker in speakers_dict[ds_name]])
    else:
      if speaker_name not in speakers_dict[ds_name]:
        continue
      expanded_speakers.append((ds_name, speaker_name))
  expanded_speakers = list(sorted(set(expanded_speakers)))
  return expanded_speakers


def filter_symbols(data: MergedDataset, symbols: SymbolIdDict, accent_ids: AccentsDict, speakers: SpeakersDict, allowed_symbol_ids: Set[int], logger: Logger) -> MergedDatasetContainer:
  # maybe check all symbol ids are valid before
  allowed_symbols = [symbols.get_symbol(x) for x in allowed_symbol_ids]
  not_allowed_symbols = [symbols.get_symbol(x)
                         for x in symbols.get_all_symbol_ids() if x not in allowed_symbol_ids]
  logger.info(f"Keep utterances with these symbols: {' '.join(allowed_symbols)}")
  logger.info(f"Remove utterances with these symbols: {' '.join(not_allowed_symbols)}")
  logger.info("Statistics before filtering:")
  log_stats(data, symbols, accent_ids, speakers, logger)
  result = MergedDataset([x for x in data.items() if contains_only_allowed_symbols(
    deserialize_list(x.serialized_symbol_ids), allowed_symbol_ids)])
  if len(result) > 0:
    logger.info(
        f"Removed {len(data) - len(result)} from {len(data)} total entries and got {len(result)} entries ({len(result)/len(data)*100:.2f}%).")
  else:
    logger.info("Removed all utterances!")
  new_symbol_ids = update_symbols(result, symbols)
  new_accent_ids = update_accents(result, accent_ids)
  new_speaker_ids = update_speakers(result, speakers)
  logger.info("Statistics after filtering:")
  log_stats(result, new_symbol_ids, new_accent_ids, new_speaker_ids, logger)

  res = MergedDatasetContainer(
    name=None,
    data=result,
    accent_ids=new_accent_ids,
    speaker_ids=new_speaker_ids,
    symbol_ids=new_symbol_ids,
  )
  return res


def update_accents(data: MergedDataset, accent_ids: AccentsDict) -> AccentsDict:
  new_accents: Set[str] = {x for y in data.items()
                           for x in accent_ids.get_accents(y.serialized_accent_ids)}
  new_accent_ids = AccentsDict.init_from_accents_with_pad(
    new_accents, pad_accent=DEFAULT_PADDING_ACCENT)
  if new_accent_ids.get_all_accents() != accent_ids.get_all_accents():
    for entry in data.items():
      original_accents = accent_ids.get_accents(entry.serialized_accent_ids)
      entry.serialized_accent_ids = new_accent_ids.get_serialized_ids(original_accents)
  return new_accent_ids


def update_symbols(data: MergedDataset, symbols: SymbolIdDict) -> SymbolIdDict:
  new_symbols: Set[str] = {x for y in data.items()
                           for x in symbols.get_symbols(y.serialized_symbol_ids)}
  new_symbol_ids = SymbolIdDict.init_from_symbols_with_pad(
    new_symbols, pad_symbol=DEFAULT_PADDING_SYMBOL)
  if new_symbol_ids.get_all_symbols() != symbols.get_all_symbols():
    for entry in data.items():
      original_symbols = symbols.get_symbols(entry.serialized_symbol_ids)
      entry.serialized_symbol_ids = new_symbol_ids.get_serialized_ids(original_symbols)
  return new_symbol_ids


def update_speakers(data: MergedDataset, speakers: SpeakersDict) -> SpeakersDict:
  new_speakers: Set[str] = {speakers.get_speaker(y.speaker_id) for y in data.items()}
  new_speaker_ids = SpeakersDict.fromlist(new_speakers)
  if new_speaker_ids.get_all_speakers() != speakers.get_all_speakers():
    for entry in data.items():
      old_speaker_name = speakers.get_speaker(entry.speaker_id)
      entry.speaker_id = new_speaker_ids.get_id(old_speaker_name)
  return new_speaker_ids


# def extract_random_subset(data: MergedDataset, symbols: SymbolIdDict, accent_ids: AccentsDict, speakers: SpeakersDict, shards: int):
#   text_data = {x.entry_id: symbols.get_symbols(x.serialized_symbol_ids) for x in data.items()}
#   res = main_rand(
#     text_data=text_data,
#     shards=shards,
#   )
#   result = MergedDataset([x for x in data.items() if x.entry_id in res.keys()])
#   # new_symbols: Set[str] = set([x for y in res.values() for x in y])
#   new_symbol_ids = update_symbols(result, symbols)
#   new_accent_ids = update_accents(result, accent_ids)
#   new_speaker_ids = update_speakers(result, speakers)

#   return result, new_symbol_ids, new_accent_ids, new_speaker_ids
