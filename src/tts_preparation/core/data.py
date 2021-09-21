from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, Set

from general_utils import GenericList
from text_utils.gender import Gender
from text_utils.language import Language
from text_utils.symbol_format import SymbolFormat
from text_utils.types import Speaker, SpeakerId, SymbolIds, Symbols

EntryId = int


@dataclass()
class PreparedData:
  entry_id: EntryId
  ds_entry_id: EntryId
  identifier: str
  speaker_id: SpeakerId
  speaker_name: Speaker
  speaker_gender: Gender
  symbol_ids: SymbolIds
  symbols_language: Language
  symbols_original: Symbols
  symbols_original_format: SymbolFormat
  symbols: Symbols
  symbols_format: SymbolFormat
  wav_original_absolute_path: Path
  wav_absolute_path: Path
  wav_duration: float
  wav_sampling_rate: int
  mel_absolute_path: Path
  mel_n_channels: int
  one_gram_rarity: float
  two_gram_rarity: float
  three_gram_rarity: float

  @property
  def combined_rarity(self) -> float:
    return self.one_gram_rarity + self.two_gram_rarity + self.three_gram_rarity


class PreparedDataList(GenericList[PreparedData]):
  # @staticmethod
  # def _get_key_for_sorting(elem: PreparedData) -> int:
  #   return elem.speaker_id, elem.ds_entry_id

  # def custom_sort(self):
  #   self.sort(key=PreparedDataList._get_key_for_sorting, reverse=False)

  @staticmethod
  def _get_key_for_entry_id_sorting(elem: PreparedData) -> int:
    return elem.entry_id

  def sort_after_entry_id(self):
    self.sort(key=PreparedDataList._get_key_for_entry_id_sorting, reverse=False)

  @property
  def unique_entry_ids(self) -> Set[int]:
    res = {x.entry_id for x in self.items()}
    return res

  @property
  def total_duration_s(self) -> float:
    durations = [entry.wav_duration for entry in self.items()]
    total_duration = sum(durations)
    return total_duration


def get_speaker_wise(data: PreparedDataList) -> Dict[Speaker, PreparedDataList]:
  speaker_data: Dict[Speaker, PreparedDataList] = {}
  for entry in data.items():
    if entry.speaker_name not in speaker_data:
      speaker_data[entry.speaker_name] = PreparedDataList()
    speaker_data[entry.speaker_name].append(entry)
  return speaker_data


class DatasetType(IntEnum):
  TRAINING = 0
  VALIDATION = 1
  TEST = 2

  def __str__(self) -> str:
    if self == DatasetType.TRAINING:
      return "training set"
    if self == DatasetType.VALIDATION:
      return "validation set"
    if self == DatasetType.TEST:
      return "test set"
    raise Exception()


# @dataclass()
# class PreparedData:
#   entry_id: int
#   ds_entry_id: int
#   text_original: str
#   text: str
#   wav_path: Path
#   mel_path: Path
#   serialized_symbol_ids: str
#   serialized_accent_ids: str
#   duration_s: float
#   speaker_id: int
#   one_gram_rarity: float
#   two_gram_rarity: float
#   three_gram_rarity: float
#   combined_rarity: float

#   def load_init(self):
#     pass
#     # self.duration = float(self.duration)
#     # self.speaker_id = int(self.speaker_id)
#     # self.entry_id = int(self.entry_id)
#     # self.ds_entry_id = int(self.ds_entry_id)


# class PreparedDataList(GenericList[PreparedData]):
#   def load_init(self):
#     for item in self.items():
#       item.load_init()

#   def get_total_duration_s(self):
#     durations = [x.wav_duration for x in self.items()]
#     total_duration = sum(durations)
#     return total_duration

#   def get_entry_ids(self) -> Set[int]:
#     res = {x.entry_id for x in self.items()}
#     return res

#   def get_for_validation(self, entry_ids: Optional[Set[int]], speaker_id: Optional[int], seed: Optional[int] = 1234) -> List[PreparedData]:
#     if entry_ids is not None:
#       entries = [x for x in self.items() if x.entry_id in entry_ids]
#       return entries

#     if speaker_id is not None:
#       assert seed is not None
#       return [self._get_random_entry_speaker_id(speaker_id, seed)]

#     return [self.get_random_entry(seed)]

#   def _get_entry(self, entry_id: int) -> PreparedData:
#     for entry in self.items():
#       if entry.entry_id == entry_id:
#         return entry
#     raise Exception()

#   def _get_random_entry_speaker_id(self, speaker_id: int, seed: int) -> PreparedData:
#     relevant_entries = [x for x in self.items() if x.speaker_id == speaker_id]
#     assert len(relevant_entries) > 0
#     random.seed(seed)
#     entry = random.choice(relevant_entries)
#     return entry

#   @staticmethod
#   def _get_key_for_sorting(elem: PreparedData) -> int:
#     return elem.speaker_id, elem.ds_entry_id

#   def custom_sort(self):
#     self.sort(key=PreparedDataList._get_key_for_sorting, reverse=False)

#   @staticmethod
#   def _get_key_for_entry_id_sorting(elem: PreparedData) -> int:
#     return elem.entry_id

#   def sort_after_entry_id(self):
#     self.sort(key=PreparedDataList._get_key_for_entry_id_sorting, reverse=False)

#   @classmethod
#   def init_from_merged_ds(cls, merged_ds: MergedDataset):
#     res = cls()
#     for entry in merged_ds.items():
#       prep_data = PreparedData(
#         entry_id=-1,
#         ds_entry_id=entry.entry_id,
#         text=entry.text,
#         text_original=entry.text_original,
#         duration_s=entry.duration,
#         mel_path=entry.absolute_mel_path,
#         speaker_id=entry.speaker_id,
#         serialized_accent_ids=entry.serialized_accent_ids,
#         serialized_symbol_ids=entry.serialized_symbol_ids,
#         wav_path=entry.absolute_wav_path,
#         one_gram_rarity=entry.one_gram_rarity,
#         two_gram_rarity=entry.two_gram_rarity,
#         three_gram_rarity=entry.three_gram_rarity,
#         combined_rarity=entry.combined_rarity,
#       )
#       res.append(prep_data)
#     res.custom_sort()
#     for i, entry in enumerate(res.items()):
#       entry.entry_id = i
#     return res
