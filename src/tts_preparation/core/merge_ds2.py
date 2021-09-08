from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
from typing import OrderedDict as OrderedDictType
from typing import Set, Tuple

from speech_dataset_preprocessing import FinalDsEntry, FinalDsEntryList
from text_utils.gender import Gender
from text_utils.language import Language
from text_utils.speakers_dict import SpeakersDict
from text_utils.symbol_format import SymbolFormat
from text_utils.symbol_id_dict import SymbolIdDict
from text_utils.types import Speaker, SpeakerId, Speakers, SymbolIds, Symbols
from tts_preparation.globals import DEFAULT_PADDING_SYMBOL
from tts_preparation.utils import GenericList

ALL_SPEAKERS_INDICATOR = "all"


@dataclass()
class PreparedData:
  entry_id: int
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
  combined_rarity: float

  def load_init(self):
    pass
    # self.duration = float(self.duration)
    # self.speaker_id = int(self.speaker_id)
    # self.entry_id = int(self.entry_id)
    # self.ds_entry_id = int(self.ds_entry_id)


class PreparedDataList(GenericList[PreparedData]):
  pass


def expand_speakers(speakers_to_ds_names: Dict[str, Set[Speaker]], ds_speakers: List[Tuple[str, Speaker]]) -> Dict[str, Set[Speaker]]:
  expanded_speakers: Dict[str, Set[Speaker]] = {ds_name: {} for ds_name, _ in speakers_to_ds_names}
  for ds_name, speaker_name in ds_speakers:
    if ds_name not in speakers_to_ds_names:
      continue
    if speaker_name == ALL_SPEAKERS_INDICATOR:
      expanded_speakers[ds_name] |= speakers_to_ds_names[ds_name]
    else:
      if speaker_name not in speakers_to_ds_names[ds_name]:
        continue
      expanded_speakers[ds_name].add(speaker_name)
  return expanded_speakers


def get_speakers_of_data(final_ds_list: FinalDsEntryList) -> Set[Speaker]:
  result = {entry.speaker_name for entry in final_ds_list.items()}
  return result


def get_ds_speaker_name(ds_name: str, speaker: Speaker) -> Speaker:
  result = f"{ds_name}:{speaker}"
  return result


def merge(datasets: List[Tuple[str, FinalDsEntryList]], ds_speakers: List[Tuple[str, Speaker]]) -> FinalDsEntryList:
  speakers_to_ds_name = {ds_name: get_speakers_of_data(data) for ds_name, data in datasets}
  selected_ds_speakers = expand_speakers(speakers_to_ds_name, ds_speakers)
  res = FinalDsEntryList()
  for ds_name, final_ds_data in datasets:
    for entry in final_ds_data.items():
      take_entry = entry.speaker_name in selected_ds_speakers[ds_name]
      if take_entry:
        copied_entry = deepcopy(entry)
        copied_entry.speaker_name = get_ds_speaker_name(ds_name, entry.speaker_name)
        res.append(copied_entry)
  return res


def create_symbol_id_dict(data: FinalDsEntryList) -> SymbolIdDict:
  all_symbols = {symbol for entry in data.items() for symbol in entry.symbols}
  result = SymbolIdDict.init_from_symbols_with_pad(all_symbols, pad_symbol=DEFAULT_PADDING_SYMBOL)
  return result


def create_speaker_id_dict(data: FinalDsEntryList) -> SpeakersDict:
  all_speakers = get_speakers_of_data(data)
  result = SpeakersDict.fromlist(list(sorted(all_speakers)))
  return result


def map_to_prepared_data(data: FinalDsEntryList, symbol_id_dict: SymbolIdDict, speaker_id_dict: SpeakersDict) -> PreparedDataList:
  result = PreparedDataList()
  for entry in data.items():
    # TODO
    prep_entry = PreparedData(
      entry_id=entry.entry_id,
      symbol_ids=symbol_id_dict.get_ids(entry.symbols),
    )
    result.append(prep_entry)
  return result
