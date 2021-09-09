from dataclasses import dataclass
from logging import getLogger
from typing import List

from text_utils import SymbolFormat
from text_utils.language import Language
from text_utils.symbol_id_dict import SymbolIdDict
from text_utils.text import text_to_sentences, text_to_symbols
from text_utils.types import Symbol, SymbolIds, Symbols
from tts_preparation.utils import GenericList

UNINFERABLE_SYMBOL_MARKER = "█"


@dataclass
class InferableUtterance:
  utterance_id: int
  symbols: Symbols
  symbols_format: SymbolFormat
  symbols_uninferable_marked: Symbols
  inferable_symbols: Symbols
  inferable_symbol_ids: SymbolIds
  original_symbols: Symbols
  original_symbols_format: SymbolFormat
  language: Language

  @property
  def can_all_symbols_be_inferred(self) -> bool:
    return len(self.inferable_symbols) == len(self.symbols)


class InferableUtterances(GenericList[InferableUtterance]):
  pass


def log_utterance(utterance: InferableUtterance) -> None:
  logger = getLogger(__name__)
  utterance_id_str = f"{utterance.utterance_id}: "
  logger.info(f"{utterance_id_str}{''.join(utterance.original_symbols)}")
  logger.info(
    f"{len(utterance_id_str)*' '}{''.join(utterance.symbols_uninferable_marked)} ({len(utterance.inferable_symbols)})")


def __text_to_utterances(text: str, language: Language, text_format: SymbolFormat) -> List[str]:
  # each line is at least regarded as one sentence.
  lines = text.split("\n")
  all_utterances: List[str] = []
  for line in lines:
    utterance = text_to_sentences(
      text=line.strip(),
      lang=language,
      text_format=text_format,
    )
    all_utterances.extend(utterance)
  return all_utterances


def __remove_non_existent_symbols(symbols: Symbols, symbol_id_dict: SymbolIdDict) -> Symbols:
  result = tuple(symbol for symbol in symbols if symbol_id_dict.symbol_exists(symbol))
  return result


def replace_non_existent_symbols(symbols: Symbols, symbol_id_dict: SymbolIdDict, replace_with_symbol: Symbol) -> Symbols:
  result = tuple(symbol if symbol_id_dict.symbol_exists(symbol)
                 else replace_with_symbol for symbol in symbols)
  return result


def __text_utterances_to_inferable_utterances(text_utterances: List[str], language: Language, text_format: SymbolFormat, symbol_id_dict: SymbolIdDict):
  utterances = InferableUtterances()
  for utterance_id, text_utterance in enumerate(text_utterances, start=1):
    utterance = __text_utterance_to_inferable_utterance(
      text_utterance=text_utterance,
      language=language,
      symbol_id_dict=symbol_id_dict,
      text_format=text_format,
      utterance_id=utterance_id,
    )
    utterances.append(utterance)
  return utterances


def __text_utterance_to_inferable_utterance(text_utterance: str, language: Language, text_format: SymbolFormat, symbol_id_dict: SymbolIdDict, utterance_id: int):
  symbols = text_to_symbols(text_utterance, text_format, language)
  inferable_symbols = __remove_non_existent_symbols(symbols, symbol_id_dict)
  utterance = InferableUtterance(
    utterance_id=utterance_id,
    symbols=symbols,
    symbols_format=text_format,
    inferable_symbols=inferable_symbols,
    inferable_symbol_ids=symbol_id_dict.get_ids(inferable_symbols),
    symbols_uninferable_marked=replace_non_existent_symbols(
      symbols, symbol_id_dict, UNINFERABLE_SYMBOL_MARKER),
    language=language,
    original_symbols=symbols,
    original_symbols_format=text_format,
  )
  return utterance


def add_text(text: str, language: Language, text_format: SymbolFormat, symbol_id_dict: SymbolIdDict) -> InferableUtterances:
  text_utterances = __text_to_utterances(text, language, text_format)
  inferable_utterances = __text_utterances_to_inferable_utterances(
    text_utterances, language, text_format, symbol_id_dict)
  return inferable_utterances


def utterances_normalize(utterances: InferableUtterances, symbol_id_dict: SymbolIdDict) -> None:
  # TODO
  pass


def utterances_apply_mapping_table(utterances: InferableUtterances, symbol_id_dict: SymbolIdDict) -> None:
  # TODO
  pass


def utterances_convert_to_ipa(utterances: InferableUtterances, symbol_id_dict: SymbolIdDict) -> None:
  # TODO
  pass


def utterances_change_ipa(utterances: InferableUtterances, symbol_id_dict: SymbolIdDict) -> None:
  # TODO
  pass


def utterances_apply_symbols_map(utterances: InferableUtterances, symbol_id_dict: SymbolIdDict) -> None:
  # TODO
  pass
