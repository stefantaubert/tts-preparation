import random
import string
from dataclasses import dataclass
from functools import partial
from logging import getLogger
from typing import Optional

from accent_analyser.core.word_probabilities import (ProbabilitiesDict,
                                                     replace_with_prob)
from general_utils import GenericList, console_out_len
from sentence2pronunciation import sentence2pronunciation
from sentence2pronunciation.lookup_cache import get_empty_cache
from text_utils import (EngToIPAMode, Language, Symbol, SymbolFormat,
                        SymbolIdDict, SymbolIds, Symbols, SymbolsMap)
from text_utils import change_ipa as change_ipa_method
from text_utils import (symbols_to_ipa, symbols_to_sentences, text_normalize,
                        text_to_symbols)
from text_utils.text import change_symbols


@dataclass
class InferableUtterance:
  utterance_id: int
  language: Language
  symbols: Symbols
  symbols_format: SymbolFormat
  symbol_ids: SymbolIds

  def get_symbols_uninferable_marked(self, marker: Symbol) -> Symbols:
    result = tuple(symbol if symbol_id is not None else marker * console_out_len(symbol) for symbol,
                   symbol_id in zip(self.symbols, self.symbol_ids))
    return result

  @property
  def can_all_symbols_be_inferred(self) -> bool:
    return None not in self.symbol_ids


class InferableUtterances(GenericList[InferableUtterance]):
  pass


def get_utterances_txt(utterances: InferableUtterances, marker: Symbol) -> str:
  lines = []
  for utterance in utterances.items():
    line = f"{utterance.utterance_id}.: {''.join(utterance.get_symbols_uninferable_marked(marker))}"
    lines.append(line)
  return '\n'.join(lines)


def log_utterance(utterance: InferableUtterance, marker: Symbol) -> None:
  logger = getLogger(__name__)
  utterance_id_str = f"{utterance.utterance_id}.: "
  logger.info(
    f"{utterance_id_str}{''.join(utterance.get_symbols_uninferable_marked(marker))}")
  if not utterance.can_all_symbols_be_inferred:
    logger.info(
        f"{(len(utterance_id_str)-1)*' '}({''.join(utterance.symbols)})")

  logger.info(
    f"{len(utterance_id_str)*' '}{len(utterance.symbols)} {utterance.language!r} {utterance.symbols_format!r}")
  if not utterance.can_all_symbols_be_inferred:
    logger.warning("Not all symbols can be synthesized!")


def log_utterances(utterances: InferableUtterances, marker: Symbol) -> None:
  for utterance in utterances:
    log_utterance(utterance, marker)


def add_utterances_from_text(text: str, language: Language, text_format: SymbolFormat, symbol_id_dict: SymbolIdDict) -> InferableUtterances:
  new_utterances = InferableUtterances()
  # each non-empty line is regarded as one utterance.
  lines = text.split("\n")
  non_empty_lines = [line for line in lines if line != ""]
  for line_nr, line in enumerate(non_empty_lines, start=1):
    symbols = text_to_symbols(line, text_format, language)
    utterance = InferableUtterance(
      utterance_id=line_nr,
      language=language,
      symbols=symbols,
      symbols_format=text_format,
      symbol_ids=symbol_id_dict.get_ids(symbols),
    )
    new_utterances.append(utterance)
  return new_utterances


def utterances_split(utterances: InferableUtterances, symbol_id_dict: SymbolIdDict) -> InferableUtterances:
  new_utterances = InferableUtterances()
  counter = 1
  for utterance in utterances.items():
    sentences = symbols_to_sentences(
      symbols=utterance.symbols,
      symbols_format=utterance.symbols_format,
      lang=utterance.language,
    )

    for sentence_symbols in sentences:
      utterance = InferableUtterance(
        utterance_id=counter,
        language=utterance.language,
        symbols=sentence_symbols,
        symbols_format=utterance.symbols_format,
        symbol_ids=symbol_id_dict.get_ids(sentence_symbols),
      )
      new_utterances.append(utterance)
      counter += 1
  return new_utterances


def utterances_normalize(utterances: InferableUtterances, symbol_id_dict: SymbolIdDict) -> None:
  for utterance in utterances.items():
    new_text = text_normalize(
      text=''.join(utterance.symbols),
      lang=utterance.language,
      text_format=utterance.symbols_format,
    )

    new_symbols = text_to_symbols(
      text=new_text,
      lang=utterance.language,
      text_format=utterance.symbols_format,
    )

    utterance.symbols = new_symbols
    utterance.symbol_ids = symbol_id_dict.get_ids(new_symbols)


def utterances_convert_to_ipa(utterances: InferableUtterances, symbol_id_dict: SymbolIdDict, mode: Optional[EngToIPAMode], consider_annotations: Optional[bool]) -> None:
  cache = get_empty_cache()
  for utterance in utterances.items():
    new_symbols, new_format = symbols_to_ipa(
      symbols=utterance.symbols,
      lang=utterance.language,
      symbols_format=utterance.symbols_format,
      mode=mode,
      consider_annotations=consider_annotations,
      cache=cache,
    )

    utterance.symbols = new_symbols
    utterance.symbols_format = new_format
    utterance.symbol_ids = symbol_id_dict.get_ids(new_symbols)


def utterances_change_ipa(utterances: InferableUtterances, symbol_id_dict: SymbolIdDict, ignore_tones: bool, ignore_arcs: bool, ignore_stress: bool, break_n_thongs: bool, build_n_thongs: bool) -> None:
  for utterance in utterances.items(True):
    new_symbols = change_ipa_method(
      symbols=utterance.symbols,
      ignore_tones=ignore_tones,
      ignore_arcs=ignore_arcs,
      ignore_stress=ignore_stress,
      break_n_thongs=break_n_thongs,
      build_n_thongs=build_n_thongs,
      language=utterance.language,
    )

    utterance.symbols = new_symbols
    utterance.symbol_ids = symbol_id_dict.get_ids(new_symbols)


def utterances_change_text(utterances: InferableUtterances, symbol_id_dict: SymbolIdDict, remove_space_around_punctuation: bool) -> None:
  for utterance in utterances.items():
    new_symbols = change_symbols(
      symbols=utterance.symbols,
      remove_space_around_punctuation=remove_space_around_punctuation,
      lang=utterance.language,
    )

    utterance.symbols = new_symbols
    utterance.symbol_ids = symbol_id_dict.get_ids(new_symbols)


def utterances_apply_symbols_map(utterances: InferableUtterances, symbol_id_dict: SymbolIdDict, symbols_map: SymbolsMap) -> None:
  for utterance in utterances.items():
    new_symbols = symbols_map.apply_to_symbols(utterance.symbols)
    utterance.symbols = new_symbols
    utterance.symbol_ids = symbol_id_dict.get_ids(new_symbols)


def __get_pronunciation_from_mapping_table(word: Symbols, mapping_table: ProbabilitiesDict) -> Symbols:
  if word not in mapping_table:
    return word

  word_replaced = replace_with_prob(word, mapping_table)
  mapped_something = word != word_replaced
  if mapped_something:
    logger = getLogger(__name__)
    logger.info(
      f"Mapped \"{''.join(word)}\" to \"{''.join(word_replaced)}\".")
  return word_replaced


def utterances_apply_mapping_table(utterances: InferableUtterances, symbol_id_dict: SymbolIdDict, probabilities_dict: ProbabilitiesDict, seed: int) -> None:
  random.seed(seed)
  get_pronun_method = partial(__get_pronunciation_from_mapping_table,
                              mapping_table=probabilities_dict)
  for utterance in utterances.items():
    new_symbols = sentence2pronunciation(
      sentence=utterance.symbols,
      trim_symbols=set(string.punctuation),
      get_pronunciation=get_pronun_method,
      split_on_hyphen=True,
      consider_annotation=False,
      annotation_split_symbol=None,
    )

    utterance.symbols = new_symbols
    utterance.symbol_ids = symbol_id_dict.get_ids(new_symbols)
