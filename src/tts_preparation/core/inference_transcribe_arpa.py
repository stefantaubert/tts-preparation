from logging import getLogger
from multiprocessing import cpu_count
from typing import Dict, Generator, Iterable, List, Optional, Set, Tuple, cast

from g2p_en import G2p
from ordered_set import OrderedSet
from pronunciation_dict_parser import (Pronunciation, PronunciationDict,
                                       PublicDictType, parse_public_dict)
from sentence2pronunciation import sentences2pronunciations_from_cache_mp, sentence2pronunciation
from sentence2pronunciation.lookup_cache import LookupCache
from sentence2pronunciation.multiprocessing import prepare_cache_mp
from text_utils import StringFormat, Symbol, Symbols, symbols_to_upper, SymbolIdDict, SymbolFormat
from text_utils.utils import (pronunciation_dict_to_tuple_dict, symbols_ignore,
                              symbols_split)
from textgrid import Interval, IntervalTier, TextGrid
from tqdm import tqdm
from tts_preparation.core.inference import InferableUtterances

from tts_preparation.globals import DEFAULT_PUNCTUATION


def transcribe_to_arpa(utterances: InferableUtterances, symbol_id_dict: SymbolIdDict, consider_annotations: bool, dictionary: PublicDictType):
  logger = getLogger(__name__)

  logger.debug(f"Getting {dictionary!r} dictionary...")
  arpa_dict = parse_public_dict(dictionary)

  global PROCESS_OOV_MODEL
  global PROCESS_DICTIONARY
  PROCESS_DICTIONARY = pronunciation_dict_to_tuple_dict(arpa_dict)
  PROCESS_OOV_MODEL = G2p()

  logger.debug("Converting all words to ARPA...")

  for utterance in utterances.items():
    new_symbols = sentence2pronunciation(
      sentence=utterance.symbols,
      trim_symbols=DEFAULT_PUNCTUATION,
      annotation_split_symbol="/",
      consider_annotation=consider_annotations,
      get_pronunciation=process_lookup_dict,
      split_on_hyphen=True,
    )

    utterance.symbols = new_symbols
    utterance.symbols_format = SymbolFormat.PHONEMES_ARPA
    utterance.symbol_ids = symbol_id_dict.get_ids(new_symbols)

  logger.debug("Done.")


def __get_arpa_oov(model: G2p, word: Pronunciation) -> Pronunciation:
  word_str = ''.join(word)
  oov_arpa = model.predict(word_str)
  logger = getLogger(__name__)
  logger.info(f"Transliterated OOV word \"{word_str}\" to \"{' '.join(oov_arpa)}\".")
  return oov_arpa


PROCESS_OOV_MODEL: Optional[G2p] = None
PROCESS_DICTIONARY: Optional[Dict[Pronunciation, Pronunciation]] = None


def process_lookup_dict(word: Pronunciation) -> Pronunciation:
  global PROCESS_OOV_MODEL
  global PROCESS_DICTIONARY
  assert PROCESS_OOV_MODEL is not None
  assert PROCESS_DICTIONARY is not None
  return lookup_dict(word, PROCESS_DICTIONARY, PROCESS_OOV_MODEL)


def lookup_dict(word: Pronunciation, dictionary: Dict[Symbols, Symbols], model: G2p) -> Pronunciation:
  word_upper = symbols_to_upper(word)
  if word_upper in dictionary:
    return dictionary[word_upper][0]
  return __get_arpa_oov(model, word)
