from typing import Callable, Optional, Set

from sentence2pronunciation import \
    sentence2pronunciaton as sentence2pronunciaton_orig
from sentence2pronunciation.core import Pronunciation, Symbol


def sentence2pronunciaton(sentence: Pronunciation, trim_symbols: Set[Symbol], split_on_hyphen: bool, get_pronunciation: Callable[[Pronunciation], Pronunciation], consider_annotation: bool, annotation_split_symbol: Optional[Symbol]) -> Pronunciation:
  return sentence2pronunciaton_orig(
    sentence=''.join(sentence),
    annotation_split_symbol=annotation_split_symbol,
    consider_annotation=consider_annotation,
    get_pronunciation=get_pronunciation,
    split_on_hyphen=split_on_hyphen,
    trim_symbols=trim_symbols,
  )
