from debug.globals import BASE_DIR
from text_utils import Language, SymbolFormat
from text_utils.pronunciation.main import EngToIPAMode
from tts_preparation.app.inference import (add_text, change_ipa_text,
                                           ipa_convert_text, map_text,
                                           normalize_text, split_text)


def main():

  merge_name = "debug_ljs_test"
  text_name = "debug"

  add_text(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    language=Language.ENG,
    text="This is a test.\nHello /\u02C8ɑ/ŋ̋̏/ what up? And not!\n",
    text_filepath=None,
    text_format=SymbolFormat.GRAPHEMES,
    text_name=text_name,
  )

  # normalize_text(
  #   base_dir=BASE_DIR,
  #   merge_name=merge_name,
  #   text_name=text_name,
  # )

  split_text(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    text_name=text_name,
  )

  ipa_convert_text(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    text_name=text_name,
    consider_ipa_annotations=True,
    mode=EngToIPAMode.EPITRAN,
  )

  change_ipa_text(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    text_name=text_name,
    ignore_arcs=True,
    ignore_stress=True,
    ignore_tones=True,
  )

  map_text(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    symbols_map_path=None,
    text_name=text_name,
  )

  # apply_mapping_table(
  #   base_dir=BASE_DIR,
  #   merge_name=merge_name,
  #   text_name=text_name,
  #   mapping_table_path=
  # )


if __name__ == "__main__":
  main()
