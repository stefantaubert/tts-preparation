from debug.globals import BASE_DIR
from text_utils import Language, SymbolFormat
from text_utils.pronunciation.main import EngToIPAMode
from text_utils.symbols_map import create_or_update_inference_map
from tts_preparation.app.inference import (add_text, ipa_convert_text,
                                           map_text, normalize_text)


def main():

  merge_name = "debug_ljs_test"
  text_name = "debug"

  add_text(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    language=Language.ENG,
    text="This is a test.\nHello/ɑ/ what up?\n",
    text_filepath=None,
    text_format=SymbolFormat.GRAPHEMES,
    text_name=text_name,
  )

  normalize_text(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    text_name=text_name,
  )

  # ipa_convert_text(
  #   base_dir=BASE_DIR,
  #   merge_name=merge_name,
  #   text_name=text_name,
  #   consider_ipa_annotations=True,
  #   mode=EngToIPAMode.EPITRAN,
  # )

  map_text(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    symbols_map_path=None,
    text_name=text_name,
  )


if __name__ == "__main__":
  main()
