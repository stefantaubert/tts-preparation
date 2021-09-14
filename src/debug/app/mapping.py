from pathlib import Path

from debug.globals import BASE_DIR
from text_utils import Language, SymbolFormat
from text_utils.pronunciation.main import EngToIPAMode
from text_utils.symbols_map import create_or_update_inference_map
from tts_preparation.app.inference import (add_text, ipa_convert_text,
                                           normalize_text)
from tts_preparation.app.mapping import create_or_update_inference_map_main


def main():
  merge_name = "debug_ljs_test"

  create_or_update_inference_map_main(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    # template_map=Path("maps/inference/eng_ipa.json"),
  )

  # mode = 2
  # if mode == 1:
  #   create_or_update_weights_map_main(
  #     base_dir="/datasets/models/taco2pt_v5",
  #     merge_name="ljs_ipa",
  #     weights_merge_name="thchs"
  #   )
  # elif mode == 2:
  #   create_or_update_inference_map_main(
  #     base_dir="/datasets/models/taco2pt_v5",
  #     merge_name="ljs_ipa",
  #     template_map="maps/inference/eng_ipa.json"
  #   )


if __name__ == "__main__":
  main()
