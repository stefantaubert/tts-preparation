from tts_preparation.app.inference import *
from tts_preparation.app.mapping import (create_or_update_inference_map_main,
                                         get_infer_symbols_path)
from tts_preparation.globals_debug import BASE_DIR


def add_text_debug_ljs():
  merge_name = "debug_ljs"
  text_name = "test_sents"

  add_text(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    text_name=text_name,
    text_filepath="examples/en/test_sents.txt",
    language=Language.ENG,
  )

  normalize_text(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    text_name=text_name,
  )

  ipa_convert_text(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    text_name=text_name,
    mode=EngToIpaMode.EPITRAN,
    ignore_arcs=False,
    ignore_tones=False,
  )

  create_or_update_inference_map_main(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    template_map=None,
  )

  merge_dir = get_merged_dir(
    base_dir=BASE_DIR,
    merge_name=merge_name,
  )
  infer_map_path = get_infer_map_path(merge_dir)
  infer_symbols_path = get_infer_symbols_path(merge_dir)

  change_symbols_in_map(
    map_path=infer_map_path,
    symbol_path=infer_symbols_path,
    arrow_type="inference",
    to_key="-",
    map_symbol=" ",
  )

  map_to_prep_symbols(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    text_name=text_name,
    ignore_arcs=True,
  )


if __name__ == "__main__":
  apply_mapping_table(
    base_dir=BASE_DIR,
    merge_name="nnlv_pilot_phd1",
    text_name="eng-north",
    mapping_table_path=Path("/home/mi/code/accent-analyser/out/word_probs.csv"),
    seed=1234,
  )

  add_text_debug_ljs()

  mode = 0
  if mode == 0:
    add_text(
      base_dir=BASE_DIR,
      merge_name="debug",
      text_name="coma",
      text_filepath="examples/en/coma.txt",
      language=Language.ENG,
    )

    ipa_convert_text(
      base_dir=BASE_DIR,
      merge_name="debug",
      text_name="coma",
      mode=EngToIpaMode.EPITRAN,
    )
    add_text(
      base_dir=BASE_DIR,
      merge_name="debug",
      text_name="stella",
      text_filepath="examples/en/stella.txt",
      language=Language.ENG,
    )

    add_text(
      base_dir=BASE_DIR,
      merge_name="debug",
      text_name="north",
      text_filepath="examples/en/north.txt",
      language=Language.ENG,
    )

    add_text(
      base_dir=BASE_DIR,
      merge_name="debug",
      text_name="test_sents",
      text_filepath="examples/en/test_sents.txt",
      language=Language.ENG,
    )

  if mode == 1:
    add_text(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="arctic_ipa",
      text_name="north",
      text_filepath="examples/en/north.txt",
      language=Language.ENG,
    )

    normalize_text(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="arctic_ipa",
      text_name="north",
    )

    accent_set(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="arctic_ipa",
      text_name="north",
      accent="Chinese-BWC"
    )

    ipa_convert_text(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="arctic_ipa",
      text_name="north",
    )

    # create_or_update_inference_map(
    #   base_dir="/datasets/models/taco2pt_v5",
    #   prep_name="arctic_ipa",
    #   dest=None,
    #   existing_map=None,
    #   orig=None,
    #   template_map=None,
    # )

    # map_text(
    #   base_dir="/datasets/models/taco2pt_v5",
    #   prep_name="arctic_ipa",
    #   text_name="north",
    #   symbols_map="",
    # )

    map_to_prep_symbols(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="arctic_ipa",
      text_name="north"
    )

  elif mode == 2:
    accent_apply(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="arctic_ipa",
      text_name="north",
    )
  elif mode == 3:
    add_text(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="ljs_ipa",
      text_name="ipa-north_sven_orig",
      text_filepath="examples/ipa/north_sven_orig.txt",
      language=Language.IPA,
    )

    normalize_text(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="ljs_ipa",
      text_name="ipa-north_sven_orig",
    )

  elif mode == 4:
    add_text(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="ljs_ipa",
      text_name="en-coma",
      text_filepath="examples/en/coma.txt",
      language=Language.ENG,
    )

    normalize_text(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="ljs_ipa",
      text_name="en-coma",
    )

    ipa_convert_text(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="ljs_ipa",
      text_name="en-coma",
    )

    map_to_prep_symbols(
      base_dir="/datasets/models/taco2pt_v5",
      prep_name="ljs_ipa",
      text_name="en-coma",
    )
