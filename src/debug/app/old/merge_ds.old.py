from tts_preparation.app.merge_ds import ds_filter_symbols, merge_ds
from tts_preparation.globals_debug import BASE_DIR, SDP_DIR


def create_debug_ljs():
  merge_name = "debug_ljs"
  merge_ds(
    base_dir=BASE_DIR,
    sdp_dir=SDP_DIR,
    merge_name=merge_name,
    ds_speakers=[("ljs", "all")],
    #ds_text_audio=[("thchs", "ipa", "22050Hz_normalized_nosil")]
    ds_text_audio=[("ljs", "ipa_norm_epi", "22050Hz")],
    overwrite=True,
  )

  ds_filter_symbols(
    base_dir=BASE_DIR,
    orig_merge_name=merge_name,
    dest_merge_name=merge_name,
    allowed_symbol_ids={
      0, 1, 2, 6, 8, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49
    },
    overwrite=True,
  )


if __name__ == "__main__":
  create_debug_ljs()

  merge_ds(
    base_dir=BASE_DIR,
    sdp_dir=SDP_DIR,
    merge_name="debug",
    ds_speakers=[("ljs", "all")],
    #ds_text_audio=[("thchs", "ipa", "22050Hz_normalized_nosil")]
    ds_text_audio=[("ljs", "ipa_norm_epi", "22050Hz")],
    overwrite=True,
  )

  ds_filter_symbols(
    base_dir=BASE_DIR,
    orig_merge_name="debug",
    dest_merge_name="debug",
    allowed_symbol_ids={
      0, 1, 2, 6, 8, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49
    },
    overwrite=True,
  )

  merge_ds(
    base_dir=BASE_DIR,
    merge_name="ljs_ipa_epi",
    ds_speakers=[("ljs", "all")],
    ds_text_audio=[("ljs", "ipa_norm_epi", "22050Hz")],
    overwrite=True,
  )

  ds_filter_symbols(
    base_dir=BASE_DIR,
    orig_merge_name="ljs_ipa_epi",
    dest_merge_name="ljs_ipa_epi_filtered",
    allowed_symbol_ids={
      0, 1, 2, 6, 8, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49
    },
    overwrite=True,
  )

  merge_ds(
    base_dir=BASE_DIR,
    merge_name="debug",
    ds_speakers=[("ljs", "all")],
    #ds_text_audio=[("thchs", "ipa", "22050Hz_normalized_nosil")]
    ds_text_audio=[("ljs", "ipa_norm_epi", "22050kHz")],
    overwrite=True,
  )

  # merge_ds(
  #   base_dir="/datasets/models/taco2pt_v5",
  #   prep_name="thchs_ljs_ipa",
  #   ds_speakers=[("ljs", "all"), ("thchs", "all")],
  #   ds_text_audio=[("ljs", "ipa_norm", "22050Hz"), ("thchs", "ipa", "22050kHz_normalized_nosil")]
  # )

  # merge_ds(
  #   base_dir="/datasets/models/taco2pt_v5",
  #   prep_name="ljs",
  #   ds_speakers=[("ljs", "all")],
  #   ds_text_audio=[("ljs", "ipa_norm", "22050Hz")]
  # )

  # merge_ds(
  #   base_dir="/datasets/models/taco2pt_v5",
  #   prep_name="thchs",
  #   ds_speakers=[("thchs", "all")],
  #   ds_text_audio=[("thchs", "ipa", "22050Hz_norm_wo_sil")]
  # )
