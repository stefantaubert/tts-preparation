import string

from debug.globals import BASE_DIR, SDP_DIR
from tts_preparation.app.merge_ds import ds_filter_symbols, merge_ds


def main():
  merge_ds(
    base_dir=BASE_DIR,
    sdp_dir=SDP_DIR,
    ds_speakers=[("debug_ljs_test", "Linda Johnson")],
    ds_text_audio=[("debug_ljs_test", "en", "eng")],
    overwrite=True,
    merge_name="debug_ljs_test",
  )

  ds_filter_symbols(
    base_dir=BASE_DIR,
    allowed_symbols=set(string.printable),
    orig_merge_name="debug_ljs_test",
    dest_merge_name="debug_ljs_test2",
    overwrite=True,
  )


if __name__ == "__main__":
  main()
