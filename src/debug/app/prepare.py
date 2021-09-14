from debug.globals import BASE_DIR
from tts_preparation.app.prepare import (app_add_random_percent, app_add_rest,
                                         app_prepare)
from tts_preparation.core.data import DatasetType


def main():
  merge_name = "debug_ljs_test2"
  prep_name = "debug"

  app_prepare(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    prep_name=prep_name,
    overwrite=True,
  )

  app_add_random_percent(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    orig_prep_name=prep_name,
    dest_prep_name=prep_name,
    dataset=DatasetType.TEST,
    seed=1111,
    percent=98,
    overwrite=True,
  )

  app_add_random_percent(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    orig_prep_name=prep_name,
    dest_prep_name=prep_name,
    dataset=DatasetType.VALIDATION,
    seed=1111,
    percent=5,
    overwrite=True,
  )

  app_add_rest(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    orig_prep_name=prep_name,
    dest_prep_name=prep_name,
    dataset=DatasetType.TRAINING,
    overwrite=True,
  )


if __name__ == "__main__":
  main()
