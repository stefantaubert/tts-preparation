from tts_preparation.app.merge_ds import get_merged_dir
from tts_preparation.app.prepare import app_add_random_percent, app_add_rest, app_prepare, load_trainset
from tts_preparation.core.data import DatasetType
from debug.globals_debug import BASE_DIR


def create_debug_ljs():
  merge_name = "debug_ljs"
  prep_name = "default"

  app_prepare(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    prep_name=prep_name,
  )

  app_add_random_percent(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    orig_prep_name=prep_name,
    dest_prep_name=prep_name,
    dataset=DatasetType.TEST,
    seed=1111,
    percent=98,
  )

  app_add_random_percent(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    orig_prep_name=prep_name,
    dest_prep_name=prep_name,
    dataset=DatasetType.VALIDATION,
    seed=1111,
    percent=5,
  )

  app_add_rest(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    orig_prep_name=prep_name,
    dest_prep_name=prep_name,
    dataset=DatasetType.TRAINING,
  )


def compare_random_approaches():
  n_sets = 3
  minutes = 420

  merge_name = "debug_ljs"
  prep_name = "debug"

  app_prepare(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    prep_name=prep_name,
  )

  app_add_n_diverse_random_minutes(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    orig_prep_name=prep_name,
    dest_prep_name=prep_name,
    dataset=DatasetType.TRAINING,
    minutes=minutes,
    n=n_sets,
    overwrite=True,
    seed=1111,
  )

  for i in range(n_sets):
    app_add_random_minutes(
      base_dir=BASE_DIR,
      merge_name=merge_name,
      orig_prep_name=prep_name,
      dest_prep_name=f"{prep_name}_{i+1}_normal",
      dataset=DatasetType.TRAINING,
      minutes=minutes,
      overwrite=True,
      seed=1111 + i,
    )

  merge_dir = get_merged_dir(
    base_dir=BASE_DIR,
    merge_name=merge_name,
    create=False
  )

  trainsets = [
   {x.entry_id for x in load_trainset(
    get_prep_dir(merge_dir, f"{prep_name}_{i+1}", create=False)).items()}
      for i in range(n_sets)
  ]

  res = get_total_number_of_common_elements(trainsets)
  print(f"Overlapping (app_add_n_divergent_random_seconds): {res}")

  trainsets = [
   {x.entry_id for x in load_trainset(
    get_prep_dir(merge_dir, f"{prep_name}_{i+1}_normal", create=False)).items()}
      for i in range(n_sets)
  ]

  res = get_total_number_of_common_elements(trainsets)
  print(f"Overlapping (normal): {res}")


if __name__ == "__main__":
  create_debug_ljs()

  compare_random_approaches()

  merge_dir = get_merged_dir(
    base_dir=BASE_DIR,
    merge_name="debug",
    create=False
  )

  trainsets = [
   {x.entry_id for x in load_trainset(
    get_prep_dir(merge_dir, "random_8h_1440", create=False)).items()},
   {x.entry_id for x in load_trainset(
    get_prep_dir(merge_dir, "random_8h_1440", create=False)).items()},
   {x.entry_id for x in load_trainset(
    get_prep_dir(merge_dir, "random_8h_1440", create=False)).items()},
   {x.entry_id for x in load_trainset(
    get_prep_dir(merge_dir, "random_8h_1440", create=False)).items()},
   {x.entry_id for x in load_trainset(
    get_prep_dir(merge_dir, "random_8h_1440", create=False)).items()},
  ]

  print("all same")
  res = get_total_number_of_common_elements(trainsets)
  print(res)

  trainsets = [
   {x.entry_id for x in load_trainset(
    get_prep_dir(merge_dir, "random_8h_xxx_4350", create=False)).items()},
   {x.entry_id for x in load_trainset(
    get_prep_dir(merge_dir, "random_8h_xxx_1475", create=False)).items()},
   {x.entry_id for x in load_trainset(
    get_prep_dir(merge_dir, "random_8h_xxx_3276", create=False)).items()},
   {x.entry_id for x in load_trainset(
    get_prep_dir(merge_dir, "random_8h_xxx_776", create=False)).items()},
   {x.entry_id for x in load_trainset(
    get_prep_dir(merge_dir, "random_8h_xxx_4143", create=False)).items()},
  ]

  res = get_total_number_of_common_elements(trainsets)
  print("all random")
  print(res)

  trainsets = [
   {x.entry_id for x in load_trainset(
    get_prep_dir(merge_dir, "random_8h_1440", create=False)).items()},
   {x.entry_id for x in load_trainset(
    get_prep_dir(merge_dir, "random_8h_1657", create=False)).items()},
   {x.entry_id for x in load_trainset(
    get_prep_dir(merge_dir, "random_8h_3951", create=False)).items()},
   {x.entry_id for x in load_trainset(
    get_prep_dir(merge_dir, "random_8h_4102", create=False)).items()},
   {x.entry_id for x in load_trainset(
    get_prep_dir(merge_dir, "random_8h_4985", create=False)).items()},
  ]

  res = get_total_number_of_common_elements(trainsets)
  print("all random divergence")
  print(res)

  # create_debug_ljs()

  app_prepare(
    base_dir=BASE_DIR,
    merge_name="debug_ljs",
    prep_name="debug",
  )

  app_get_random_seconds_divergent_seeds(
    base_dir=BASE_DIR,
    merge_name="debug_ljs",
    prep_name="debug",
    minutes=60,
    samples=100,
    n=5,
    seed=1111,
  )

  app_add_random_minutes(
    base_dir=BASE_DIR,
    merge_name="debug_ljs",
    orig_prep_name="debug",
    dest_prep_name="debug",
    dataset=DatasetType.TRAINING,
    minutes=60,
    respect_existing=False,
    seed=23,
  )

  app_prepare(
    base_dir=BASE_DIR,
    merge_name="debug_ljs",
    prep_name="debug",
  )

  app_get_random_seconds_divergent_seeds(
    base_dir=BASE_DIR,
    merge_name="debug_ljs",
    prep_name="debug",
    minutes=60,
    samples=100,
    n=5,
    seed=1111,
  )

  app_prepare(
    base_dir=BASE_DIR,
    merge_name="debug",
    prep_name="debug",
  )

  app_get_random_seconds_divergent_seeds(
    base_dir=BASE_DIR,
    merge_name="debug",
    prep_name="debug",
    minutes=6,
    samples=100,
    n=5,
    seed=1111,
  )

  app_add_greedy_ngram_epochs(
    base_dir=BASE_DIR,
    merge_name="debug",
    orig_prep_name="debug",
    dest_prep_name="debug",
    n_gram=1,
    epochs=5,
    dataset=DatasetType.TEST,
  )

  app_add_ngram_cover(
    base_dir=BASE_DIR,
    merge_name="debug",
    orig_prep_name="debug",
    dest_prep_name="debug",
    n_gram=2,
    top_percent=0.65,
    dataset=DatasetType.TEST,
  )

  app_add_ngram_cover(
    base_dir=BASE_DIR,
    merge_name="debug",
    orig_prep_name="debug",
    dest_prep_name="debug",
    n_gram=3,
    top_percent=0.30,
    dataset=DatasetType.TEST,
  )

  app_add_random_minutes(
    base_dir=BASE_DIR,
    merge_name="debug",
    orig_prep_name="debug",
    dest_prep_name="debug",
    dataset=DatasetType.TEST,
    minutes=30,
    respect_existing=True,
    seed=1111,
  )

  print_and_save_stats(
    base_dir=BASE_DIR,
    merge_name="debug",
    prep_name="debug",
  )

  # prepare(
  #   base_dir=BASE_DIR,
  #   merge_name="debug",
  #   prep_name="default_init",
  # )

  app_add_random_ngram_cover_minutes(
    base_dir=BASE_DIR,
    merge_name="debug",
    orig_prep_name="debug",
    dest_prep_name="debug2",
    dataset=DatasetType.TEST,
    n_gram=1,
    ignore_symbol_ids={3},
    minutes=60,
    seed=1234,
  )

  app_add_greedy_kld_ngram_minutes(
    base_dir=BASE_DIR,
    merge_name="debug",
    orig_prep_name="debug2",
    dest_prep_name="debug2",
    dataset=DatasetType.TEST,
    n_gram=1,
    ignore_symbol_ids={3},
    minutes=1,
  )

  app_add_greedy_ngram_minutes(
    base_dir=BASE_DIR,
    merge_name="debug",
    orig_prep_name="debug2",
    dest_prep_name="debug2",
    dataset=DatasetType.TEST,
    n_gram=1,
    ignore_symbol_ids={3},
    minutes=10,
  )

  app_add_symbols(
    base_dir=BASE_DIR,
    merge_name="debug",
    orig_prep_name="debug2",
    dest_prep_name="debug2",
    dataset=DatasetType.TEST,
    cover_symbol_ids={3},
  )

  app_add_random_percent(
    base_dir=BASE_DIR,
    merge_name="debug",
    orig_prep_name="debug2",
    dest_prep_name="debug2",
    dataset=DatasetType.TEST,
    seed=1234,
    percent=20,
  )

  print_and_save_stats(
    base_dir=BASE_DIR,
    merge_name="debug",
    prep_name="debug2",
  )
