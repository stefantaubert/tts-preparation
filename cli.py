import os
from argparse import ArgumentParser

from text_utils import EngToIpaMode, Language

from tts_preparation.app.inference import (accent_apply, accent_set, add_text,
                                           ipa_convert_text, map_text,
                                           map_to_prep_symbols, normalize_text)
from tts_preparation.app.mapping import (create_or_update_inference_map_main,
                                         create_or_update_weights_map_main)
from tts_preparation.app.merge_ds import ds_filter_symbols, merge_ds
from tts_preparation.app.prepare import (add_ngram, add_ngram_kld,
                                         add_random_count)
from tts_preparation.app.prepare2 import (app_add_greedy_kld_ngram_minutes,
                                          app_add_greedy_ngram_epochs,
                                          app_add_greedy_ngram_minutes,
                                          app_add_ngram_cover,
                                          app_add_random_minutes,
                                          app_add_random_ngram_cover_minutes,
                                          app_add_random_percent, app_add_rest,
                                          app_add_symbols, app_prepare,
                                          print_and_save_stats)
from tts_preparation.core.data import DatasetType
from tts_preparation.utils import parse_tuple_list, split_int_set_str

BASE_DIR_VAR = "base_dir"


def add_base_dir(parser: ArgumentParser):
  assert BASE_DIR_VAR in os.environ.keys()
  base_dir = os.environ[BASE_DIR_VAR]
  parser.set_defaults(base_dir=base_dir)


def _add_parser_to(subparsers, name: str, init_method):
  parser = subparsers.add_parser(name, help=f"{name} help")
  invoke_method = init_method(parser)
  parser.set_defaults(invoke_handler=invoke_method)
  add_base_dir(parser)
  return parser


def init_merge_ds_parser(parser: ArgumentParser):
  parser.add_argument('--sdp_dir', type=str, required=True)
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--ds_speakers', type=str, required=True)
  parser.add_argument('--ds_text_audio', type=str, required=True)
  return merge_ds_cli


def merge_ds_cli(**args):
  args["ds_speakers"] = parse_tuple_list(args["ds_speakers"])
  args["ds_text_audio"] = parse_tuple_list(args["ds_text_audio"])
  merge_ds(**args)


def init_merge_ds_filter_symbols_parser(parser: ArgumentParser):
  parser.add_argument('--orig_merge_name', type=str, required=True)
  parser.add_argument('--dest_merge_name', type=str, required=True)
  parser.add_argument('--allowed_symbol_ids', type=str, required=True)
  parser.set_defaults(overwrite=True)
  return ds_filter_symbols_cli


def ds_filter_symbols_cli(**args):
  args["allowed_symbol_ids"] = split_int_set_str(args["allowed_symbol_ids"])
  ds_filter_symbols(**args)


# def init_split_ds_parser(parser: ArgumentParser):
#   parser.add_argument('--merge_name', type=str, required=True)
#   parser.add_argument('--prep_name', type=str, required=True)
#   parser.add_argument('--validation_size', type=float, default=0.1)
#   parser.add_argument('--test_size', type=float, default=0.001)
#   parser.add_argument('--split_seed', type=int, default=1234)
#   return split_dataset


def init_prepare_ds_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--prep_name', type=str, required=True)
  parser.set_defaults(overwrite=True)
  return app_prepare


def init_prepare_ds_add_rest_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.set_defaults(overwrite=True)
  return app_add_rest


def init_prepare_ds_print_and_save_stats_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--prep_name', type=str, required=True)
  return print_and_save_stats


# def init_prepare_ds_add_random_percent_parser(parser: ArgumentParser):
#   parser.add_argument('--merge_name', type=str, required=True)
#   parser.add_argument('--orig_prep_name', type=str, required=True)
#   parser.add_argument('--dest_prep_name', type=str, required=True)
#   parser.add_argument('--percent', type=float, required=True)
#   parser.add_argument('--seed', type=int, default=1234)
#   parser.add_argument('--dataset', choices=DatasetType,
#                       type=DatasetType.__getitem__)
#   parser.set_defaults(overwrite=True)
#   return add_random_percent


def init_prepare_ds_add_random_percent_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--percent', type=float, required=True)
  parser.add_argument('--seed', type=int)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.set_defaults(overwrite=True)
  return app_add_random_percent


def init_prepare_ds_add_random_minutes(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--minutes', type=float, required=True)
  parser.add_argument('--seed', type=int)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.add_argument('--respect_existing', action='store_true')
  parser.set_defaults(overwrite=True)
  return app_add_random_minutes


def init_prepare_ds_add_random_count_parser_legacy(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--shards_per_speaker', type=int, required=True)
  parser.add_argument('--seed', type=int, default=1234)
  parser.add_argument('--ignore_already_added', action='store_true')
  parser.add_argument('--min_count_symbol', type=int, default=0)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.set_defaults(overwrite=True)
  return add_random_count


def init_prepare_ds_add_ngram_kld_parser_legacy(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--n_gram', type=int, required=True)
  parser.add_argument('--shards_per_speaker', type=int)
  parser.add_argument('--n_its', type=int)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.add_argument('--ignore_already_added', action='store_true')
  parser.add_argument('--min_count_symbol', type=int, default=0)
  parser.add_argument('--top_percent', type=float, default=100)
  parser.set_defaults(overwrite=True)
  return add_ngram_kld


def init_prepare_ds_add_ngram_parser_legacy(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--n_gram', type=int, required=True)
  parser.add_argument('--shards_per_speaker', type=int)
  parser.add_argument('--n_its', type=int)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.add_argument('--ignore_already_added', action='store_true')
  parser.add_argument('--min_count_symbol', type=int, default=0)
  parser.add_argument('--top_percent', type=float, default=100)
  parser.set_defaults(overwrite=True)
  return add_ngram


# def init_prepare_ds_add_greedy_ngram_epochs_parser(parser: ArgumentParser):
#   parser.add_argument('--merge_name', type=str, required=True)
#   parser.add_argument('--orig_prep_name', type=str, required=True)
#   parser.add_argument('--dest_prep_name', type=str, required=True)
#   parser.add_argument('--n_gram', type=int, required=True)
#   parser.add_argument('--epochs', type=int)
#   parser.add_argument('--dataset', choices=DatasetType,
#                       type=DatasetType.__getitem__)
#   parser.set_defaults(overwrite=True)
#   return app_add_ngram_greedy_epochs

def init_prepare_ds_add_ngram_cover_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.add_argument('--n_gram', type=int, required=True)
  parser.add_argument('--ignore_symbol_ids', type=str)
  parser.add_argument('--top_percent', type=float)
  parser.set_defaults(overwrite=True)
  return app_add_ngram_cover_cli


def app_add_ngram_cover_cli(**args):
  args["ignore_symbol_ids"] = split_int_set_str(args["ignore_symbol_ids"])
  app_add_ngram_cover(**args)


def init_prepare_ds_add_random_ngram_cover_minutes_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.add_argument('--n_gram', type=int, required=True)
  parser.add_argument('--ignore_symbol_ids', type=str)
  parser.add_argument('--seed', type=int, required=True)
  parser.add_argument('--minutes', type=float, required=True)
  parser.set_defaults(overwrite=True)
  return app_add_random_ngram_cover_minutes_cli


def app_add_random_ngram_cover_minutes_cli(**args):
  args["ignore_symbol_ids"] = split_int_set_str(args["ignore_symbol_ids"])
  app_add_random_ngram_cover_minutes(**args)


def init_prepare_ds_add_ngrams_kld_minutes_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.add_argument('--n_gram', type=int, required=True)
  parser.add_argument('--ignore_symbol_ids', type=str)
  parser.add_argument('--minutes', type=float, required=True)
  parser.set_defaults(overwrite=True)
  return app_add_ngram_greedy_kld_minutes_cli


def app_add_ngram_greedy_kld_minutes_cli(**args):
  args["ignore_symbol_ids"] = split_int_set_str(args["ignore_symbol_ids"])
  app_add_greedy_kld_ngram_minutes(**args)


def init_prepare_ds_add_symbols_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--cover_symbol_ids', type=str)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.set_defaults(overwrite=True)
  return app_add_symbols_cli


def app_add_symbols_cli(**args):
  args["cover_symbol_ids"] = split_int_set_str(args["cover_symbol_ids"])
  app_add_symbols(**args)


def init_prepare_ds_add_ngram_epochs_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.add_argument('--n_gram', type=int, required=True)
  parser.add_argument('--epochs', type=int, required=True)
  parser.add_argument('--ignore_symbol_ids', type=str)
  parser.set_defaults(overwrite=True)
  return app_add_greedy_ngram_epochs_cli


def app_add_greedy_ngram_epochs_cli(**args):
  args["ignore_symbol_ids"] = split_int_set_str(args["ignore_symbol_ids"])
  app_add_greedy_ngram_epochs(**args)


def init_prepare_ds_add_ngram_minute_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.add_argument('--n_gram', type=int, required=True)
  parser.add_argument('--ignore_symbol_ids', type=str)
  parser.add_argument('--minutes', type=float, required=True)
  parser.set_defaults(overwrite=True)
  return app_add_greedy_ngram_minutes_cli


def app_add_greedy_ngram_minutes_cli(**args):
  args["ignore_symbol_ids"] = split_int_set_str(args["ignore_symbol_ids"])
  app_add_greedy_ngram_minutes(**args)


def init_create_or_update_weights_map_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True,
                      help="The prepared name for the model which will be trained.")
  parser.add_argument('--weights_merge_name', type=str, required=True,
                      help="The prepared name of which were used by the pretrained model.")
  parser.add_argument('--template_map', type=str)
  return create_or_update_weights_map_main


def init_create_or_update_inference_map_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--template_map', type=str)
  return create_or_update_inference_map_main


def init_add_text_parser(parser: ArgumentParser):
  parser.add_argument('--filepath', type=str, required=False)
  parser.add_argument('--text', type=str, required=False)
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  parser.add_argument('--lang', choices=Language, type=Language.__getitem__, required=True)
  return add_text


def init_normalize_text_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  return normalize_text


def init_convert_to_ipa_text_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  parser.add_argument('--ignore_tones', action='store_true')
  #parser.add_argument('--ignore_arcs', action='store_true')
  parser.add_argument('--mode', choices=EngToIpaMode,
                      type=EngToIpaMode.__getitem__)
  parser.set_defaults(ignore_arcs=True)
  return ipa_convert_text


def init_accent_apply_text_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  return accent_apply


def init_accent_set_text_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  parser.add_argument('--accent', type=str, required=True)
  return accent_set


def init_map_text_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  parser.add_argument('--symbols_map_path', type=str, required=True)
  parser.set_defaults(ignore_arcs=True)
  return map_text


def init_map_to_prep_symbols_parser(parser: ArgumentParser):
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  parser.set_defaults(ignore_arcs=True)
  return map_to_prep_symbols


def _init_parser():
  result = ArgumentParser()
  subparsers = result.add_subparsers(help='sub-command help')

  _add_parser_to(subparsers, "merge-ds", init_merge_ds_parser)
  _add_parser_to(subparsers, "merge-ds-filter", init_merge_ds_filter_symbols_parser)
  # _add_parser_to(subparsers, "split-ds", init_split_ds_parser)
  _add_parser_to(subparsers, "prepare-ds", init_prepare_ds_parser)
  _add_parser_to(subparsers, "prepare-ds-add-symbols", init_prepare_ds_add_symbols_parser)
  _add_parser_to(subparsers, "prepare-ds-add-ngram", init_prepare_ds_add_ngram_parser_legacy)
  _add_parser_to(subparsers, "prepare-ds-add-ngram-minutes",
                 init_prepare_ds_add_ngram_minute_parser)
  _add_parser_to(subparsers, "prepare-ds-add-ngram-epochs",
                 init_prepare_ds_add_ngram_epochs_parser)
  _add_parser_to(subparsers, "prepare-ds-add-ngram-cover",
                 init_prepare_ds_add_ngram_cover_parser)
  #_add_parser_to(subparsers, "prepare-ds-add-ngram-epochs",init_prepare_ds_add_greedy_ngram_epochs_parser)
  _add_parser_to(subparsers, "prepare-ds-add-ngram-kld", init_prepare_ds_add_ngram_parser_legacy)
  _add_parser_to(subparsers, "prepare-ds-add-ngram-kld-minutes",
                 init_prepare_ds_add_ngrams_kld_minutes_parser)
  _add_parser_to(subparsers, "prepare-ds-add-random-count",
                 init_prepare_ds_add_random_count_parser_legacy)
  _add_parser_to(subparsers, "prepare-ds-add-random-percent",
                 init_prepare_ds_add_random_percent_parser)
  _add_parser_to(subparsers, "prepare-ds-add-random-minutes",
                 init_prepare_ds_add_random_minutes)
  _add_parser_to(subparsers, "prepare-ds-add-ngram-random-cover-minutes",
                 init_prepare_ds_add_random_ngram_cover_minutes_parser)
  _add_parser_to(subparsers, "prepare-ds-add-rest", init_prepare_ds_add_rest_parser)
  _add_parser_to(subparsers, "prepare-ds-stats", init_prepare_ds_print_and_save_stats_parser)

  _add_parser_to(subparsers, "inference-text-add", init_add_text_parser)
  _add_parser_to(subparsers, "inference-text-normalize", init_normalize_text_parser)
  _add_parser_to(subparsers, "inference-text-to-ipa", init_convert_to_ipa_text_parser)
  _add_parser_to(subparsers, "inference-text-set-accent", init_accent_set_text_parser)
  _add_parser_to(subparsers, "inference-text-apply-accents", init_accent_apply_text_parser)
  _add_parser_to(subparsers, "inference-text-map", init_map_text_parser)
  _add_parser_to(subparsers, "inference-text-automap", init_map_to_prep_symbols_parser)
  _add_parser_to(subparsers, "inference-create-map", init_create_or_update_inference_map_parser)
  _add_parser_to(subparsers, "merged-ds-weights-map", init_create_or_update_weights_map_parser)

  return result


def _process_args(args):
  params = vars(args)
  invoke_handler = params.pop("invoke_handler")
  invoke_handler(**params)


if __name__ == "__main__":
  main_parser = _init_parser()

  received_args = main_parser.parse_args()
  #args = main_parser.parse_args("ljs-text --base_dir=/datasets/models/taco2pt_v2 --mel_name=ljs --ds_name=test_ljs --convert_to_ipa".split())

  _process_args(received_args)
