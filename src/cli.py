from text_utils import StringFormat2
from collections import OrderedDict
from pronunciation_dict_parser import PublicDictType
from logging import getLogger
import logging
import os
from argparse import ArgumentParser
from pathlib import Path
import re
import sys
from typing import List

from general_utils import parse_tuple_list, split_str_set_symbols
from text_utils import EngToIPAMode, Language, SymbolFormat

from tts_preparation.app import (add_text, app_add_greedy_kld_ngram_minutes,
                                 app_add_greedy_ngram_epochs,
                                 app_add_greedy_ngram_minutes,
                                 app_add_n_diverse_random_minutes,
                                 app_add_ngram_cover, app_add_random_minutes,
                                 app_add_random_ngram_cover_minutes,
                                 app_add_random_percent, app_add_rest,
                                 app_add_symbols,
                                 app_get_random_seconds_divergent_seeds,
                                 app_prepare, apply_mapping_table,
                                 change_ipa_text,
                                 create_or_update_inference_map_main,
                                 create_or_update_weights_map_main,
                                 ds_filter_symbols, ipa_convert_text, map_text,
                                 merge_ds, normalize_text,
                                 print_and_save_stats, split_text)
from tts_preparation.app.export import (export_audios,
                                        export_for_text_selection,
                                        import_from_selection)
from tts_preparation.app.inference import arpa_convert_text, change_text, export_text, map_arpa_to_ipa
from tts_preparation.core import DatasetType

BASE_DIR_VAR = "base_dir"


def add_base_dir(parser: ArgumentParser) -> None:
  assert BASE_DIR_VAR in os.environ.keys()
  base_dir = Path(os.environ[BASE_DIR_VAR])
  parser.set_defaults(base_dir=base_dir)


def _add_parser_to(subparsers, name: str, init_method) -> None:
  parser = subparsers.add_parser(name, help=f"{name} help")
  invoke_method = init_method(parser)
  parser.set_defaults(invoke_handler=invoke_method)
  add_base_dir(parser)
  return parser


def init_merge_ds_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--sdp_dir', type=Path, required=True)
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--ds_speakers', type=str, required=True)
  parser.add_argument('--ds_final_name', type=str, required=True)
  return merge_ds_cli


def merge_ds_cli(**args) -> None:
  args["ds_speakers"] = parse_tuple_list(args["ds_speakers"])
  args["ds_final_name"] = parse_tuple_list(args["ds_final_name"])
  merge_ds(**args)


def init_merge_ds_filter_symbols_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--orig_merge_name', type=str, required=True)
  parser.add_argument('--dest_merge_name', type=str, required=True)
  parser.add_argument('--allowed_symbols', type=str, required=True)
  parser.set_defaults(overwrite=True)
  return ds_filter_symbols_cli


def ds_filter_symbols_cli(**args) -> None:
  args["allowed_symbols"] = split_str_set_symbols(args["allowed_symbols"])
  ds_filter_symbols(**args)


def init_prepare_ds_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--prep_name', type=str, required=True)
  parser.set_defaults(overwrite=True)
  return app_prepare


def init_prepare_ds_add_rest_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.set_defaults(overwrite=True)
  return app_add_rest


def init_prepare_ds_print_and_save_stats_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--prep_name', type=str, required=True)
  return print_and_save_stats


def init_prepare_ds_add_random_percent_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--percent', type=float, required=True)
  parser.add_argument('--seed', type=int)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.set_defaults(overwrite=True)
  return app_add_random_percent


def init_prepare_ds_add_n_diverse_random_minutes(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--minutes', type=float, required=True)
  parser.add_argument('--n', type=int)
  parser.add_argument('--seed', type=int)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.set_defaults(overwrite=True)
  return app_add_n_diverse_random_minutes


def init_prepare_ds_add_random_minutes(parser: ArgumentParser) -> None:
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

def init_prepare_ds_add_ngram_cover_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.add_argument('--n_gram', type=int, required=True)
  parser.add_argument('--ignore_symbols', type=str)
  parser.add_argument('--top_percent', type=float)
  parser.set_defaults(overwrite=True)
  return app_add_ngram_cover_cli


def app_add_ngram_cover_cli(**args) -> None:
  args["ignore_symbols"] = split_str_set_symbols(args["ignore_symbols"])
  app_add_ngram_cover(**args)


def init_prepare_ds_add_random_ngram_cover_minutes_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.add_argument('--n_gram', type=int, required=True)
  parser.add_argument('--ignore_symbols', type=str)
  parser.add_argument('--seed', type=int, required=True)
  parser.add_argument('--minutes', type=float, required=True)
  parser.set_defaults(overwrite=True)
  return app_add_random_ngram_cover_minutes_cli


def app_add_random_ngram_cover_minutes_cli(**args) -> None:
  args["ignore_symbols"] = split_str_set_symbols(args["ignore_symbols"])
  app_add_random_ngram_cover_minutes(**args)


def init_prepare_ds_add_ngrams_kld_minutes_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.add_argument('--n_gram', type=int, required=True)
  parser.add_argument('--ignore_symbols', type=str, nargs="*", default=[])
  parser.add_argument('--minutes', type=float, required=True)
  parser.set_defaults(overwrite=True)
  return app_add_ngram_greedy_kld_minutes_cli


def app_add_ngram_greedy_kld_minutes_cli(**args) -> None:
  args["ignore_symbols"] = set(args["ignore_symbols"])
  app_add_greedy_kld_ngram_minutes(**args)


def init_export_audios_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.add_argument('--output-directory', type=Path, required=True)
  parser.add_argument('--overwrite', action='store_true')
  return export_audios


def init_export_for_text_selection_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--output-directory', type=Path, required=True)
  parser.add_argument('--overwrite', action='store_true')
  return export_for_text_selection


def init_import_from_selection_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--import-directory', type=Path, required=True)
  return import_from_selection


def init_prepare_ds_add_symbols_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--cover_symbols', type=str)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.set_defaults(overwrite=True)
  return app_add_symbols_cli


def app_add_symbols_cli(**args) -> None:
  args["cover_symbols"] = split_str_set_symbols(args["cover_symbols"])
  app_add_symbols(**args)


def init_prepare_ds_add_ngram_epochs_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.add_argument('--n_gram', type=int, required=True)
  parser.add_argument('--epochs', type=int, required=True)
  parser.add_argument('--ignore_symbols', type=str)
  parser.set_defaults(overwrite=True)
  return app_add_greedy_ngram_epochs_cli


def app_add_greedy_ngram_epochs_cli(**args) -> None:
  args["ignore_symbols"] = split_str_set_symbols(args["ignore_symbols"])
  app_add_greedy_ngram_epochs(**args)


def init_prepare_ds_add_ngram_minute_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--orig_prep_name', type=str, required=True)
  parser.add_argument('--dest_prep_name', type=str, required=True)
  parser.add_argument('--dataset', choices=DatasetType,
                      type=DatasetType.__getitem__)
  parser.add_argument('--n_gram', type=int, required=True)
  parser.add_argument('--ignore_symbols', type=str)
  parser.add_argument('--minutes', type=float, required=True)
  parser.set_defaults(overwrite=True)
  return app_add_greedy_ngram_minutes_cli


def app_add_greedy_ngram_minutes_cli(**args) -> None:
  args["ignore_symbols"] = split_str_set_symbols(args["ignore_symbols"])
  app_add_greedy_ngram_minutes(**args)


def init_get_random_seconds_divergent_seeds_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--prep_name', type=str, required=True)
  parser.add_argument('--minutes', type=float, required=True)
  parser.add_argument('--seed', type=int, required=True)
  parser.add_argument('--samples', type=int, required=True)
  parser.add_argument('--n', type=int, required=True)
  return app_get_random_seconds_divergent_seeds


def init_create_or_update_weights_map_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True,
                      help="The prepared name for the model which will be trained.")
  parser.add_argument('--weights_merge_name', type=str, required=True,
                      help="The prepared name of which were used by the pretrained model.")
  parser.add_argument('--template_map', type=str)
  return create_or_update_weights_map_main


def init_create_or_update_inference_map_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--template_map', type=Path)
  return create_or_update_inference_map_main


def init_add_text_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--text_filepath', type=Path, required=False)
  parser.add_argument('--text', type=str, required=False)
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  parser.add_argument('--language', choices=Language, type=Language.__getitem__, required=True)
  parser.add_argument('--text_format', choices=SymbolFormat,
                      type=SymbolFormat.__getitem__, required=True)
  parser.add_argument('--string_format', choices=StringFormat2,
                      type=StringFormat2.__getitem__, required=True)
  return add_text


def init_split_text_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  return split_text


def init_export_text_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  parser.add_argument('--string_format', choices=StringFormat2,
                      type=StringFormat2.__getitem__, default=StringFormat2.SPACED)
  return export_text


def init_normalize_text_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  return normalize_text


def init_convert_to_ipa_text_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  parser.add_argument('--consider_annotations', action='store_true')
  parser.add_argument('--mode', choices=EngToIPAMode,
                      type=EngToIPAMode.__getitem__)
  return ipa_convert_text


def init_convert_to_arpa_text_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  parser.add_argument('--consider_annotations', action='store_true')
  parser.set_defaults(dictionary=PublicDictType.MFA_ARPA)
  return arpa_convert_text


def add_dictionary_argument(parser: ArgumentParser) -> None:
  names = OrderedDict((
    (PublicDictType.MFA_ARPA, "MFA"),
    (PublicDictType.CMU_ARPA, "CMU"),
    (PublicDictType.LIBRISPEECH_ARPA, "LibriSpeech"),
    (PublicDictType.PROSODYLAB_ARPA, "Prosodylab"),
  ))

  values_to_names = dict(zip(
    names.values(),
    names.keys()
  ))

  help_str = "pronunciation dictionary (ARPAbet) which should be used to look up the words; if a pronunciation is not available it will be estimated"
  parser.add_argument(
    "-d", "--dictionary",
    metavar=list(names.values()),
    choices=names.keys(),
    type=values_to_names.get,
    default=names[PublicDictType.MFA_ARPA],
    help=help_str,
  )


def init_change_ipa_text_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  parser.add_argument('--ignore_tones', action='store_true')
  parser.add_argument('--ignore_arcs', action='store_true')
  parser.add_argument('--ignore_stress', action='store_true')
  parser.add_argument('--break_n_thongs', action='store_true')
  parser.add_argument('--build_n_thongs', action='store_true')
  return change_ipa_text


def init_map_arpa_to_ipa_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  return map_arpa_to_ipa


def init_change_text_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  parser.add_argument('--remove_space_around_punctuation', action='store_true')
  return change_text


def init_apply_mapping_table_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  parser.add_argument('--mapping_table_path', type=Path, required=True)
  parser.add_argument('--seed', type=int, required=True)
  return apply_mapping_table


def init_map_text_parser(parser: ArgumentParser) -> None:
  parser.add_argument('--merge_name', type=str, required=True)
  parser.add_argument('--text_name', type=str, required=True)
  parser.add_argument('--symbols_map_path', type=Path, required=False)
  return map_text


def _init_parser():
  result = ArgumentParser()
  subparsers = result.add_subparsers(help='sub-command help')

  _add_parser_to(subparsers, "merge-ds", init_merge_ds_parser)
  _add_parser_to(subparsers, "merge-ds-filter", init_merge_ds_filter_symbols_parser)
  # _add_parser_to(subparsers, "split-ds", init_split_ds_parser)
  _add_parser_to(subparsers, "prepare-ds", init_prepare_ds_parser)
  _add_parser_to(subparsers, "prepare-ds-add-symbols", init_prepare_ds_add_symbols_parser)
  _add_parser_to(subparsers, "prepare-ds-add-ngram-minutes",
                 init_prepare_ds_add_ngram_minute_parser)
  _add_parser_to(subparsers, "prepare-ds-add-ngram-epochs",
                 init_prepare_ds_add_ngram_epochs_parser)
  _add_parser_to(subparsers, "prepare-ds-add-ngram-cover",
                 init_prepare_ds_add_ngram_cover_parser)
  #_add_parser_to(subparsers, "prepare-ds-add-ngram-epochs",init_prepare_ds_add_greedy_ngram_epochs_parser)
  _add_parser_to(subparsers, "prepare-ds-add-ngram-kld-minutes",
                 init_prepare_ds_add_ngrams_kld_minutes_parser)
  _add_parser_to(subparsers, "prepare-ds-add-random-percent",
                 init_prepare_ds_add_random_percent_parser)
  _add_parser_to(subparsers, "prepare-ds-get-random-minutes-divergence-seeds",
                 init_get_random_seconds_divergent_seeds_parser)
  _add_parser_to(subparsers, "prepare-ds-add-random-minutes",
                 init_prepare_ds_add_random_minutes)
  _add_parser_to(subparsers, "prepare-ds-add-n-diverse-random-minutes",
                 init_prepare_ds_add_n_diverse_random_minutes)
  _add_parser_to(subparsers, "prepare-ds-add-ngram-random-cover-minutes",
                 init_prepare_ds_add_random_ngram_cover_minutes_parser)
  _add_parser_to(subparsers, "prepare-ds-add-rest", init_prepare_ds_add_rest_parser)
  _add_parser_to(subparsers, "prepare-ds-stats", init_prepare_ds_print_and_save_stats_parser)

  _add_parser_to(subparsers, "inference-text-add", init_add_text_parser)
  _add_parser_to(subparsers, "inference-text-split", init_split_text_parser)
  _add_parser_to(subparsers, "inference-text-normalize", init_normalize_text_parser)
  _add_parser_to(subparsers, "inference-text-change-text", init_change_text_parser)
  _add_parser_to(subparsers, "inference-text-to-arpa", init_convert_to_arpa_text_parser)
  _add_parser_to(subparsers, "inference-text-to-ipa", init_convert_to_ipa_text_parser)
  _add_parser_to(subparsers, "inference-text-change-ipa", init_change_ipa_text_parser)
  _add_parser_to(subparsers, "inference-text-map", init_map_text_parser)
  _add_parser_to(subparsers, "inference-text-arpa-to-ipa", init_map_arpa_to_ipa_parser)
  _add_parser_to(subparsers, "inference-text-apply-mapping-table", init_apply_mapping_table_parser)
  _add_parser_to(subparsers, "inference-create-map", init_create_or_update_inference_map_parser)
  _add_parser_to(subparsers, "inference-text-export", init_export_text_parser)
  _add_parser_to(subparsers, "merged-ds-weights-map", init_create_or_update_weights_map_parser)

  _add_parser_to(subparsers, "export-audio", init_export_audios_parser)
  _add_parser_to(subparsers, "export-for-selection", init_export_for_text_selection_parser)
  _add_parser_to(subparsers, "import-from-selection", init_import_from_selection_parser)

  return result


def configure_logger() -> None:
  loglevel = logging.DEBUG if __debug__ else logging.INFO
  main_logger = getLogger()
  main_logger.setLevel(loglevel)
  main_logger.manager.disable = logging.NOTSET
  if len(main_logger.handlers) > 0:
    console = main_logger.handlers[0]
  else:
    console = logging.StreamHandler()
    main_logger.addHandler(console)

  logging_formatter = logging.Formatter(
    '[%(asctime)s.%(msecs)03d] (%(levelname)s) %(message)s',
    '%Y/%m/%d %H:%M:%S',
  )
  console.setFormatter(logging_formatter)
  console.setLevel(loglevel)


def parse_args(args: List[str]):
  configure_logger()
  logger = getLogger(__name__)
  if BASE_DIR_VAR in os.environ:
    logger.debug(f"base_dir: {os.environ[BASE_DIR_VAR]}")
  logger.debug(f"Received args: {str(args)}")

  parser = _init_parser()
  received_args = parser.parse_args(args)

  params = vars(received_args)
  invoke_handler = params.pop("invoke_handler")
  invoke_handler(**params)


if __name__ == "__main__":
  arguments = sys.argv[1:]
  parse_args(arguments)
