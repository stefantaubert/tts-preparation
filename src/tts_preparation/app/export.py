
from logging import getLogger
from pathlib import Path
from shutil import copy2, rmtree

from text_utils import StringFormat
from tts_preparation.app.io import get_merged_dir
from tts_preparation.app.prepare import get_prep_dir, load_set
from tts_preparation.core.data import DatasetType


def export_audios(base_dir: Path, merge_name: str, prep_name: str, dataset: DatasetType, output_directory: Path, overwrite: bool) -> None:
  logger = getLogger(__name__)
  logger.info(f"Exporting audio files {str(dataset)}...")
  merge_dir = get_merged_dir(base_dir, merge_name)
  prep_dir = get_prep_dir(merge_dir, prep_name)

  if output_directory.exists():
    if overwrite:
      rmtree(output_directory)
      logger.info("Removed existing output directory.")
    else:
      logger.error("Output directory already exists!")

  output_directory.mkdir(parents=True, exist_ok=False)

  ds = load_set(prep_dir, dataset)
  for entry in ds.items_tqdm():
    assert entry.wav_absolute_path.is_file()
    output_wav_path = output_directory / f"{entry.entry_id}.wav"
    output_text_path = output_directory / f"{entry.entry_id}.txt"
    text = StringFormat.SYMBOLS.convert_symbols_to_string(entry.symbols)

    copy2(entry.wav_absolute_path, output_wav_path, follow_symlinks=False)
    output_text_path.write_text(text, encoding="UTF-8")
