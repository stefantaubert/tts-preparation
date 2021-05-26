

from text_selection.greedy_export import greedy_ngrams_epochs

from tts_preparation.app.merge_ds import (get_merged_dir,
                                          load_merged_symbol_converter)
from tts_preparation.app.prepare import get_prep_dir, load_valset
from tts_preparation.core.data import PreparedDataList
from tts_preparation.core.helper import (prep_data_list_to_dict_with_symbols,
                                         select_entities_from_prep_data)
from tts_preparation.globals_debug import BASE_DIR

merge_name = "txt_sel_exp"

merge_dir = get_merged_dir(
  base_dir=BASE_DIR,
  merge_name=merge_name,
  create=False
)

prep_name = "valset"
prep_dir = get_prep_dir(
  merged_dir=merge_dir,
  prep_name=prep_name,
)

symbols = load_merged_symbol_converter(merge_dir)

valset = load_valset(prep_dir)

sentence_ids = {
  367, 430, 632, 658, 787, 951, 966, 1120, 1474, 1541, 1547, 1560, 1640, 1670, 1846, 1969, 2060, 2076, 2151, 2173, 2309, 2682, 2808, 2942, 3452, 3594, 3645, 3738, 3826, 3844, 3851, 3963, 4101, 4114, 4147, 4187, 4320, 4446, 4595, 4683, 4882, 4940, 4992, 5182, 5458, 5591, 5974, 6140, 6281, 6283, 6312, 6601, 6661, 6691, 6922, 6967, 7012, 7097, 7262, 7285, 7306, 7409, 7479, 7501, 7568, 7903, 7959, 7979, 8268, 8972, 9317, 9468, 9473, 9534, 9537, 9756
}

# sentence_ids.remove(1474)
sentence_ids.remove(6281) # only names
# sentence_ids.remove(2808)

sentence_entries = PreparedDataList([x for x in valset.items() if x.entry_id in sentence_ids])

sentence_dict = prep_data_list_to_dict_with_symbols(
  l=sentence_entries,
  symbols=symbols,
)

res = greedy_ngrams_epochs(sentence_dict, n_gram=1, epochs=1, ignore_symbols=None)
res_entries = select_entities_from_prep_data(res, valset)
for x in res_entries.items():
  print(x.entry_id, x.text_original)

for i in res:
  sentence_dict.pop(i)

res = greedy_ngrams_epochs(sentence_dict, n_gram=1, epochs=1, ignore_symbols=None)
res_entries = select_entities_from_prep_data(res, valset)
for x in res_entries.items():
  print(x.entry_id, x.text_original)

for i in res:
  sentence_dict.pop(i)


res = greedy_ngrams_epochs(sentence_dict, n_gram=1, epochs=1, ignore_symbols=None)
res_entries = select_entities_from_prep_data(res, valset)
for x in res_entries.items():
  print(x.entry_id, x.text_original)
