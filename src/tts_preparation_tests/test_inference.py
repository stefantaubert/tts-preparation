# from logging import getLogger

# from tts_preparation.core.inference import *


# # def test_all():
# #   example_text = "This is a test. And an other one.\nAnd a new line.\r\nAnd a line with \r.\n\nAnd a line with \n in it. This is a question? This is a error!"
# #   #example_text = read_text("examples/en/democritus.txt")
# #   conv = SymbolIdDict.init_from_symbols({"T", "h", "i", "s"})
# #   sents = add_text(example_text, Language.ENG, conv)
# #   print(sents)
# #   sents = sents_normalize(sents)
# #   print(sents)
# #   #sents = sents_map(sents, symbols_map=SymbolsMap.from_tuples([("o", "b"), ("a", ".")]))
# #   print(sents)
# #   sents = sents_convert_to_ipa(sents, ignore_tones=True, ignore_arcs=True)
# #   print(sents)
# def test_add_text__chn__splits_sents():
#   _, sentences = add_text("暖耀着。旅行者。", lang=Language.CHN, logger=getLogger())
#   self.assertEqual(2, len(sentences))


# def test_get_formatted_core():
#   symbols = list("this is a test!")
#   accent_ids = list("000016834864123")
#   accent_id_dict = AccentsDict.init_from_accents({f"a{i}" for i in range(10)})
#   sent_id = 1
#   max_pairs = 4
#   spacelength = 1

#   res = get_formatted_core(
#     sent_id,
#     symbols,
#     accent_ids,
#     max_pairs_per_line=max_pairs,
#     space_length=spacelength,
#     accent_id_dict=accent_id_dict
#   )

#   self.assertEqual(
#     '1: t h i s\n   0 0 0 0\n   0=a0\n   \n     i s  \n   1 6 8 3\n   1=a1, 3=a3, 6=a6, 8=a8\n   \n   a   t e\n   4 8 6 4\n   4=a4, 6=a6, 8=a8\n   \n   s t ! (15)\n   1 2 3\n   1=a1, 2=a2, 3=a3\n   ', res)


# def test_get_formatted():
#   sents = SentenceList([
#     Sentence(0, "", 0, "1,2", "2,1"),
#     Sentence(1, "", 0, "0", "0")
#   ])
#   symbol_ids = SymbolIdDict.init_from_symbols({"a", "b", "c"})
#   accent_ids = AccentsDict.init_from_accents({"a1", "a2", "a3"})

#   res = sents.get_formatted(symbol_ids, accent_ids)
#   self.assertEqual('0: b c (2)\n   2 1\n   1=a2, 2=a3\n   \n1: a (1)\n   0\n   0=a1\n   ', res)


# def test_sents_accent_template():
#   sents = SentenceList([
#     Sentence(0, "", 0, "1,2", "2,1"),
#     Sentence(0, "", 0, "0", "0")
#   ])
#   symbol_ids = SymbolIdDict.init_from_symbols({"a", "b", "c"})
#   accent_ids = AccentsDict.init_from_accents({"a1", "a2", "a3"})

#   res = sents_accent_template(sents, symbol_ids, accent_ids)

#   self.assertEqual(3, len(res))

#   self.assertEqual("0-0", res.items()[0].position)
#   self.assertEqual("b", res.items()[0].symbol)
#   self.assertEqual("a3", res.items()[0].accent)

#   self.assertEqual("0-1", res.items()[1].position)
#   self.assertEqual("c", res.items()[1].symbol)
#   self.assertEqual("a2", res.items()[1].accent)

#   self.assertEqual("1-0", res.items()[2].position)
#   self.assertEqual("a", res.items()[2].symbol)
#   self.assertEqual("a1", res.items()[2].accent)


# def test_sents_accent_apply():
#   sents = SentenceList([
#     Sentence(0, "", 0, "1,2", "2,1"),
#     Sentence(0, "", 0, "0", "0")
#   ])
#   symbol_ids = SymbolIdDict.init_from_symbols({"a", "b", "c"})
#   accent_ids = AccentsDict.init_from_accents({"a1", "a2", "a3"})
#   acc_sents_template = sents_accent_template(sents, symbol_ids, accent_ids)

#   acc_sents_template.items()[0].accent = "a1"
#   acc_sents_template.items()[1].accent = "a3"
#   acc_sents_template.items()[2].accent = "a2"

#   res = sents_accent_apply(sents, acc_sents_template, accent_ids)

#   self.assertEqual(2, len(sents))
#   self.assertEqual("0,2", res.items()[0].serialized_accents)
#   self.assertEqual("1", res.items()[1].serialized_accents)
