# Remote Dependencies

- text-utils
  - pronunciation_dict_parser
  - g2p_en
  - sentence2pronunciation
- speech-dataset-preprocessing
  - speech-dataset-parser
  - text-utils
  - audio-utils
  - image-utils
- accent-analyser
  - text-utils
- text-selection
- sentence2pronunciation

## Pipfile

### Local

```Pipfile
text-utils = {editable = true, path = "./../text-utils"}
pronunciation_dict_parser = {editable = true, path = "./../pronunciation_dict_parser"}
g2p_en = {editable = true, path = "./../g2p"}
sentence2pronunciation = {editable = true, path = "./../sentence2pronunciation"}
speech-dataset-preprocessing = {editable = true, path = "./../speech-dataset-preprocessing"}
speech-dataset-parser = {editable = true, path = "./../speech-dataset-parser"}
audio-utils = {editable = true, path = "./../audio-utils"}
image-utils = {editable = true, path = "./../image-utils"}
accent-analyser = {editable = true, path = "./../accent-analyser"}
text-selection = {editable = true, path = "./../text-selection"}
```

### Remote

```Pipfile
text-utils = {editable = true, ref = "master", git = "https://github.com/stefantaubert/text-utils.git"}
speech-dataset-preprocessing = {editable = true, ref = "master", git = "https://github.com/stefantaubert/speech-dataset-preprocessing.git"}
accent-analyser = {editable = true, ref = "master", git = "https://github.com/stefantaubert/accent-analyser.git"}
text-selection = {editable = true, ref = "master", git = "https://github.com/stefantaubert/text-selection.git"}
sentence2pronunciation = {editable = true, ref = "main", git = "https://github.com/jasminsternkopf/sentence2pronunciation.git"}
```

## setup.cfg

```cfg
text_utils@git+https://github.com/stefantaubert/text-utils.git@master
speech_dataset_preprocessing@git+https://github.com/stefantaubert/speech-dataset-preprocessing.git@master
text_selection@git+https://github.com/stefantaubert/text-selection.git@master
accent_analyser@git+https://github.com/stefantaubert/accent-analyser.git@master
sentence2pronunciation@git+https://github.com/jasminsternkopf/sentence2pronunciation@main
```
