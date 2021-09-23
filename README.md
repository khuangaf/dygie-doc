# DyGIE-Doc

This project extends [DyGIE++](https://github.com/dwadden/dygiepp), a general IE framework, to handle document-level IE tasks. Specifically, it establishes benchmarks for __end-to-end document-level relation extraction__ on three well-known datasets: DocRED, CDR, and GDA. A demo of this project can be found [here](#).


## Table of Contents
- [Dependencies](#dependencies)
- [Model training](#training-a-model)
- [Model evaluation](#evaluating-a-model)
- [Pretrained models](#pretrained-models)
- [Making predictions on existing datasets](#making-predictions-on-existing-datasets)
- [Contact](#contact)

See the `doc` folder for documentation with more details on the [data](doc/data.md), [model implementation and debugging](doc/model.md), and [model configuration](doc/config.md).


## Dependencies

Clone this repository and navigate the the root of the repo on your system. Then execute:

```
conda create --name dygiepp python=3.7
pip install -r requirements.txt
conda develop .   # Adds DyGIE to your PYTHONPATH
```

Similar to DyGIE++, this codebase also relies on [AllenNLP](https://allennlp.org) and uses AllenNLP shell [commands](https://docs.allennlp.org/master/#package-overview) to for training models and making predictions.

If you run into an issue installing `jsonnet`, [this issue](https://github.com/allenai/allennlp/issues/2779) may prove helpful.

## Training a model

The training procedure is exactly the same as the original DyGIE++. Please refer to [this section](https://github.com/dwadden/dygiepp#training-a-model) for details. In short, to train a model, simply run:

```
bash scripts/train.sh CONFIG
```
, where `CONFIG.jsonnet` is the configuration file defined under the `training_config` directory.

### DocRED

The original DocRED dataset did not release the test set annotation. Therefore, we split the dev set into `devdev` and `devtest` sets. The corresponding dockeys for each `devdev` and `devtest` are defined in `scripts/data/docred/dev-dev_dockey.list` and `scripts/data/docred/dev-test_dockey.list`.

- **Download the data**. From the top-level folder for this repo, run `bash ./scripts/data/get_docred.sh`.
- **Preprocess the data**. Run `bash ./scripts/data/docred/process_docred.sh`. This will produce mention-level data `*.json` for training DyGIE++ as well as entity-level data `*.entlvl.json` for final evaluation in `data/docred/processed-data`.
- **Train the model**. Run `bash scripts/train.sh docred`. The default encoder is `bert-base-cased`, but you can easily replace the `bert-base-cased` with `roberta-base` for slightly better performance.

### CDR & GDA

The data pre-processing and model training for CDR & GDA are mostly the same, and both of them very simlar to DocRED's. Below shows how CDR data is pre-process and how a model can be trained on CDR.

- **Download the data**. From the top-level folder for this repo, enter `bash ./scripts/data/get_cdr.sh`.
- **Train the model**. Enter `bash scripts/train cdr`.
- As with SciERC, we also offer a "lightweight" version with a context width of 1 and no coreference propagation.


### GDA


## Evaluating a model

To evaluate sentence-level IE tasks, such as NER, relation extraction and event extraction, please refer to [how DyGIE++ evaluate a model](https://github.com/dwadden/dygiepp#evaluating-a-model). For end-to-end document-level relation extraction, it's currently implemented as a separate evaluator in `eval.py`. You need to pass the __entity-level__ predictions from the model and the `entity-level` gold data to the script as follows:

```
python eval.py --pred-file PATH_TO_PREDICTON_FILE.entlvl.json --gold-file PATH_TO_GOLD_FILE.entlvl.json
```
***Note that passing mention-level predictions (output from `allennlp predict`) to the evaluator would not work.***
[Below](#making-predictions) describes how you make predictions.


## Pretrained models

A number of models are available for download. They are named for the dataset they are trained on. "Lightweight" models are models trained on datasets for which coreference resolution annotations were available, but we didn't use them. This is "lightweight" because coreference resolution is expensive, since it requires predicting cross-sentence relationships between spans.

If you want to use one of these pretrained models to make predictions on a new dataset, you need to set the `dataset` field for the instances in your new dataset to match the name of the `dataset` the model was trained on. For example, to make predictions using the pretrained SciERC model, set the `dataset` field in your new instances to `scierc`. For more information on the `dataset` field, see [data.md](doc/data.md).

To download all available models, run `scripts/pretrained/get_dygiepp_pretrained.sh`. Or, click on the links below to download only a single model.

### Available models (coming soon)

Below are links to the available models, followed by the name of the `dataset` the model was trained on.

- [DocRED](#): 
- [CDR](#): 
- [GDA](#): 


### Performance of pretrained models

| Dataset      | Entity F1 | Relation F1 |
| ----------- | ----------- | -----------|
| DocRED     | 87.56       | 51.15 |
| CDR   | 79.07        | 50.90|
| GDA   | 83.63        |  76.88 |




## Making predictions

To make mention-level predictions, you can use `allennlp predict`. Details are described in the original [DyGIE++ repo](https://github.com/dwadden/dygiepp#making-predictions-on-existing-datasets).

Mention-level relation predictions can be gathered into entity-level relation predictions with `postprocess.py`, which aggreegates mention-level relations between every pairs of entities with voting. A typical usage is:

```
python postprocess.py  --input-path predictions/cdr.test.json  --output-path predictions/cdr.test.entlvl.json
```
where `predictions/cdr.test.json` is the output from the `allennlp predict` command.


# Contact

For questions or problems with the code, create a GitHub issue (preferred) or email `khhuang3@illinois.edu`.
