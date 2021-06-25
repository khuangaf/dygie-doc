local template = import "template.libsonnet";

template.DyGIE {
  bert_model: "allenai/longformer-base-4096",
  cuda_device: 2,
  data_paths: {
    train: "data/gda/processed-data/train.json",
    validation: "data/gda/processed-data/dev.json",
    test: "data/gda/processed-data/test.json",
  },
  loss_weights: {
    ner: 0.2,
    relation: 0.0,
    coref: 1.0,
    events: 0.0,
    document_relation: 1.0,
    document_events:0.0,
    event_coref: 0.0,

  },
  target_task: "document_relation",
  encode_document: true,
  window_size: 5,
  gradient_checkpointing:true,
  max_tokens_per_sentence:100,

}
