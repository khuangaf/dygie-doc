import logging
from typing import Any, Dict, List, Optional, Callable
import numpy as np

import torch
import torch.nn.functional as F
from overrides import overrides

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import util, RegularizerApplicator
from allennlp.modules import TimeDistributed

from dygie.training.document_relation_metrics import DocumentRelationMetrics
from dygie.models.entity_beam_pruner import Pruner
from dygie.data.dataset_readers import document

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



class DocumentRelationExtractor(Model):
    """
    Relation extraction module of DyGIE model.
    """
    # TODO(dwadden) add option to make `mention_feedforward` be the NER tagger.

    def __init__(self,
                 vocab: Vocabulary,
                 make_feedforward: Callable,
                 span_emb_dim: int,
                 feature_size: int,
                 spans_per_word: float,
                 positive_label_weight: float = 1.0,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._namespaces = [entry for entry in vocab.get_namespaces() if "document_relation_labels" in entry]
        self._n_labels = {name: vocab.get_vocab_size(name) for name in self._namespaces}

        self._mention_pruners = torch.nn.ModuleDict()
        self._relation_feedforwards = torch.nn.ModuleDict()
        self._relation_scorers = torch.nn.ModuleDict()
        self._relation_metrics = {}

        for namespace in self._namespaces:
            mention_feedforward = make_feedforward(input_dim=span_emb_dim)
            feedforward_scorer = torch.nn.Sequential(
                TimeDistributed(mention_feedforward),
                TimeDistributed(torch.nn.Linear(mention_feedforward.get_output_dim(), 1)))
            self._mention_pruners[namespace] = Pruner(feedforward_scorer)

            relation_scorer_dim = 3 * span_emb_dim
            relation_feedforward = make_feedforward(input_dim=relation_scorer_dim)
            self._relation_feedforwards[namespace] = relation_feedforward
            relation_scorer = torch.nn.Linear(
                relation_feedforward.get_output_dim(), self._n_labels[namespace])
            self._relation_scorers[namespace] = relation_scorer

            self._relation_metrics[namespace] = DocumentRelationMetrics()

        self._spans_per_word = spans_per_word
        self._active_namespace = None

        self._loss = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)

    @overrides
    def forward(self,  # type: ignore
                spans: torch.IntTensor,
                span_mask,
                span_embeddings,  # TODO(dwadden) add type.
                sentence_lengths,
                predicted_coref, 
                document_relation_labels: torch.IntTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        """
        We assume the input to this module has consecutive spans that correspond to the same document.

        """
        self._active_namespace = f"{metadata.dataset}__document_relation_labels"
        

        # create batch dimensions for labels bc the labels have been flattened
        document_relation_labels = document_relation_labels.unsqueeze(0)

        # flatten span embedding 
        doc_span_embeddings = span_embeddings[torch.where(span_mask > 0)].view(-1, span_embeddings.size(-1)).unsqueeze(0)

        # compute document lengths for determining how many mentions to keep
        doc_length = sentence_lengths.sum().unsqueeze(0)
        
        # Original spans have offsets w.r.t. each sentence. Compute document-level spans w.r.t. document.
        span_plus_sentence = spans.clone()
        sentence_starts = np.cumsum(sentence_lengths.cpu())
        sentence_starts = np.roll(sentence_starts, 1)
        sentence_starts[0] = 0
        sentence_starts = sentence_starts.tolist()
        
        for sent_ix, sentence_start in enumerate(sentence_starts):
            span_plus_sentence[sent_ix] = span_plus_sentence[sent_ix] + sentence_start
        doc_spans = span_plus_sentence[torch.where(span_mask > 0)].view(-1, 2).unsqueeze(0) #(1, num_doc_spans, 2)

        doc_span_mask = torch.ones_like(doc_span_embeddings)[:,:,0] # all the spans are valid bc we have already filtered the invalid using torch.where(span_mask > 0)
        # if training, use gold coref
        entity_embeddings = []
        if self.training:
            clusters = metadata.cluster_list
        else:
            clusters = predicted_coref
        
        for cluster in clusters:
            this_entity_embeddings = []
            for span in cluster:
                mention_span_mask = doc_spans[0,:,0]==span[0]  * doc_spans[0,:,0]==span[1] # (num_doc_spans)
                mention_embeddings = doc_span_embeddings[:,mention_span_mask, :] #(1, 1 emb_dim)
                assert mention_embeddings.size(1) == 1, mention_embeddings.size()
                this_entity_embeddings.append(mention_embeddings)
            this_entity_embeddings = torch.cat(this_entity_embeddings, dim=1).max(dim=1)[0] # (1, emb_dim)
            entity_embeddings.append(this_entity_embeddings)

        entity_embeddings = entity_embeddings.cat(entity_embeddings, dim=0).unsqueeze(0) # (1, num_entities, emb_dim)


        relation_scores = self._compute_relation_scores(
            self._compute_span_pair_embeddings(entity_embeddings))

        prediction_dict, predictions = self.predict(relation_scores.detach().cpu(), metadata)

        output_dict = {"predictions": predictions}

        # Evaluate loss and F1 if labels were provided.
        if document_relation_labels is not None:
            # Compute cross-entropy loss.
            # gold_relations = self._get_pruned_gold_relations(
            #     document_relation_labels, top_span_indices, top_span_mask)
            # TODO (steeve): not sure if this is correct
            gold_relations =  document_relation_labels[0]
            cross_entropy = self._get_cross_entropy_loss(relation_scores, gold_relations)

            # Compute F1.
            # assert len(prediction_dict) == len(metadata)  # Make sure length of predictions is right.
            relation_metrics = self._relation_metrics[self._active_namespace]
            relation_metrics(prediction_dict, metadata)

            output_dict["loss"] = cross_entropy
        return output_dict

    def _prune_spans(self, spans, span_mask, span_embeddings, sentence_lengths):
        # Prune
        num_spans = spans.size(1)  # Max number of spans for the minibatch.

        # Keep different number of spans for each minibatch entry.
        num_spans_to_keep = torch.ceil(sentence_lengths.float() * self._spans_per_word).long()
        
        pruner = self._mention_pruners[self._active_namespace]
        (top_span_embeddings, top_span_mask,
         top_span_indices, top_span_mention_scores, num_spans_kept) = pruner(
             span_embeddings, span_mask, num_spans_to_keep)

        top_span_mask = top_span_mask.unsqueeze(-1)

        flat_top_span_indices = util.flatten_and_batch_shift_indices(top_span_indices, num_spans)
        top_spans = util.batched_index_select(spans,
                                              top_span_indices,
                                              flat_top_span_indices)

        return top_span_embeddings, top_span_mention_scores, num_spans_to_keep, top_span_mask, top_span_indices, top_spans

    def predict(self, relation_scores, metadata):
        '''
        pred_dict_doc: a dictionary
        predictions_doc: a list of relations
        '''
        # predictions = []
        doc = metadata
        
        
        pred_dict_doc, predictions_doc = self._predict_document(
            relation_scores[0], doc)
            
        # predictions.append(predictions_doc)

        return pred_dict_doc, predictions_doc

    def _predict_document(self, relation_scores, doc):
        # keep = num_spans_to_keep.item()
        # top_spans = [tuple(x) for x in top_spans.tolist()]

        # Iterate over all span pairs and labels. Record the span if the label isn't null.
        predicted_scores_raw, predicted_labels = relation_scores.max(dim=-1)
        softmax_scores = F.softmax(relation_scores, dim=-1)
        predicted_scores_softmax, _ = softmax_scores.max(dim=-1)
        predicted_labels -= 1  # Subtract 1 so that null labels get -1.

        # keep_mask = torch.zeros(len(top_spans))
        # keep_mask[:keep] = 1
        # keep_mask = keep_mask.bool()

        # ix = (predicted_labels >= 0) & keep_mask
        ix = (predicted_labels >= 0) 

        res_dict = {}
        predictions = []

        for i, j in ix.nonzero(as_tuple=False):
            # span_1 = top_spans[i]
            # span_2 = top_spans[j]
            label = predicted_labels[i, j].item()
            raw_score = predicted_scores_raw[i, j].item()
            softmax_score = predicted_scores_softmax[i, j].item()

            label_name = self.vocab.get_token_from_index(label, namespace=self._active_namespace)
            res_dict[(i, j)] = label_name
            list_entry = (i, j, label_name, raw_score, softmax_score)
            predictions.append(document.PredictedDocumentRelation(list_entry, doc.sentences))

        return res_dict, predictions

    # TODO(dwadden) This code is repeated elsewhere. Refactor.
    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        "Loop over the metrics for all namespaces, and return as dict."
        res = {}
        for namespace, metrics in self._relation_metrics.items():
            precision, recall, f1 = metrics.get_metric(reset)
            prefix = namespace.replace("_labels", "")
            # prefix = 'document_' + prefix
            to_update = {f"{prefix}_precision": precision,
                         f"{prefix}_recall": recall,
                         f"{prefix}_f1": f1}
            res.update(to_update)

        res_avg = {}
        for name in ["precision", "recall", "f1"]:
            values = [res[key] for key in res if name in key]
            res_avg[f"MEAN__document_relation_{name}"] = sum(values) / len(values) if values else 0
            res.update(res_avg)

        return res

    @staticmethod
    def _compute_span_pair_embeddings(top_span_embeddings: torch.FloatTensor):
        """
        TODO(dwadden) document me and add comments.
        """
        # Shape: (batch_size, num_spans_to_keep, num_spans_to_keep, embedding_size)
        num_candidates = top_span_embeddings.size(1)

        embeddings_1_expanded = top_span_embeddings.unsqueeze(2)
        embeddings_1_tiled = embeddings_1_expanded.repeat(1, 1, num_candidates, 1)

        embeddings_2_expanded = top_span_embeddings.unsqueeze(1)
        embeddings_2_tiled = embeddings_2_expanded.repeat(1, num_candidates, 1, 1)

        similarity_embeddings = embeddings_1_expanded * embeddings_2_expanded

        pair_embeddings_list = [embeddings_1_tiled, embeddings_2_tiled, similarity_embeddings]
        pair_embeddings = torch.cat(pair_embeddings_list, dim=3)

        return pair_embeddings

    def _compute_relation_scores(self, pairwise_embeddings):
        relation_feedforward = self._relation_feedforwards[self._active_namespace]
        relation_scorer = self._relation_scorers[self._active_namespace]

        batch_size = pairwise_embeddings.size(0)
        max_num_spans = pairwise_embeddings.size(1)
        feature_dim = relation_feedforward.input_dim

        embeddings_flat = pairwise_embeddings.view(-1, feature_dim)

        relation_projected_flat = relation_feedforward(embeddings_flat)
        relation_scores_flat = relation_scorer(relation_projected_flat)

        relation_scores = relation_scores_flat.view(batch_size, max_num_spans, max_num_spans, -1)


        shape = [relation_scores.size(0), relation_scores.size(1), relation_scores.size(2), 1]
        dummy_scores = relation_scores.new_zeros(*shape)

        relation_scores = torch.cat([dummy_scores, relation_scores], -1)
        return relation_scores

    @staticmethod
    def _get_pruned_gold_relations(relation_labels, top_span_indices, top_span_masks):
        """
        Loop over each slice and get the labels for the spans from that slice.
        All labels are offset by 1 so that the "null" label gets class zero. This is the desired
        behavior for the softmax. Labels corresponding to masked relations keep the label -1, which
        the softmax loss ignores.
        """
        # TODO(dwadden) Test and possibly optimize.
        relations = []

        zipped = zip(relation_labels, top_span_indices, top_span_masks.bool())
        for sliced, ixs, top_span_mask in zipped:
            entry = sliced[ixs][:, ixs].unsqueeze(0)
            mask_entry = top_span_mask & top_span_mask.transpose(0, 1).unsqueeze(0)
            entry[mask_entry] += 1
            entry[~mask_entry] = -1
            relations.append(entry)

        return torch.cat(relations, dim=0)

    def _get_cross_entropy_loss(self, relation_scores, relation_labels):
        """
        Compute cross-entropy loss on relation labels. Ignore diagonal entries and entries giving
        relations between masked out spans.
        """
        # Need to add one for the null class.
        n_labels = self._n_labels[self._active_namespace] + 1
        scores_flat = relation_scores.view(-1, n_labels)
        # Need to add 1 so that the null label is 0, to line up with indices into prediction matrix.
        labels_flat = relation_labels.view(-1)
        # Compute cross-entropy loss.
        loss = self._loss(scores_flat, labels_flat)
        return loss
