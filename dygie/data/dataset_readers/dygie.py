import logging
from typing import Any, Dict, List, Optional, Tuple, DefaultDict, Set, Union
import json
import pickle as pkl
import warnings
import numpy as np
from copy import deepcopy
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (ListField, TextField, SpanField, MetadataField,
                                  SequenceLabelField, AdjacencyField, LabelField, IndexField)
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans

from dygie.data.fields.adjacency_field_assym import AdjacencyFieldAssym
from dygie.data.dataset_readers.document import ClusterMember, Document, Sentence

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DyGIEDataException(Exception):
    pass


@DatasetReader.register("dygie")
class DyGIEReader(DatasetReader):
    """
    Reads a single JSON-formatted file. This is the same file format as used in the
    scierc, but is preprocessed
    """
    def __init__(self,
                 max_span_width: int,
                 window_size:int,
                 max_tokens_per_sentence: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_span_width = max_span_width
        self._window_size = window_size
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._max_tokens_per_sentence = max_tokens_per_sentence

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache

        def compute_sentence_starts(sentences):
            return np.cumsum([0] + [len(sent) for sent in sentences]).tolist()[:-1]
        

        file_path = cached_path(file_path)

        with open(file_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            # Loop over the documents.
            doc_text = json.loads(line)
            
            
            if 'train' in doc_text['_split'] :
                # TODO: i think the below if block is buggy, not working on docred
                if self._max_tokens_per_sentence != -1: # if == -1, don't split sentences
                    new_sentences = []
                    annotated_tokens = set()
                    for key in ['ner','events','relation','document_relations','clusters','event_clusters']:
                        if key == 'ner':
                            for sent_ner in doc_text[key]:
                                for mention_ner in sent_ner:
                                    for i in range(mention_ner[0], mention_ner[1]):
                                        annotated_tokens.add(i+1)
                                    
                        elif key == 'clusters':
                            for cluster in doc_text[key]:
                                for span in cluster:
                                    for i in range(span[0], span[1]):
                                        annotated_tokens.add(i+1)
                        # elif key == 'document_relations':
                            
                        #     for rel in doc_text[key]:
                        #         for i in range(rel[0], rel[1]):
                        #             annotated_tokens.add(i+1)
                        #         for i in range(rel[2], rel[3]):
                        #             annotated_tokens.add(i+1)
                        else:
                            pass
                            # raise NotImplementedError
                            
                    # print(annotated_tokens)
                    doc_tokens = [tok for sent in doc_text['sentences'] for tok in sent]
                    prev_split = 0
                    while len(doc_tokens) > 0:
                        
                        next_split = prev_split + self._max_tokens_per_sentence
                        # make sure it doesn't split on annotated tokens
                        while next_split in annotated_tokens:
                            next_split -= 1
                        new_sentences.append(doc_tokens[:next_split-prev_split])
                        doc_tokens = doc_tokens[next_split-prev_split:]
                        prev_split = next_split
                        # append dangling token to the last sentence
                        if len(doc_tokens) <= 1:
                            new_sentences[-1] += doc_tokens
                            break

                    doc_text['sentences'] = new_sentences

                doc_text['_sentence_starts'] = compute_sentence_starts(doc_text['sentences'])

                # adjust sentence-based annotation into corresponding sentences
                doc_text = self._adjust_annotation(doc_text)
                
                num_sentences = len(doc_text['sentences'])
                batch_start = 0

                for batch_start in range(0, num_sentences, self._window_size):
                    batch_end = min(batch_start+self._window_size-1, num_sentences-1)
                # sent_lengths = [len(sent) for sent in doc_text['sentences']]
                # while batch_start < num_sentences:
                #     batch_end = self._compute_batch_end(batch_start, sent_lengths)
                    # TODO (steeve): This is buggy!! should use sentence ends to compute chunk end.
                    chunk_start, chunk_end = doc_text['_sentence_starts'][batch_start], doc_text['_sentence_starts'][batch_end]
                    sentence_chunk = self.form_sentence_chunk(doc_text, chunk_start, chunk_end, batch_start, batch_end)
                    sentence_chunk = self.normalize_chunk(sentence_chunk, chunk_start, chunk_end)
                    instance = self.text_to_instance(sentence_chunk)
                    yield instance   
            else:
                # remove invalid spans in clusters
                chunk_start = 0
                chunk_end = len([tok for sent in doc_text['sentences'] for tok in sent])
                doc_text['clusters'], cluster_mapping = self._truncate_clusters(doc_text['clusters'], chunk_start, chunk_end)
                doc_text['document_relations'] = self._truncate_document_relations(doc_text['document_relations'], chunk_start, chunk_end, cluster_mapping)
                doc_text['_sentence_starts'] = compute_sentence_starts(doc_text['sentences'])
                instance = self.text_to_instance(doc_text)
                yield instance
    def _adjust_annotation(self, doc_text):
        '''
        Adjust sentence-based annotation based on re-split sentences.
        '''
        def _get_sentence_idx(span):
            for sentence_idx, (start, end) in enumerate(zip(sentence_starts, sentence_ends)):
                if span[0] >= start and span[1] <= end:
                    return sentence_idx
            raise ValueError(f"Span{span} sentence idx cannot be found. starts:{sentence_starts}, ends:{sentence_ends}.")

        sentence_starts = doc_text['_sentence_starts']
        doc_length = len([tok for sent in doc_text['sentences'] for tok in sent])
        sentence_ends = [x - 1 for x in sentence_starts[1:]] + [doc_length - 1]
        for key in ['ner','relation','events']:
            if key in doc_text:
                new_annotations = [[] for _ in range(len(sentence_starts))]
                annotations = [annotation for sent_annotation in doc_text[key] for annotation in sent_annotation]
                for annotation in annotations:
                    if key == 'ner':
                        try:
                            sentence_idx = _get_sentence_idx(annotation[:2])
                            new_annotations[sentence_idx].append(annotation)
                        except Exception as e:
                            print(e)
                            
                    elif key == 'relation':
                        raise NotImplementedError
                    elif key == 'events':
                        raise NotImplementedError
                doc_text[key] = new_annotations
        return doc_text

    def _compute_batch_end(self, batch_start, sentence_lengths):
        token_count = sentence_lengths[batch_start]
        batch_end = batch_start + 1
        while batch_end < len(sentence_lengths) and token_count + sentence_lengths[batch_end]:
            batch_end += 1
        return batch_end - 1

    def form_sentence_chunk(self, doc_text, chunk_start, chunk_end, batch_start, batch_end):
        sentence_chunk = deepcopy(doc_text)
        
        for key in ['events','relation','ner','clusters','event_clusters','document_events','document_relations','sentences', '_sentence_starts']:
            if key in doc_text:
                if key in {'sentences', '_sentence_starts','events','relation','ner'}:
                    sentence_chunk[key] = self._truncate_sentences(doc_text[key], batch_start, batch_end)
                elif key in {'clusters','event_clusters'}:
                    sentence_chunk[key], _cluster_mapping = self._truncate_clusters(doc_text[key], chunk_start, chunk_end)
                    if key == 'clusters':
                        cluster_mapping = _cluster_mapping
                elif key == 'document_relations':
                    sentence_chunk[key] = self._truncate_document_relations(doc_text[key], chunk_start, chunk_end, cluster_mapping)
                elif key == 'document_events':
                    sentence_chunk[key] = self._truncate_document_events(doc_text[key], chunk_start, chunk_end)
                else:
                    raise NotImplementedError
        return sentence_chunk

    def _within_range(self, span, chunk_start, chunk_end):
        return span[0] >= chunk_start and span[1] <= chunk_end

    def _truncate_clusters(self, clusters, chunk_start, chunk_end):
        '''
        Given a document, gather only the instance that falls in chunk_start, chunk_end
        '''
        new_clusters = []
        cluster_mapping = {} # map from original cluster idx to new cluster idx
        for original_cluster_idx, cluster in enumerate(clusters):
            new_cluster = [[span[0], span[1]] for span in cluster if self._within_range(span, chunk_start, chunk_end) and not self._too_long(span)]
            if len(new_cluster) >= 1:
                cluster_mapping[original_cluster_idx] = len(new_clusters)
                new_clusters.append(new_cluster)
        return new_clusters, cluster_mapping

    def _truncate_document_events(self, document_events, chunk_start, chunk_end):
        '''
        Given a document, gather only the document event instance whose trigger is within the chunk range and filter out out-side-the-range arguments.
        '''
        new_document_events = []
        for document_event in document_events:
            trig = document_event[0]
            args = document_event[1:]
            if self._within_range((trig[0], trig[0]), chunk_start, chunk_end):

                new_document_event = [trig]
                for arg in args:
                    if self._within_range((arg[0], arg[1]), chunk_start, chunk_end):
                        new_document_event.append(arg)
                if len(new_document_event) == 1:
                    print("Warning! document event only contains trigger!")
                new_document_events.append(new_document_event)
        return new_document_events

    def _truncate_document_relations(self, document_relations, chunk_start, chunk_end, cluster_mapping):
        '''
        Given a document, gather only the document relation instance where both span are within chunk_start.
        '''
        new_document_relations = []
        for document_relation in document_relations:
            # span1, span2 = (document_relation[:2], document_relation[2:4])
            entity_idx1, entity_idx2, relation_label = document_relation

            if entity_idx1 in cluster_mapping and entity_idx2 in cluster_mapping:
                new_document_relation = [cluster_mapping[entity_idx1], cluster_mapping[entity_idx2], relation_label]
                new_document_relations.append(new_document_relation)
        return new_document_relations
    
    def _truncate_sentences(self, sentence_annotations, batch_start, batch_end):
        return [sentence_annotations[i] for i in range(batch_start, batch_end+1)]

    def normalize_chunk(self, sentence_chunk, chunk_start, chunk_end):
        for key in ['_sentence_starts','events','relation', 'ner', 'document_events','document_relations','clusters','event_clusters']:
            if key in sentence_chunk:
                sentence_chunk[key] = self._normalize(sentence_chunk, key, chunk_start, chunk_end)
        return sentence_chunk

    def _normalize(self, sentence_chunk, key, chunk_start, chunk_end):
        '''
        decrease each offset by chunk_start
        ''' 

        if key == '_sentence_starts':
            sentence_chunk[key] = [ss - chunk_start for ss in sentence_chunk[key]]
        elif key == 'events':
            for sent_events in sentence_chunk[key]:
                for event in sent_events:
                    event[0][0] -= chunk_start
                    for arg in event[1:]:
                        arg[0] -= chunk_start
                        arg[1] -= chunk_start
        elif key == 'ner':
            for sent_ner in sentence_chunk[key]:
                for ner in sent_ner:
                    ner[0] -= chunk_start
                    ner[1] -= chunk_start
        elif key == 'document_events':
            for document_event in sentence_chunk[key]:
                document_event[0][0] -= chunk_start
                for arg in document_event[1:]:
                    arg[0] -= chunk_start
                    arg[1] -= chunk_start
        elif key == 'document_relations':
            return sentence_chunk[key]
        #     for document_relation in sentence_chunk[key]:
        #         for i in range(4):
        #             document_relation[i] -= chunk_start
        elif key in {'clusters', 'event_clusters'}:
            for cluster in sentence_chunk[key]:
                for span in cluster:
                    span[0] -= chunk_start
                    span[1] -= chunk_start
        else:
            raise NotImplementedError(f"{key} not implemented.")
        return sentence_chunk[key]
        
    def _too_long(self, span):
        return span[1] - span[0] + 1 > self._max_span_width
    def _invalid(self, span):
        return span[0] > span[1]

    def _process_ner(self, span_tuples, sent):
        ner_labels = [""] * len(span_tuples)
        
        for span, label in sent.ner_dict.items():
            if self._too_long(span) or self._invalid(span):
                continue
            
            ix = span_tuples.index(span)
            ner_labels[ix] = label

        return ner_labels

    def _process_coref(self, span_tuples, sent):
        coref_labels = [-1] * len(span_tuples)

        for span, label in sent.cluster_dict.items():
            if self._too_long(span) or self._invalid(span):
                print(f"Span {span} too long or invalid.")
                continue
            ix = span_tuples.index(span)
            coref_labels[ix] = label
        return coref_labels

    def _process_relations(self, span_tuples, sent):
        relations = []
        relation_indices = []

        # Loop over the gold spans. Look up their indices in the list of span tuples and store
        # values.
        for (span1, span2), label in sent.relation_dict.items():
            # If either span is beyond the max span width, skip it.
            if self._too_long(span1) or self._too_long(span2) or self._invalid(span1) or self._invalid(span2):
                continue
            ix1 = span_tuples.index(span1)
            ix2 = span_tuples.index(span2)
            relation_indices.append((ix1, ix2))
            relations.append(label)

        return relations, relation_indices

    def _process_events(self, span_tuples, sent):
        n_tokens = len(sent.text)

        trigger_labels = [""] * n_tokens
        for tok_ix, trig_label in sent.events.trigger_dict.items():
            trigger_labels[tok_ix] = trig_label

        arguments = []
        argument_indices = []

        for (trig_ix, arg_span), arg_label in sent.events.argument_dict.items():
            if self._too_long(arg_span) or self._invalid(arg_span):
                continue
            arg_span_ix = span_tuples.index(arg_span)
            argument_indices.append((trig_ix, arg_span_ix))
            arguments.append(arg_label)

        return trigger_labels, arguments, argument_indices

    def _process_document_relations(self, span_tuples, doc):
        relations = []
        relation_indices = []

        
        for (ix1, ix2), label in doc.document_relation_dict.items():
            # If either span is beyond the max span width, skip it.
            
            try:
                # ix1 = span_tuples.index(span1)
                # ix2 = span_tuples.index(span2)
                relation_indices.append((ix1, ix2))
                relations.append(label)
            except:
                print(doc.sentences)
                print(span_tuples)
                print(doc.document_relation_dict)
                print(doc.doc_key)
                # print(span1, span2)
                exit()

        return relations, relation_indices
    
    def _process_sentence(self, sent: Sentence, dataset: str):
        # Get the sentence text and define the `text_field`.
        sentence_text = [self._normalize_word(word) for word in sent.text]
        text_field = TextField([Token(word) for word in sentence_text], self._token_indexers)

        # Enumerate spans.
        spans = []
        for start, end in enumerate_spans(sentence_text, max_span_width=self._max_span_width):
            spans.append(SpanField(start, end, text_field))
        span_field = ListField(spans)
        span_tuples = [(span.span_start, span.span_end) for span in spans]

        # Convert data to fields.
        # NOTE: The `ner_labels` and `coref_labels` would ideally have type
        # `ListField[SequenceLabelField]`, where the sequence labels are over the `SpanField` of
        # `spans`. But calling `as_tensor_dict()` fails on this specific data type. Matt G
        # recognized that this is an AllenNLP API issue and suggested that represent these as
        # `ListField[ListField[LabelField]]` instead.
        fields = {}
        fields["text"] = text_field
        fields["spans"] = span_field
        if sent.ner is not None:
            ner_labels = self._process_ner(span_tuples, sent)
            fields["ner_labels"] = ListField(
                [LabelField(entry, label_namespace=f"{dataset}__ner_labels")
                 for entry in ner_labels])
        if sent.cluster_dict is not None:
            # Skip indexing for coref labels, which are ints.
            coref_labels = self._process_coref(span_tuples, sent)
            fields["coref_labels"] = ListField(
                [LabelField(entry, label_namespace="coref_labels", skip_indexing=True)
                 for entry in coref_labels])
        if sent.relations is not None:
            relation_labels, relation_indices = self._process_relations(span_tuples, sent)
            fields["relation_labels"] = AdjacencyField(
                indices=relation_indices, sequence_field=span_field, labels=relation_labels,
                label_namespace=f"{dataset}__relation_labels")
        if sent.events is not None:
            trigger_labels, argument_labels, argument_indices = self._process_events(span_tuples, sent)
            fields["trigger_labels"] = SequenceLabelField(
                trigger_labels, text_field, label_namespace=f"{dataset}__trigger_labels")
            fields["argument_labels"] = AdjacencyFieldAssym(
                indices=argument_indices, row_field=text_field, col_field=span_field,
                labels=argument_labels, label_namespace=f"{dataset}__argument_labels")

        return fields

    def _process_sentence_fields(self, doc: Document):
        # Process each sentence.
        sentence_fields = [self._process_sentence(sent, doc.dataset) for sent in doc.sentences]

        # Make sure that all sentences have the same set of keys.
        first_keys = set(sentence_fields[0].keys())
        for entry in sentence_fields:
            if set(entry.keys()) != first_keys:
                raise DyGIEDataException(
                    f"Keys do not match across sentences for document {doc.doc_key}.")

        # For each field, store the data from all sentences together in a ListField.
        fields = {}
        keys = sentence_fields[0].keys()
        for key in keys:
            this_field = ListField([sent[key] for sent in sentence_fields])
            fields[key] = this_field

        return fields

    def _process_doc_fields(self, doc, fields):
        # Flatten the list of sentences into a document of tokens. This maybe helpful for using longformers.
        document_text = [self._normalize_word(word) for sent in doc.sentences for word in sent.text]
        text_field = TextField([Token(word) for word in document_text], self._token_indexers)
        dataset= doc.dataset
        sentence_starts = doc.sentence_starts
        # Enumerate spans. Make span_tuples have offset w.r.t. doc.
        spans = []
        for sent, sentence_start in zip(doc.sentences, sentence_starts):
        
            sentence_text = [self._normalize_word(word) for word in sent.text]
            for start, end in enumerate_spans(sentence_text, max_span_width=self._max_span_width):
                spans.append(SpanField(start + sentence_start, end + sentence_start, text_field))
        span_field = ListField(spans)
        span_tuples = [(span.span_start, span.span_end) for span in spans]
        
        fields['doc_text'] = ListField([text_field])
        
        if doc.document_events is not None:
            # build event fields
            document_trigger_labels, document_argument_labels, argument_indices = self._process_document_events(span_tuples, doc)
            fields["document_trigger_labels"] = SequenceLabelField(
                document_trigger_labels, text_field, label_namespace=f"{dataset}__document_trigger_labels")
            fields["document_argument_labels"] = AdjacencyFieldAssym(
                indices=argument_indices, row_field=text_field, col_field=span_field,
                labels=document_argument_labels, label_namespace=f"{dataset}__document_argument_labels")

        if doc.document_relations is not None:

            # build clusters
            # a work-around for AdjacencyField
            cluster_field = [IndexField(cluster_idx, sequence_field=span_field)
                 for cluster_idx in range(len(doc.cluster_list)+1)]
            
            cluster_field = ListField(cluster_field)

            

            # build document relations fileds
            document_relation_labels, relation_indices = self._process_document_relations(span_tuples, doc)
            fields["document_relation_labels"] = AdjacencyField(
                indices=relation_indices, sequence_field=cluster_field, labels=document_relation_labels,
                label_namespace=f"{dataset}__document_relation_labels")
        return fields
    @overrides
    def text_to_instance(self, doc_text: Dict[str, Any]):
        """
        Convert a Document object into an instance.
        """
        doc = Document.from_json(doc_text)

        # Make sure there are no single-token sentences; these break things.
        sent_lengths = [len(x) for x in doc.sentences]
        if min(sent_lengths) < 2:
            msg = (f"Document {doc.doc_key} has a sentence with a single token or no tokens. "
                   "This may break the modeling code.")
            warnings.warn(msg)

        fields = self._process_sentence_fields(doc)

        fields = self._process_doc_fields(doc, fields)

        
        fields["metadata"] = MetadataField(doc)
        
        return Instance(fields)

    @overrides
    def _instances_from_cache_file(self, cache_filename):
        with open(cache_filename, "rb") as f:
            for entry in pkl.load(f):
                yield entry

    @overrides
    def _instances_to_cache_file(self, cache_filename, instances):
        with open(cache_filename, "wb") as f:
            pkl.dump(instances, f, protocol=pkl.HIGHEST_PROTOCOL)

    @staticmethod
    def _normalize_word(word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word
