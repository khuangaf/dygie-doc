# Data

We provide details on the data preprocessing for each of the datasets available here.



## Table of contents

- [Mention-level representation](#mention-level-representation)
- [Entity-level representation](#entity-level-representation)

## Mention-level representation

The format of mention-level data is almost the same as DyGIE++'s format, which follows the representation of [SciERC](http://nlp.cs.washington.edu/sciIE/). Please follow the original [DyGIE++ repo](https://github.com/dwadden/dygiepp) for more details. In addition to the `ner` and `clusters` fields defined in DyGIE++, we create a new field, `document_relations`, which allows for representing cross-sentence relations. `document_relations` stores mention-level relations. For datasets that annotate relations at the entity level. We permute all possible combinations of mentions from both entities to produce mention-level relations. For instance, an entity-level relation where the first entity contains 3 mentions, and the second entity contains 2 mentions would result in 3*2=6 pairs of mention-level relations. An example of the `document_relations` is as follows
```
[
  [34, 37, 12, 12, 'P571'],
  [1, 4, 12, 12, 'P571'],
  [54, 54, 12, 12, 'P571'],
  [104, 104, 12, 12, 'P571'],
  [201, 201, 12, 12, 'P571'],
]
```
Compared to the original sentnece-level `relation` field, we `document_relations` remove the notion of "sentence". Hence, the representation of `document_relations` is a two-layered nested list instead of three-layered as in the `relation` field.


## Entity-level representation
The entity-level format is used for evaluation. The original relations of the three datasets used in this project: CDR, GDA, and DocRED are all annotated in entity-level. The fields are as follows:

* `doc_key`: The ID of the document. Same as mention-level `doc_key`.
* `sentences`: Input doucment represented as a nested list of tokens. Same as mention-level `doc_key`.
* `clusters`: A dictionary with the entity ID being the key, and spans of entity mentions being the value. For instance,

```
{
  '3553': [[2, 4], [26, 31], [192, 194], [213, 215], [253, 255]],
  'D009103': [[16, 16], [54, 54], [61, 61], [79, 79], [95, 95], [152, 152], [202, 202]],
  '3557': [[33, 37], [39, 41], [188, 190], [206, 208], [249, 251]]
}
```

* `relations`: A list of relations. Each relation contains the entity ID for the first and second entity as well as their relation. For instance,

```
[
  ['D009103', '3553', '1:GDA:2'], 
  ['D009103', '3557', '1:GDA:2']
]
```