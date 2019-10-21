`base*` are the global soft attention models.
`retrain1` or `retrain2` in the end means that it imports an earlier model and retrains.

`hard-att-local*` are the local windowed soft attention models.

`hard-segm-prob*` are the segmental soft attention models.

`hard-att*` (but not `local`) are the hard attention models.

Anything with `imp` in the first part means that it imports the global soft attention baseline.
`imp*-retrain1` means that it imports the `...retrain1` global soft attention baseline.
Without `imp` means that it trains from scratch.

Anything with `-recog` in the first parts means that this was only used for recognition.
E.g. it varies some search related parameter such as beam size.
