# Wrist-EMG handwriting decoding: evaluation notes

Rohit Ravi. Summary accompanying /projects/handwriting-emg.
This is an aggregate evaluation summary, not a source-code release or complete reproduction package.

## Task and split

Offline character decoding from Meta's released surface-EMG handwriting dataset and checkpoint. Ten released test users were held out from generic-model training. Supervised personal adaptation uses recording folds from each user, evaluated on separate recordings. Hyperparameters were selected on validation users and then fixed for the test users. This is not zero-shot personalization.

## Adaptation and decoding

Full personal fine-tuning used learning rate 1e-4 for 500 steps. This was the selected recipe at the edge of the tested grid, not a claim of global optimality.

The language model is a 7-gram Kneser–Ney character model trained on 30 million WikiText-103 characters plus deduplicated training prompts. Paragraphs containing test-prompt sequences of at least three words were removed. CTC prefix beam search used beam width 32, alpha 0.6 and beta 2.0, selected using five validation users and fixed before test evaluation.

## Aggregate character error rate (%)

| Encoder | Greedy | With language model |
|---|---:|---:|
| Generic, reproduced | 30.07 | 25.43 |
| Early input correction | 26.99 | 21.87 |
| Early adapter | 26.76 | 21.89 |
| Personal full fine-tune | 20.33 | 15.02 |

Fine-tuning on the other nine users instead gave 39.74% CER; none of the ten users improved. Personal calibration and language modeling therefore should not be presented as interchangeable interventions.

## Limits

Ten users, offline CER, supervised recording-specific adaptation, and the released checkpoint. No online words-per-minute result or clinical benefit is established. Numbers are rounded independently, so subtracting displayed endpoints may differ from the recorded unrounded change. Dataset and upstream code use their own license terms (CC BY-NC 4.0); this summary does not grant additional rights to them.

Upstream: https://github.com/facebookresearch/generic-neuromotor-interface
