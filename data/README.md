# ADD 2026 Track 2 Training Data

`train/` and `dev/` contain 8,600 and 2,000 WAV files.
`labels/train_labels.jsonl` and `labels/dev_labels.jsonl` contain `file_id`,
`label` (`real` or `fake`), and `explanation`.
`file_id` matches the WAV filename without `.wav` in the same split.

`explanation` contains an XML reference answer: `<Evidence>` with
`<Spectral>`, `<Prosody>`, `<Pronunciation>`, `<Noise>`, and `<OtherAnomalies>`,
followed by `<Decision>`.
Submission format and scoring follow the official challenge website.
