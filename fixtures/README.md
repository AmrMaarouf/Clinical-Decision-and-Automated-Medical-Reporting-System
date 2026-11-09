# fixtures

Fixed example files that every service develops and tests against.
All of them describe the same synthetic case (`case_id = demo001`).

| file | what it is |
|---|---|
| `v1/transcript.json` | **Synthetic** short doctor/patient dialogue written by hand. NOT a real recording, NOT real patient data. |
| `v1/extraction.json` | Expected extraction for that dialogue (hand-written gold). |
| `v1/labs.json` | **Synthetic** CBC result (values and printed ranges are made up). `source_file` points to an image that does not exist yet. |
| `v1/xray.json` | **Placeholder** model output (fake model, fake probabilities). No real image. |
| `v1/report.json` | Example draft report built from the four files above. |

Rules: never commit real patient data. Anything derived from a real source must be
de-identified first, and its source must be written in this file.
