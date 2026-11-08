# fixtures

Fixed example files that every service develops and tests against.

| file | what it is |
|---|---|
| `v1/transcript.json` | **Synthetic** short doctor/patient dialogue written by hand. It is NOT a real recording and NOT real patient data. |
| `v1/extraction.json` | The expected extraction for that dialogue (hand-written gold). |

Rules: never commit real patient data. Anything derived from a real source must be
de-identified first, and its source must be written in this file.
