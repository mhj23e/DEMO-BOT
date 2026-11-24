# DEMO-BOT test plan

Verify source chunking, cosine-similarity ranking, and recovery from
file and URL ingestion failures.

| Case | Exercise | Pass criteria |
| --- | --- | --- |
| DEMO-01 | Chunk text with irregular whitespace and a specified overlap | Whitespace is normalized and the expected chunks retain their source |
| DEMO-02 | Chunk blank input | No chunks are created |
| DEMO-03 | Query a FAISS index with controlled embeddings of different magnitudes | The closest source appears first in context and citations |
| DEMO-04 | Index a corrupt PDF alongside a valid TXT file | The TXT stays available and one warning identifies the PDF |
| DEMO-05 | Index one successful web page and one HTTP 404 | Valid text is indexed and one warning identifies the failed URL and status |
| DEMO-06 | Query without any sources | Retrieval returns the empty-index message and skips embedding model initialization |

Run `python -m pip install -r requirements-test.txt` and
`python -m pytest -q`. The same suite runs on every GitHub push and pull
request. Pass when all cases pass. No live URLs or model downloads are
required. Live generation and Streamlit rendering are outside this suite.

Report defects in GitHub Issues with the revision, environment, case ID,
steps, expected behavior, and actual behavior. Attach synthetic fixtures.
