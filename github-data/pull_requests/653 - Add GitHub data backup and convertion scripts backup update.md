## 🔀 [Pull Request #653](https://github.com/ikawrakow/ik_llama.cpp/pull/653) - Add GitHub data: backup and convertion scripts + backup update

| **Author** | `ThomasBaruzier` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Source Branch** | `tb/github-data-scripts` |
| **Target Branch** | `main` |
| **Created** | 2025-07-26 |
| **Updated** | 2025-07-28 |

---

## 📄 Description

Hello!  

I've refined the scraping and conversion scripts. While they should work with any repository, I haven't extensively tested them beyond the current use case. For this repository, the scripts consistently complete in ~30 seconds (initially 750s!) using just ~~11~~ 10 API requests to fetch all issues, pull requests, and discussions.  

I initially explored resumable/incremental scraping but abandoned the idea due to reliability issues: the `updatedAt` field only reflects edits to the issue/PR body, not new activity. Instead, I focused on optimization, achieving the results below.  

---

### Usage  
```bash
export GITHUB_TOKEN='github_pat_...'
cd github-data
rm -rf issues discussions pull_requests index.md ik.json
python ghscrape.py ikawrakow/ik_llama.cpp -o ik.json
python ghconvert.py ik.json -o .
```

#### Or as a one-liner (ensure that you are in `github-data/`):
```bash
rm -rf issues discussions pull_requests index.md ik.json && GITHUB_TOKEN='github_pat_...' python ghscrape.py ikawrakow/ik_llama.cpp -o ik.json && python ghconvert.py ik.json -o .
```

---

### Scraping Demo

```
python ghscrape.py ikawrakow/ik_llama.cpp -o ik.json
```
```
INFO: Fetching all issues...
INFO: API Rate Limit (Req #1): 4997 points remaining, resets in 59m 54s.
INFO: Processed 100 issues...
INFO: API Rate Limit (Req #2): 4994 points remaining, resets in 59m 52s.
INFO: Processed 131 issues...
INFO: Fetching all nested data for 131 items (1 pages)...
INFO: API Rate Limit (Req #3): 4993 points remaining, resets in 59m 52s.
INFO: Processed batch of 1 pages. 0 pages remaining.
INFO: Structuring final items for issues...
INFO: Finished issues: Found and processed 131 items.
INFO: Fetching all pull requests...
INFO: API Rate Limit (Req #4): 4888 points remaining, resets in 59m 49s.
INFO: Processed 100 pull_requests...
INFO: API Rate Limit (Req #5): 4783 points remaining, resets in 59m 46s.
INFO: Processed 200 pull_requests...
INFO: API Rate Limit (Req #6): 4678 points remaining, resets in 59m 41s.
INFO: Processed 300 pull_requests...
INFO: API Rate Limit (Req #7): 4573 points remaining, resets in 59m 36s.
INFO: Processed 400 pull_requests...
INFO: API Rate Limit (Req #8): 4468 points remaining, resets in 59m 34s.
INFO: Processed 452 pull_requests...
INFO: Fetching all nested data for 452 items (0 pages)...
INFO: Structuring final items for pull_requests...
INFO: Finished pull_requests: Found and processed 452 items.
INFO: Fetching all discussions...
INFO: API Rate Limit (Req #9): 4366 points remaining, resets in 59m 30s.
INFO: Processed 71 discussions...
INFO: Fetching all nested data for 71 items (1 pages)...
INFO: API Rate Limit (Req #10): 4365 points remaining, resets in 59m 29s.
INFO: Processed batch of 1 pages. 0 pages remaining.
INFO: Structuring final items for discussions...
INFO: Finished discussions: Found and processed 71 items.
INFO: Data successfully saved to ik.json
INFO: Total execution time: 30.55 seconds
```

### Conversion Demo

```
python ghconvert.py ik.json -o .
```
```
Processing 131 issues...
Processing 452 pull_requests...
Processing 71 discussions...
Generating index.md summary file...
Successfully generated 654 Markdown files.
Files are in the '.' directory.
```

---

### Relevant links jump:

Scripts:
- https://github.com/ThomasBaruzier/ik_llama.cpp/blob/tb/github-data-scripts/github-data/ghscrape.py
- https://github.com/ThomasBaruzier/ik_llama.cpp/blob/tb/github-data-scripts/github-data/ghconvert.py

Index:
- https://github.com/ThomasBaruzier/ik_llama.cpp/blob/tb/github-data-scripts/github-data/index.md

Discussion example:
- https://github.com/ThomasBaruzier/ik_llama.cpp/blob/tb/github-data-scripts/github-data/discussions/477%20-%20DeepSeek-R1-0528%20ik%20quants.md

PR example:
- https://github.com/ThomasBaruzier/ik_llama.cpp/blob/tb/github-data-scripts/github-data/pull_requests/620%20-%20Bump%20Windows%20max%20open%20files%20from%20512%20to%202048.md

Issue example:
- https://github.com/ThomasBaruzier/ik_llama.cpp/blob/tb/github-data-scripts/github-data/issues/296%20-%20Possible%20numerical%20stability%20issue%20with%20experimental%20quant%20of%20DeepSeek-V3-0324.md

---

### Notes  
- ~~Content extraction for reviews isn’t fully implemented yet (see [example](https://github.com/ThomasBaruzier/ik_llama.cpp/blob/tb/github-data-scripts/github-data/pull_requests/620%20-%20Bump%20Windows%20max%20open%20files%20from%20512%20to%202048.md)). This could be added later if needed.~~ Fixed.
- Wiki backups are not implemented.

- [x] I’ve read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md).  
- **Self-reported review complexity**:  
  - [ ] Low
  - [x] Medium
  - [ ] High