### [Pull Request #653](https://github.com/ikawrakow/ik_llama.cpp/pull/653) - Add GitHub data: backup and convertion scripts + backup update

| **Author** | `ThomasBaruzier` |
| :--- | :--- |
| **State** | ✅ **Open** |
| **Created** | 2025-07-26 |
| **Updated** | 2025-07-26 |

---

#### Description

Hello!  

I've refined the scraping and conversion scripts. While they should work with any repository, I haven't extensively tested them beyond the current use case. For this repository, the scripts consistently complete in ~30 seconds (initially 750s!) using just 11 API requests to fetch all issues, pull requests, and discussions.  

I initially explored resumable/incremental scraping but abandoned the idea due to reliability issues: the `updatedAt` field only reflects edits to the issue/PR body, not new activity. Instead, I focused on optimization, achieving the results below.  

---

### Usage  
```bash
GITHUB_TOKEN='github_pat_...' python ghscrape.py ikawrakow/ik_llama.cpp -o ik.json && python ghconvert.py ik.json -o .
```

---

### Scraping Demo  
```
GITHUB_TOKEN='github_pat_...' python ghscrape.py ikawrakow/ik_llama.cpp -o ik.json
INFO: Fetching all issues...
INFO: API Rate Limit (Req #1): 4997 points remaining, resets in 59m 54s.
INFO: Processed 100 issues...
INFO: API Rate Limit (Req #2): 4994 points remaining, resets in 59m 52s.
INFO: Processed 129 issues...
INFO: Fetching all nested data for 129 items (1 pages)...
INFO: API Rate Limit (Req #3): 4993 points remaining, resets in 59m 52s.
INFO: Finished issues: Found and processed 129 items.
INFO: Fetching all pull requests...
INFO: API Rate Limit (Req #4): 4888 points remaining, resets in 59m 50s.
INFO: Processed 100 pull_requests...
INFO: API Rate Limit (Req #5): 4783 points remaining, resets in 59m 47s.
INFO: Processed 200 pull_requests...
INFO: API Rate Limit (Req #6): 4678 points remaining, resets in 59m 42s.
INFO: Processed 300 pull_requests...
INFO: API Rate Limit (Req #7): 4573 points remaining, resets in 59m 37s.
INFO: Processed 400 pull_requests...
INFO: API Rate Limit (Req #8): 4468 points remaining, resets in 59m 34s.
INFO: Processed 450 pull_requests...
INFO: Fetching all nested data for 450 items (32 pages)...
INFO: API Rate Limit (Req #9): 4467 points remaining, resets in 59m 34s.
INFO: Finished pull_requests: Found and processed 450 items.
INFO: Fetching all discussions...
INFO: API Rate Limit (Req #10): 4365 points remaining, resets in 59m 31s.
INFO: Processed 69 discussions...
INFO: Fetching all nested data for 69 items (123 pages)...
INFO: API Rate Limit (Req #11): 4364 points remaining, resets in 59m 29s.
INFO: Finished discussions: Found and processed 69 items.
INFO: Data successfully saved to ik.json
INFO: Total execution time: 30.70 seconds
```

### Conversion Demo  
```
python ghconvert.py ik.json -o .
Processing 129 issues...
Processing 450 pull_requests...
Processing 69 discussions...
Successfully generated 648 Markdown files.
Files are in the '.' directory.
```

---

### Notes  
- Content extraction for reviews isn’t fully implemented yet (see [example](https://github.com/ThomasBaruzier/ik_llama.cpp/blob/tb/github-data-scripts/github-data/pull_requests/620%20-%20Bump%20Windows%20max%20open%20files%20from%20512%20to%202048.md)). This could be added later if needed.
- Wiki backups are not implemented.
- Filename sanitization has been made a bit more lenient, max 100 char, and allowing a broader set of characters that should still be compatible with all OSes.
- Putting this PR as draft until I verify the above claim on a Windows machine.

- [x] I’ve read the [contributing guidelines](https://github.com/ggerganov/llama.cpp/blob/master/CONTRIBUTING.md).  
- **Self-reported review complexity**:  
  - [ ] Low
  - [x] Medium
  - [ ] High