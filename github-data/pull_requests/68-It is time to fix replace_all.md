### 🐛 [#68](https://github.com/ikawrakow/ik_llama.cpp/pull/68) - It is time to fix replace_all

| **Author** | `ikawrakow` |
| :--- | :--- |
| **State** | ❌ **Closed** |
| **Created** | 2024-09-28 |
| **Updated** | 2024-09-28 |

---

#### Description

I have been annoyed by having to wait for close to 2 seconds for the perplexity calculation to start because that's how long tokenization took when using Phi-3.5-mini (not to mention the close to 20 seconds wait when running an imatrix calculation with `wiki.train.raw`). Today my patience got exhausted and I decided to investigate. Turns out I inherited this gem when I last synced with mainline `llama.cpp` (in `src/llama-impl.h`):
```
static void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return; // Avoid infinite loop if 'search' is an empty string
    }   
    size_t pos = 0;
    while ((pos = s.find(search, pos)) != std::string::npos) {
        s.replace(pos, search.length(), replace);
        pos += replace.length();
    }   
}
```
This innocently looking function takes 1.4 seconds to replace spaces in `wiki.test.raw` with whatever Phi-3.5-mini needs. Fittingly, it has been added in a PR titled `llama: better replace_all`.

Initially I implemented my own version that reduces the time from 1.4 seconds to 4 ms. But then I noticed that since my last sync Justine Tunney has fixed this gem in mainline `llama.cpp`, so at the end preferred to copy/paste her version to not unnecessarily diverge from mainline.