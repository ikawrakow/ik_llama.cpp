
from collections import defaultdict

import requests

MAX_CODEPOINTS = 0x110000

SCRIPT_DATA_URL = "https://www.unicode.org/Public/UCD/latest/ucd/Scripts.txt"


res = requests.get(SCRIPT_DATA_URL)
res.raise_for_status()
data = res.content.decode()

cptL_cptU_script = []
for line in data.splitlines():
    line = line.split()
    if len(line) <= 1 or line[0] == "#":
        continue

    cpt = line[0].split("..")
    if len(cpt) == 1:
        cpt += cpt
    cpt_lower, cpt_upper = cpt

    cpt_lower = int(cpt_lower, 16)
    if cpt_lower >= MAX_CODEPOINTS:
        break

    cpt_upper = int(cpt_upper, 16)
    if cpt_upper >= MAX_CODEPOINTS:
        break

    assert line[1] == ";"

    script = line[2].lower()

    assert line[3] == "#"

    # categ = line[4]
    # assert len(categ) == 2

    cptL_cptU_script.append([cpt_lower, cpt_upper, script])

cptL_cptU_script.sort(key=lambda x: x[0])  # just in case

# merge neighboring codepoints that belong to same script
im = 0  # merge index
for cpt_lower, cpt_upper, script in cptL_cptU_script[1:]:
    if (cptL_cptU_script[im][2] == script) and (cptL_cptU_script[im][1] + 1 == cpt_lower):
        cptL_cptU_script[im][1] = cpt_upper
    else:
        im += 1
        cptL_cptU_script[im] = [cpt_lower, cpt_upper, script]
del cptL_cptU_script[im + 1:]

def out(line=""):
    print(line, end='\n')  # noqa

# Generate 'unicode-script-data.cpp':
#   python scripts/gen-unicode-script-data.py > src/unicode-script-data.cpp

out("""\
// generated with scripts/gen-unicode-script-data.py

#include "unicode.h"
#include "unicode-data.h"
""")

out("""\
size_t unicode_fill_from_utf8(std::string* utf8, std::vector<uint32_t>* dst_cpts, std::vector<std::string>* dst_scripts) {
    if (utf8 == nullptr) {
        return 0;
    }
""")

out("static const std::vector<std::string> unicode_scripts = {")
for _, _, script in cptL_cptU_script:
    out("    \"%s\"," % script)
out("};")

out("static const std::vector<uint32_t> unicode_script_lasts = {")
for _, cpt_upper, _ in cptL_cptU_script:
    out("    0x%06X," % cpt_upper)
out("};")

out("""\
    const auto cpts = unicode_cpts_from_utf8(*utf8);
    const size_t n_cpt = cpts.size();

    std::vector<std::string> scripts;
    scripts.reserve(n_cpt);

    for (const auto& cpt: cpts) {
        const auto it = std::lower_bound(unicode_script_lasts.begin(), unicode_script_lasts.end(), cpt);
        if (it != unicode_script_lasts.end()) {
            scripts.push_back(unicode_scripts[std::distance(unicode_script_lasts.begin(), it)]);
        }
    }

    if (dst_cpts != nullptr) {
        *dst_cpts = cpts;
    }
    if (dst_scripts != nullptr) {
        *dst_scripts = scripts;
    }

    return n_cpt;
}
""")
