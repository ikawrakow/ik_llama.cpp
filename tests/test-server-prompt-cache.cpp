#include "server-task.h"

#include <cassert>

int main() {
    server_prompt prompt;

    prompt.pos_min = 100;
    assert(!prompt.has_rewind_checkpoint(64));

    server_prompt_checkpoint early;
    early.pos_min = 0;
    early.pos_max = 48;
    early.pos_min_prompt = 0;
    early.pos_max_prompt = 48;
    prompt.checkpoints.push_back(early);
    assert(prompt.has_rewind_checkpoint(64));

    prompt.checkpoints.clear();
    server_prompt_checkpoint prompt_aligned;
    prompt_aligned.pos_min = 120;
    prompt_aligned.pos_max = 140;
    prompt_aligned.pos_min_prompt = 40;
    prompt_aligned.pos_max_prompt = 64;
    prompt.checkpoints.push_back(prompt_aligned);
    assert(!prompt.has_rewind_checkpoint(64));

    prompt.checkpoints.clear();
    prompt.pos_min = 32;
    assert(prompt.has_rewind_checkpoint(64));

    return 0;
}
