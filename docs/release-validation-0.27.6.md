# LocalHarness 0.27.6 validation

Gate build: the three fixes on top of 0.27.5 (the `/review` picker, plain level and reply words, the answer-only reply for an ask that names a form). Model under test: `qwen-27b-swift-1.5` on llama.cpp.

## Unit tests

The feature-relevant tests are `tests/unit/test_picker.py` (the description line follows the highlight), `tests/unit/test_review_dial.py` (the picker's row and notes, the level phrases), `tests/unit/test_terminal.py` and `tests/unit/channels/test_mobile_push.py` (the revised line and title), and `tests/unit/test_critic_scorers.py` (the compose request with and without a named form). The full unit suite passed on the gate build except `test_memory_index_tools.py::test_resolve_yesterday_window`, which assumes a 24-hour yesterday and fails on the day after daylight saving ends.

## Live checks

`/review` in a tmux session drew:

```
 off  1  2  3  max
 1 check, each followed by a fix; the last fix goes out not re-checked (default)
```

before the wording fix, the second line moving with the highlight; after it the line reads `check, then fix (default)`.

`in three sentences, what is this project and what do its two functions do?` (the `code-question` scenario), on 0.27.5 and on each fix:

| Build | Reply |
| --- | --- |
| 0.27.5 | 5-6 sentences, opening with a paragraph on how the agent worked |
| compose instruction only | 3 sentences, the first of them about how it worked, speaking of "the person" |
| the form gate | 2 sentences of answer only, in 21 s, `◆ reply pass` |
