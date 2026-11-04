# LocalHarness 0.27.7 validation

Gate build: the release commit on top of 0.27.6 (`save_to` and the asks after a saved page, `browser_fetch`
scrolling, skill review and skill loading, and the two fixes), tested from a clean checkout. Model under
test: `qwen-27b-swift-1.5` on llama.cpp.

## Unit tests

The feature-relevant tests are `tests/unit/test_web_fetch_save_to.py`, `test_downloads_gate.py`,
`test_saved_pages_gate.py` and `test_saved_pages_composed.py` (saving a page and the asks that follow),
`test_browser_plugin.py`, `test_browser_interactive.py` and `test_tool_parameters_reach_execute.py` (the
scroll, `save_to` on `browser_fetch`, and every advertised parameter reaching its tool), and
`test_reply_panel.py`, `test_skills_tool.py`, `test_skills_listing.py`, `test_skills_slash.py`,
`test_skills_misc.py`, `test_skills_discovery.py` and `test_subagent_children.py` (a skill's review and
skill loading). The full suite on the release tree: 23,277 passed, 66 skipped, 4 xfailed.

## Live checks

`tests/integration/test_browser_scroll_live.py` with `LOCALHARNESS_LIVE_BROWSER=1`, through the production
renderer: nine offline pages, each built to break a naive scroll loop, all pass.

| Page | Result |
| --- | --- |
| a feed that loads only at the very bottom, ten pages, slow API | all ten pages |
| an inner scroller under a fixed header | all 100 items |
| a virtual list that drops rows as it scrolls | first screen kept |
| a feed that never ends | returns in budget, says the page holds more |
| a page that forges its height | stops at once, no large read |
| a wheel handler that blocks the page | first screen returned |
| a page that swaps in the next article | the first article, under its own address |
| a reader whose address follows the reading position | the whole chapter |
| a static page | the same text |

Real pages through `browser_fetch`: a lazy-loading web-serial chapter came back whole (14,278 characters,
ending included, 5.4 s) where the unscrolled read held 9,485; quotes.toscrape.com/scroll returned every author.

A ten-part run of a project skill that fetches a web serial, with `review: {critics: [spec-compliance]}`:
each part file held every story paragraph of its saved page, in order, with nothing added and no site
clutter, checked paragraph by paragraph against the saved pages. The reply was reviewed by spec-compliance
alone, with 15 reads and 7 greps and no web call.
