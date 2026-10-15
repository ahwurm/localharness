# Spec 21 — Terminal Keys: A Line Break in the Box and a Nudge Chord of Your Own

**Status:** Vetted, revision 3 (three rounds of design, security and friction critics; round 3: **SHIP from all three**, with the eight round-3 minors folded in here — see §15)
**Version target:** 0.18.1
**API version:** `PLUGIN_API_VERSION` unchanged ("1")
**Numbering:** 18 is the task record (reserved by spec 17 §1), 19 the delegation surface, 20 tuned presets (0.19.0, on its own branch); this is 21.
**Origin:** GitHub issues #167 (*Multiline support*) and #168 (*Allow configuration of nudge sequence*), both opened 2026-10-03 by Brian Ealdwine (@eode), the first with a working evdev sample attached. Twelve days is too long to leave a contributor waiting; this is the answer.

---

## 1. Problem

Two reports from one person, about one input box.

1. **No way to type a multi-line message (#167).** Enter submits in both input apps (`channels/terminal.py:644` classic, `:755` persistent). Ctrl+J is Enter too (`:756`: WSL-style terminals send a line feed for Return, so binding it to anything else breaks every Enter there). Alt+Enter is the forced nudge (`:770`). A pasted block with bracketed paste keeps its newlines — prompt_toolkit's own paste binding inserts the text as it came — so multi-line messages already travel the whole submit path, but only by pasting. Nothing in the box, `/help` or the README says how to break a line, and no key does it. Spec 10's `PromptSession` sketch (`docs/specs/10-cli.md:1172`) promised "Alt-Enter: Insert newline (multi-line input)"; the implementation gave Alt+Enter to the nudge and the promise was never reconciled.
   The issue proposes reading the keyboard device through evdev when Enter arrives and checking whether Shift is down. §12 says why the harness will not.
2. **The nudge has one spelling and a wrapper eats it (#168).** Alt+Enter is the only chord (`:770`). On the reporter's desktop the GUI that wraps the terminal consumes Alt+Enter before the terminal sees it; from a bare tty it works. The typed form exists — a leading `!` forces a nudge (`channels/input_router.py:52 FORCE_PREFIX`, `strip_force` at `:55`) — but it is deliberately undocumented ("kept out of help/docs", `tests/unit/test_terminal_input_box.py:306`), so the person who cannot press Alt+Enter has no nudge at all and no way to learn the one they have.

Both are the same gap: the box's keys are literals in one function, invisible to the person and closed to the machine they run on.

## 2. Ground truth (what exists)

Line numbers against `ffdf2b4` (main, v0.18.0). Every row was re-checked by the round-1 critics.

| Thing | Where | Fact |
|---|---|---|
| Persistent box bindings | `channels/terminal.py:698 _build_persistent_input_app` | `enter` and `c-j` submit (`:755-768`); `escape enter` submits `FORCE_PREFIX + text` (`:770-783`); `c-c`, `c-d`, `c-y`, `c-n`, `c-v`; `tab` / eager `escape` for the completion menu (`_add_menu_keys :374`, `:404`). `Buffer(multiline=False)` (`:737`); the input `Window(wrap_lines=True, dont_extend_height=True)` has no height and grows with the document. The submitted text is stripped in the REPL (`clean = text.strip()`), not in the box. |
| Classic app | `:625 _build_input_app` | `enter` submits and exits the app (`:644`), the caller strips (`:1534`); `c-c`, `c-d`; no nudge binding (no turn runs while it is up) and no legend. Alt+Enter there is Escape then Enter: a plain submit. |
| The nudge | `:770`, `cli/repl.py:898 _route_during_turn` | The chord submits `"!" + text`; the REPL delivers `forced` as `Route.NUDGE` ("force-bang"); with no turn running `strip_force` makes it an ordinary line. The `InputRouted` event carries `text_preview=clean[:80]` (`repl.py:946`, `:993`). |
| Footer legend | `:1779 _box_hint_frags` | Working: the literal `alt+enter nudge · ctrl+c stop`; idle: `tab commands · /help`. |
| `/help` | `cli/slash_commands.py:101 help_text()` | The slash rows and one closing sentence; no keys. Printed by `repl.py:1602`. |
| prompt_toolkit | `uv.lock`: 3.0.52, pinned `>=3.0,<4` | `input/ansi_escape_sequences.py:129` maps the xterm `modifyOtherKeys` Shift+Enter sequence `ESC [ 27 ; 2 ; 13 ~` to `Keys.ControlM` with the comment "currently unsupported". The kitty-protocol form `ESC [ 13 ; 2 u` is not in the table, so the parser emits an Escape key followed by the literal characters `[13;2u`. `ANSI_SEQUENCES` is a module-level dict the vt100 parser reads at match time (`vt100_parser.py:62`); `vt100_parser._IS_PREFIX_OF_LONGER_MATCH_CACHE` (`:67`, a private name) memoises which prefixes can still match. `Keys.ControlF3` is an enum member no table entry produces: its xterm sequence `ESC [ 1 ; 5 R` collides with the cursor-position report, so the table comments it out (`:136`). prompt_toolkit's own emacs bindings give `c-o` operate-and-get-next (`bindings/emacs.py:106`) and `c-g` abort-search / cancel-selection (`:247`, `:255`, `:357`); an application's own binding on the same key wins. Verified on this box with a pipe input: adding entries to the dict and clearing the cache before the application starts makes a binding on `Keys.ControlF3` fire for both Shift+Enter encodings while `enter`, `c-j` and `escape enter` keep their meaning and the buffer holds `a\nb\nc`. |
| What terminals send | kitty's keyboard-protocol page, "Legacy functional keys" and "Legacy text keys" | Without a protocol negotiated, kitty — and Ghostty, WezTerm, Alacritty and foot, which copy its legacy table — sends `0x0d` for Shift+Enter, the same byte as Enter, and `ESC 0x0d` for Alt+Enter. So **today, and after this spec with nothing configured, Shift+Enter on those terminals submits the half-written message.** The distinguishable `CSI 13 ; 2 u` appears only after an application pushes the protocol's "disambiguate" flag, which also re-encodes Esc and every Ctrl and Alt chord (§13). xterm sends `CSI 27 ; 2 ; 13 ~` only under `modifyOtherKeys`. Every terminal in common use can be told to send a chosen byte string for Shift+Enter (§8), and each names itself in the environment (`KITTY_WINDOW_ID`, `TERM_PROGRAM`, `TERM`, `WT_SESSION`). |
| Config | `config/models.py:2314 TerminalConfig` | `inputbox_enabled`, `input_router_tier2_enabled`, `show_reasoning`; `extra="forbid"`; each leaf names its `localharness components set terminal.<leaf>` form. A nested model's leaves are catalogued by `walk_model_fields`, a list as one leaf. |
| Machine-only keys | `config/loader.py:506 HARNESS_GLOBAL_ONLY_FIELDS` and its docstring (`:513-531`, one rationale paragraph per key) | A workspace value at a member path, or inside a member section, is dropped by `_narrow_raw` with `ignoring <path> in <file>: only the global config may set it` — a workspace block that merely restates a whole-section member's defaults warns too (the `autosense` precedent). SECURITY.md `### Machine-level-only settings` (`SECURITY.md:205`) must list the same set, both ways (`tests/unit/test_docs_convergence.py check_machine_only_list`); its lead-in sentence says what the listed keys have in common. |
| `components set` on a list | `registry/coerce.py:77`, `cli/components_cmd.py:455` | A `list[...]` annotation is "best-effort passthrough": the raw string reaches the model's validation unchanged. Each `set` validates the whole merged `HarnessConfig` after the one change, so two settings that only validate together cannot be made one at a time. |
| The channel | `channels/terminal.py:1033 TerminalChannel.__init__(bus, config, history_file, *, no_input)` | Built once, `cli/start_cmd.py:2737`. The REPL reads `terminal.inputbox_enabled` with `getattr(self._harness, "terminal", None)` (`repl.py:604`). |

## 3. Goals

- **G1** A line break in the box on every terminal with nothing configured: a backslash, then Enter.
- **G2** Shift+Enter breaks the line on any terminal that sends either known encoding; the harness recognises both, and tells the person at `/help` the one line that makes their terminal send it.
- **G3** The nudge chord is the person's choice from a closed set; the legend, `/help` and the README say what it is, and the typed `!` form is documented beside it.
- **G4** A multi-line message is one message: one echo, one history entry, one submission, routed exactly as a pasted block is today.
- **G5** A person who configures nothing sees no change to any key's meaning except one: Enter after a line that ends in a single `\` now breaks the line instead of submitting (§6.1, in the CHANGELOG's *Changed*). Enter submits, Ctrl+J is Enter, Alt+Enter nudges, Shift+Enter on a terminal that sends it as Enter still submits.

## 4. Non-goals

- Reading input devices (§12).
- Negotiating a keyboard protocol with the terminal (kitty `CSI > 1 u`, xterm `modifyOtherKeys`) — §13, which is the real fix for the five kitty-family terminals; §8's table is the stopgap until then.
- Free-form key strings in config. The vocabulary is closed (§5).
- The phone, Discord and Zed surfaces: each has its own line break and its own nudge button.
- The `!` prefix's meaning, the router's tiers, the audit of routing decisions: unchanged.

## 5. Config shape

```yaml
terminal:
  keys:
    nudge: [alt-enter]        # chords that send the line into the running turn now
    newline: [shift-enter]    # chords that insert a line break
```

`TerminalKeysConfig` nests on `TerminalConfig` as `keys`, `extra="forbid"`, both fields `list[ChordName]`. A value given as a string is read as a comma-separated list (`ctrl-o`, `alt-enter, ctrl-o`), so `localharness components set terminal.keys.nudge ctrl-o` works through the catalogue's passthrough (§2) and the model's validation, and so an override file that holds the string reads the same as one that holds the list.

The vocabulary, `ChordName`, four names:

| name | on the wire | reaches the harness on |
|---|---|---|
| `alt-enter` | `ESC CR` | every terminal, unless a window manager or wrapper takes the chord first (#168) |
| `shift-enter` | `ESC [ 13 ; 2 u` or `ESC [ 27 ; 2 ; 13 ~` | a terminal set up to send it — §8; elsewhere it is Enter, as today |
| `ctrl-g` | `0x07` | every terminal |
| `ctrl-o` | `0x0f` | every terminal |

Rules:

- An unknown name is refused by the schema with the loader's usual one line naming the file and key.
- **Precedence (DEC21-precedence).** A list a person set wins over a list still at its default: a chord the set list names is removed from the defaulted list, silently, because the person just said where that chord goes. Only when both lists were set and share a chord is the config refused, with one line built from the colliding chord and the first chord of `ctrl-o`, `ctrl-g` that neither list uses: `terminal.keys: alt-enter is in both nudge and newline — set one list without it first (e.g. localharness components set terminal.keys.nudge ctrl-o), then retry`. So `components set terminal.keys.newline alt-enter` alone moves Alt+Enter to the line break and leaves no nudge chord (`!` remains, and the legend says so); `… nudge ctrl-o` afterwards gives the nudge a chord again. Which lists were set is `model_fields_set` on the validated model, the same whether the value came from `config.yaml`, `overrides.yaml` or one `components set`: the loader validates the merged raw layers (`loader.py:1286-1291`, never a dump of the defaults), `components set` validates `raw_harness_dict()` plus the new overlay (`components_cmd.py:161-163`), and the machine-only narrowing runs before validation, so a dropped workspace block can never count as set. A person who wrote both lists into `config.yaml` by hand (even restating the defaults) has set both, and must then move a chord in the order that never collides — the README says so. The resolution is a `mode="after"` validator that reads `model_fields_set` before it touches anything and writes the trimmed list with `object.__setattr__`, so the set stays what the person wrote (a plain assignment would add the field to it); the resolved state is always disjoint, so a `model_dump()` of it validates again, but a dump-and-reload is never a config path (the dump marks every field as set).
- **Applies at the next start.** The box is built once with the keys it is given; a change written by `components set` to a `terminal.keys` path takes effect at the next `localharness start`. The command's `wrote:` line says so for those paths, as a dim parenthetical beside the file name (`wrote: overrides.yaml (the global overrides.yaml; applies at the next start)`), and the `--json` object gains `"applies": "next-start"`.
- Either list may be empty. With no nudge chord `!` still nudges; with no newline chord a backslash before Enter still breaks the line.
- Not in the vocabulary, on purpose: `enter`, `ctrl-j` (Enter on LF terminals, `terminal.py:756`), `tab`, `escape`, `ctrl-c`, `ctrl-d`, `ctrl-y`, `ctrl-n`, `ctrl-v` — the box's own keys — and `ctrl-enter`, which no terminal sends without the §13 negotiation and which the README could not teach in one line (round 1).
- Why these four: the two Enter chords are the ones people reach for, and Shift+Enter is the one a terminal can be told to encode in one line; Ctrl+G and Ctrl+O are the control characters that arrive on every terminal, that no desktop environment binds by default and that the box does not use. prompt_toolkit's own emacs defaults for them (operate-and-get-next, abort a search or a selection) are shadowed by the box's binding, which wins; a person who picks `ctrl-g` gives up aborting a `c-r` history search with it, which is their choice. Adding a name later is one `Literal` member, one table row and one label (§7).
- Lists, not single names, so a person can keep Alt+Enter and add a second spelling of the same action.

**Machine-only.** `terminal.keys` (the whole section, listed as exactly that one member — not its two leaves, unlike `instructions.*`) joins `HARNESS_GLOBAL_ONLY_FIELDS`, with its own paragraph in the loader's docstring and one clause in SECURITY.md's lead-in: the listed keys say where a request goes, what starts, what is written — and, from here, *what the person's own keys do*, because a project must not be able to rebind which key sends a line into a running turn on the person's terminal. A workspace value is dropped with the loader's standard line; a workspace block that restates the defaults warns too, as `autosense` does.

## 6. The line break and the multi-line message

**6.1 Backslash, then Enter — always on, in both apps.** Enter with the buffer text ending in exactly one `\` (the character before it is not a backslash) and the caret at its end replaces that backslash with a newline and does not submit. Text ending in `\\` submits as typed, both backslashes kept. A message that must end in exactly one backslash: type it, a space, then Enter — the trailing space is stripped on submit, as every trailing space is today. With the completion menu open, Enter still accepts the highlighted command first (`_accept_completion`); the backslash rule applies only when no completion was applied in that stroke. This is the one key whose meaning changes for an unconfigured install (G5).

**6.2 Newline chords.** Each configured chord inserts `\n` at the caret, in both apps, with no completion handling: a chord means "break here", never "accept". One exception the menu forces: `alt-enter` as a newline chord with the completion menu open is Escape then Enter — the eager Escape (`_add_menu_keys :404`) dismisses the menu and the Enter submits, exactly as the nudge chord behaves there today (`test_escape_dismissing_a_menu_is_not_the_nudge_chord`). `shift-enter`, `ctrl-g` and `ctrl-o` are single keys and break the line with the menu open.

**6.3 Nudge chords.** Each configured chord does what `escape enter` does today (`:771-783`): submit `FORCE_PREFIX + text`, no completion accept, nothing on an empty line. Only the persistent box binds them (the classic app has no turn to steer). A chord in neither list is unbound: Alt+Enter removed from both lists is Escape then Enter, a plain submit, exactly as the classic app treats it today.

**6.4 Buffer and box.** `Buffer(multiline=False)` stays: prompt_toolkit's `multiline` filter governs only its own Enter default, which both apps override, and a newline in the document already renders as a row (today's paste path). The input `Window` gains `height=Dimension(max=8)`: the box grows to eight rows and scrolls past them, so a long paste never pushes the status row and footer off the screen. Up and Down inside a multi-row document move by row and reach history from the first and last row — prompt_toolkit's `auto_up` / `auto_down` (`bindings/basic.py:208-212`), already loaded.

**6.5 Submit.** `on_submit(text)` carries the newlines; `buf.append_to_history()` stores one entry (`FileHistory` writes continuation lines with its `+` prefix and reads them back as one). `_handle_input` sees one `user_input`; mid-turn `_route_during_turn(clean, forced)` classifies the whole text (tier 1 reads `text.strip().lower()`; a block that starts with `/` queues as a slash command, as a pasted one does). One line changes in the REPL: the `InputRouted` event's `text_preview` collapses runs of whitespace to one space before the `[:80]` cut, so the audit row stays one row. The echo (`box_echo_prompt`) is untouched: it already prints a multi-row text as rows, markup-escaped.

**6.6 Legend.** Idle: `tab commands · /help · \ then enter newline`. Working: `<nudge spellings> nudge · ctrl+c stop`, where the spellings are `!text` and the configured chords' labels: `!text or alt+enter nudge`, `!text, shift+enter or ctrl+o nudge`, `!text nudge` when the list is empty. `!text` rather than `!` so the placement is plain — the mark goes before the text. The `!` form is shown because it is the one spelling no wrapper can eat (#168) and the legend is the only line a person reads mid-turn. The classic app has no legend; its line break is documented in the README only, which is accepted: the classic app is the opt-out (`inputbox_enabled: false`).

**6.7 `/help`.** `help_text(keys_block: str | None = None)`: when the REPL passes a block it is printed after the rows; the REPL passes `keys.help_block(...)` (§7) only while the persistent box is live, so the phone, Discord and the classic app read as today. Two rows, the two things a person cannot find today:

```
Keys:
  \ then enter       line break. shift+enter submits until kitty is set up — kitty.conf: map shift+enter send_text all \x1b[13;2u
  !text, alt+enter   send the line into the running turn (a nudge)
```

The first row has three forms, chosen from what the harness knows and from the configured lists:

- `shift-enter` is a newline chord, the terminal is named from the environment (§7 `detect_terminal`) and no `shift-enter` has arrived yet: the honest sentence and the terminal's one setup line, as above.
- `shift-enter` is a newline chord and has already arrived this session: `\ then enter       line break — shift+enter too`.
- Otherwise — the terminal cannot be named (GNOME Terminal, Konsole, xterm, anything under tmux or over ssh with a bare `TERM`), or `shift-enter` is not a newline chord (so the setup line would do nothing, and a `shift-enter` nudge chord would contradict the second row): no Shift+Enter clause at all, and one route that works on every terminal and keeps Alt+Enter as the nudge: `\ then enter       line break (or give ctrl+o to it: localharness components set terminal.keys.newline ctrl-o)`. The route names the first of `ctrl-o`, `ctrl-g` that neither list uses, and is omitted when both are taken. Never a README pointer: the README has no row for these terminals either, and never Alt+Enter, which is the chord #168's wrapper eats.

The rows name the configured chords' labels; with no nudge chord the second starts `!text              `.

## 7. The chords, in code

Two modules, so the config side never imports prompt_toolkit:

**`config/chords.py`** (new, pure, no prompt_toolkit): `ChordName = Literal["alt-enter", "shift-enter", "ctrl-g", "ctrl-o"]`, `CHORD_NAMES = get_args(ChordName)`, `LABELS = {"alt-enter": "alt+enter", …}`. `config/models.py` imports `ChordName` from here; this is the one source of the vocabulary.

**`channels/keys.py`** (new; imports `config/chords.py` and prompt_toolkit's `Keys`, `ANSI_SEQUENCES` and, guarded, the vt100 prefix cache; nothing from the REPL or the channel):

- `SHIFT_ENTER_KEY = Keys.ControlF3`: the member no terminal produces (§2), named once here with the reason, so the enum name never appears in a binding.
- `KEYS: dict[ChordName, tuple[Keys | str, ...]]`: `alt-enter → ("escape", "enter")`; `shift-enter → (SHIFT_ENTER_KEY,)`; `ctrl-g → ("c-g",)`; `ctrl-o → ("c-o",)`. A test holds `KEYS.keys() == set(CHORD_NAMES)`.
- `install_wire_sequences() -> bool`: idempotent; writes the two constant Shift+Enter sequences into `ANSI_SEQUENCES` (the xterm entry is replaced — its remap to Enter was prompt_toolkit's placeholder, §2; the Ctrl+Enter entries `27;5;13~` and `27;6;13~` are left as they are, Enter) and clears the prefix cache. The cache is fetched with `getattr(vt100_parser, "_IS_PREFIX_OF_LONGER_MATCH_CACHE", None)`: if a later 3.x renames it, nothing is installed, the function returns `False`, one `logger.info` line says Shift+Enter is unavailable, and `\` then Enter, `alt-enter`, `ctrl-g` and `ctrl-o` still work. A test asserts the premise against the installed prompt_toolkit (no `ANSI_SEQUENCES` value is `Keys.ControlF3`; the cache name exists) so an upgrade that changes it fails the suite, not the person's input. **Each app builder calls it only when `shift-enter` is in one of the two lists**: a person who took `shift-enter` out of both keeps prompt_toolkit's own table, so the xterm form stays Enter for them (G5). The table is process-wide, so once installed the picker (`cli/picker.py`) and any later prompt in the process see it too: there a Shift+Enter is simply unbound (in the xterm form it was Enter before; nobody picks with Shift+Enter). The tests that touch the table snapshot and restore it.
- `nudge_legend(names) -> str`, `help_block(keys, *, terminal: tuple[str, str] | None, shift_enter_seen: bool) -> str` (§6.6, §6.7).
- `detect_terminal(env) -> tuple[str, str] | None`: the terminal's name and its one setup line from §8, read from the environment only, in this order — `TMUX` set or `TERM_PROGRAM=tmux` → `None` first, because the outer terminal's variables (`KITTY_WINDOW_ID`, `WT_SESSION`) are inherited into tmux and its line would be wrong there; then `KITTY_WINDOW_ID` or `TERM=xterm-kitty` → kitty; `TERM_PROGRAM=ghostty` or `TERM=xterm-ghostty` → Ghostty; `TERM_PROGRAM=WezTerm` → WezTerm; `TERM=alacritty` or `ALACRITTY_WINDOW_ID` → Alacritty; `TERM=foot` / `foot-extra` → foot; `TERM_PROGRAM=iTerm.app` → iTerm2; `TERM_PROGRAM=vscode` → VS Code; `WT_SESSION` → Windows Terminal (under WSL, the only place the harness's vt100 input path runs it); anything else (GNOME Terminal, Konsole, xterm, ssh with a bare `TERM`) → `None`. Nothing is queried on the wire. Nothing in the harness writes these variables into its own process (the only `os.environ` writes under `src/` are the embeddings flags and the python tool's marker), and the worst a wrong value can do is pick a different help line.
- `shift_enter_seen` lives on `TerminalChannel`, not in the app: the box is rebuilt when a permission question suspends and restarts it (`_box_restart_args`), so a flag inside the builder would reset. The builder takes an `on_newline_chord` callback and calls it only when the chord fires **as a newline** (a `shift-enter` nudge chord says nothing about line breaks); the channel exposes the flag with the keys through one accessor, `keys_help_state()`, which the REPL's `/help` reads through `getattr`, as it reads `inputbox_enabled` today — the box lives in `channels/terminal.py`, `/help` in `cli/repl.py`.

Why not a two-key value such as `(Escape, ControlJ)` for Shift+Enter: the completion menu's Escape binding is eager (`_add_menu_keys :404`) and would split the pair, and on an LF terminal the pair *is* Alt+Enter. Why not teach every terminal to send `\` then CR and rely on §6.1 alone: a Shift+Enter in the middle of a line would then insert a backslash and submit. Why not negotiate a protocol: §13.

Wiring: `TerminalChannel.__init__(..., keys: TerminalKeysConfig | None = None)` (default: the model's defaults, so every existing constructor call and test stands); `start_cmd.py:2737` passes `keys=harness.terminal.keys`; both builders take `keys=`; `_box_hint_frags` reads the keys, and the REPL's `/help` reads `keys_help_state()`.

## 8. Terminals (README text)

The README gains a short **Keys** subsection under *Quick start*. Its first sentence is the honest one: **Shift+Enter submits until your terminal is set up to send it; `\` then Enter always breaks the line.** Then: Alt+Enter, or `!` before the text, sends the line into the running turn; `terminal.keys` changes the chords; `/help` names your terminal's setup line. Then one table — how to make Shift+Enter send `ESC [ 13 ; 2 u`, which the harness reads as a line break:

| terminal | one line |
|---|---|
| kitty | `kitty.conf`: `map shift+enter send_text all \x1b[13;2u` |
| Ghostty | config: `keybind = shift+enter=text:\x1b[13;2u` |
| WezTerm | `keys`: `{ key = "Enter", mods = "SHIFT", action = wezterm.action.SendString("\x1b[13;2u") }` |
| Alacritty | `[[keyboard.bindings]] key = "Enter", mods = "Shift", chars = "\u001b[13;2u"` |
| foot | `[text-bindings]`: `\x1b[13;2u=Shift+Return` |
| iTerm2 | Profiles → Keys → add Shift+Enter → *Send Escape Sequence* → `[13;2u` |
| VS Code terminal | `keybindings.json`: `{ "key": "shift+enter", "command": "workbench.action.terminal.sendSequence", "args": { "text": "\u001b[13;2u" }, "when": "terminalFocus" }` |
| Windows Terminal (WSL) | `actions`: `{ "command": { "action": "sendInput", "input": "\u001b[13;2u" }, "keys": "shift+enter" }` — the harness reads the sequence under WSL; native Windows uses another input path and is not covered |

A terminal with no such binding (GNOME Terminal, Konsole, xterm), or anything under tmux: `\` then Enter, or give a chord that every terminal sends to the line break — one command, and Alt+Enter stays the nudge:

```
localharness components set terminal.keys.newline ctrl-o
```

If you would rather have Alt+Enter break the line, `… newline alt-enter` does that in one command too; by DEC21-precedence it takes Alt+Enter from the nudge (the `!` prefix remains), and `… nudge ctrl-o` gives the nudge a chord again. A change applies at the next `localharness start`. If you wrote both lists into `config.yaml` yourself, the order matters: move a chord out of one list before you add it to the other.

## 9. Files touched

| File | Change |
|---|---|
| `src/localharness/config/chords.py` | new: `ChordName`, `CHORD_NAMES`, `LABELS` — the one vocabulary |
| `src/localharness/channels/keys.py` | new: `SHIFT_ENTER_KEY`, `KEYS`, `install_wire_sequences` (guarded), `nudge_legend`, `help_block`, `detect_terminal` (§7) |
| `src/localharness/channels/terminal.py` | both builders take `keys` and an `on_newline_chord` callback, and install the wire sequences only when `shift-enter` is configured; backslash-Enter; newline and nudge chords from the table; the 8-row cap; `shift_enter_seen` on the channel and `keys_help_state()`; the legend; `TerminalChannel(..., keys=)` |
| `src/localharness/config/models.py` | `TerminalKeysConfig` (string-or-list reading, DEC21-precedence as a `mode="after"` validator on `model_fields_set`, the both-set refusal with its command), `TerminalConfig.keys` |
| `src/localharness/config/loader.py` | `"terminal.keys"` (the section, one member) in `HARNESS_GLOBAL_ONLY_FIELDS` and its paragraph in the docstring |
| `src/localharness/cli/start_cmd.py` | `keys=` on the one construction |
| `src/localharness/cli/components_cmd.py` | the `wrote:` line's `applies at the next start` parenthetical and the JSON `"applies": "next-start"` field for paths under `terminal.keys` |
| `src/localharness/cli/slash_commands.py` | `help_text(keys_block=None)` |
| `src/localharness/cli/repl.py` | `/help` passes the block while the box is live; `text_preview` whitespace collapse at the two `InputRouted` sites |
| `SECURITY.md` | the `terminal.keys` bullet and one clause in the section's lead-in |
| `README.md` | the *Keys* subsection and the terminal table (§8) |
| `docs/specs/10-cli.md:1172` | the sketch's line becomes `Enter: submit · \ then Enter (Shift+Enter when the terminal sends it): newline · Alt+Enter: nudge — spec 21` |
| `CHANGELOG.md` | the 0.18.1 entry, crediting @eode for #167 and #168; *Changed* names the trailing-`\` Enter |
| `pyproject.toml`, `src/localharness/__init__.py`, `uv.lock`, `README.md` | 0.18.1 at release (`tests/unit/test_version_source.py` holds them to one number) |
| `tests/unit/test_terminal_keys.py` | new (§10) |
| `tests/unit/test_terminal_input_box.py`, `tests/unit/test_slash_commands.py`, `tests/unit/test_components_set*.py` (or the file that owns `components set`) | additions (§10) |

No new dependency (`tests/unit/test_declared_dependencies.py`). No plugin API change.

## 10. Tests

Unit, all hermetic (pipe input and `DummyOutput`, as `TestPersistentAppKeybindings._drive` does today; a fixture snapshots and restores `ANSI_SEQUENCES` and the prefix cache around every test that installs):

- **chords.py / keys.py.** `KEYS.keys() == set(CHORD_NAMES)` and every name has a label. `install_wire_sequences()` twice leaves two entries and the cache cleared, returns `True`; the xterm entry no longer reads `ControlM`; `27;5;13~` still does. The premise: no value of the installed `ANSI_SEQUENCES` is `Keys.ControlF3`, and the cache attribute exists. With the cache attribute monkeypatched away: returns `False`, installs nothing, logs once. `nudge_legend` for one, two and zero names. `detect_terminal` for each of the eight environments, tmux (`TMUX` set with `KITTY_WINDOW_ID` also set → `None`; `TERM_PROGRAM=tmux` → `None`) and an empty environment. `help_block` in its three first-row forms (named terminal with `shift-enter` a newline chord; `shift_enter_seen`; otherwise — unnamed terminal, `shift-enter` absent, `shift-enter` a nudge chord only — with the `ctrl-o` route, the `ctrl-g` route when `ctrl-o` is taken, and no route when both are), with each list empty.
- **Persistent box.** Default keys: `a` + `ESC [ 13 ; 2 u` + `b` + `ESC [ 27 ; 2 ; 13 ~` + `c` + CR submits exactly `["a\nb\nc"]`; `x\` + CR + `y` + CR submits `["x\ny"]`; `x\\` + CR submits `["x\\\\"]` (both backslashes); `x\ ` + CR submits `["x\\ "]` from the box and the REPL's strip leaves `x\`; the completion menu open plus `\` then CR accepts the command and does not break the line; `ESC CR` still forces (`test_alt_enter_submits_as_forced_nudge` stands); `nudge: [ctrl-o]`: `0x0f` forces, `ESC CR` submits plainly; `nudge: [ctrl-g]`: `0x07` forces; `newline: [alt-enter]` (nudge emptied by precedence): `ESC CR` breaks the line, and with the menu open dismisses it and submits; `nudge: []`: `ESC CR` submits plainly and `!x` still passes through; `nudge: [shift-enter]`: `ESC [ 13 ; 2 u` forces and `shift_enter_seen` stays false; `newline: [alt-enter]` and `nudge: [ctrl-o]` (no `shift-enter` anywhere): the builder does not install, and `ESC [ 27 ; 2 ; 13 ~` submits as Enter; the menu's eager Escape still dismisses with a plain submit (`test_escape_dismissing_a_menu_is_not_the_nudge_chord` stands); the Window's `height.max` is 8 (structural); `shift_enter_seen` flips on the first encoded chord that breaks a line, survives a box suspend-and-restart, and `keys_help_state()` reports it.
- **Classic app.** The same line-break cases; a nudge chord is unbound there (`ESC CR` submits plainly).
- **Config.** Defaults; an unknown name refused; a string value reads as the list; `newline: [alt-enter]` alone yields `nudge: []`; `nudge: [shift-enter]` alone yields `newline: []`; both set and sharing a chord refused with the one line, whose example names a chord neither list uses; both set and disjoint accepted; `model_fields_set` is unchanged by the resolution; every resolved state round-trips through `model_validate(model_dump())`; `components set terminal.keys.newline ctrl-o` and `… newline alt-enter` against a default file each succeed, the `wrote:` line carries `applies at the next start` and the JSON object `"applies": "next-start"`, and after the latter `… nudge ctrl-o` succeeds (one at a time); `… newline alt-enter` against a `config.yaml` that restates both defaults is refused with the line; `terminal.keys` in `HARNESS_GLOBAL_ONLY_FIELDS` (and neither leaf), `is_harness_global_only("terminal.keys.nudge")`; a workspace `terminal.keys` is dropped with the loader's line and the machine's own value wins (the `test_instructions_config.py:68` shape); `check_machine_only_list` passes against the new SECURITY.md bullet.
- **Legend and help.** `_box_hint_frags` working reads `!text or alt+enter nudge · ctrl+c stop` by default, `!text or ctrl+o nudge …` with `[ctrl-o]`, `!text nudge …` with `[]`; idle reads `tab commands · /help · \ then enter newline`. `help_text()` is byte-identical to today; `help_text(keys_block=…)` appends the block; the REPL passes no block when the box is off.
- **Router.** An `InputRouted` for `"a\n\nb"` carries `text_preview == "a b"`.
- **Docs.** `test_docs_convergence` (the machine-only list, the denylist: no `Phase N`, plan or requirement ids in README, SECURITY.md, CHANGELOG or `docs/**`), `test_release_docs`, `test_version_source` green.

Invariant: `uv run pytest -q tests/unit -p no:cacheprovider` green from the baseline at `ffdf2b4` — 13,903 passed, 4 skipped, 4 xfailed, 0 failed (measured 2026-10-15 on the worktree).

## 11. Security

- **No bytes from a config file reach the terminal's input parser or its screen.** The vocabulary is a closed `Literal`; the only sequences written into prompt_toolkit's table are two constants in `channels/keys.py`. Terminal detection reads the environment and queries nothing.
- **Machine-only.** `terminal.keys` is dropped from a workspace layer with the standard line; a cloned repository cannot rebind what the person's keys do. SECURITY.md's lead-in gains the clause that says why this key belongs on the list.
- **Nothing is sent to the terminal to change its keyboard mode**, so nothing is left behind by a crash (the reason §13 is deferred).
- **No device access.** §12.
- The chords move text inside the buffer or submit it through the path the `!` prefix already uses; the router, the gate and the `Ctrl+Y` / `Ctrl+N` answers are untouched; the audit event's preview is one row. The echo is unchanged.
- A multi-line message's size is unchanged (a paste is unbounded today and stays so); the box's drawn height is capped at eight rows.
- A hostile output stream cannot forge a chord: nothing is queried or pushed, and an unknown sequence is typed as text, as today.

## 12. Decisions

- **DEC21-no-evdev — the harness does not read `/dev/input`.** (1) It needs the `input` group or root, and membership makes every keystroke on the machine, in every window, readable by this process and by anything it launches: a keylogger grant the harness would be asking people to make for a line break. (2) It is a race the issue itself names: Enter has already arrived through the terminal before the modifier state is read, and a quick release reads as a plain Enter. (3) It is Linux-only and sees nothing over ssh, in tmux or on the phone. (4) A terminal that encodes the chord answers the question without any of it. The sample attached to #167 is acknowledged with thanks; the choice is to read the terminal's own encoding and document how to produce it.
- **DEC21-alt-enter-stays-nudge.** The default is unchanged; spec 10's sketch line is corrected to what ships.
- **DEC21-precedence.** A list the person set wins over a list at its default; two set lists that collide are refused with the fix named (§5). Chosen over a hard refusal because each `components set` validates the whole config, so a refusal would make the documented one-command change impossible (round 1, both critics).
- **DEC21-closed-vocabulary.** Four names (§5); `ctrl-enter` waits for §13.
- **DEC21-borrowed-member.** `Keys.ControlF3` carries the encoded Shift+Enter: prompt_toolkit offers no way to add a key, this member can never arrive from a terminal, the pair-of-keys form breaks on the menu's eager Escape, and the "send `\` then CR" form breaks mid-line (§7). The private-cache access is guarded and the premise is pinned by a test.
- **DEC21-backslash.** The one spelling that works on every terminal with nothing configured; `\\` is literal; the trailing-backslash edge is the same one every CLI with this rule has, and the escape (a space) is documented.
- **DEC21-bang-documented.** The `!` prefix enters `/help`, the legend and the README. It was kept out of them as "the router's job"; #168 shows it is the only nudge a wrapper cannot take away.
- **DEC21-string-or-list.** The two lists also read a comma-separated string so `components set` works unchanged.
- **DEC21-sense-the-terminal.** `/help` names the terminal's own setup line, read from the environment, instead of sending the person to the README (round 1, friction); for a terminal it cannot name, or when `shift-enter` is not a newline chord, it says nothing about Shift+Enter and offers the `ctrl-o` route, which every terminal sends and which leaves Alt+Enter as the nudge (rounds 2 and 3, friction and design).
- **DEC21-next-start.** A `terminal.keys` change applies at the next start, and `components set` says so; the box is not rebuilt mid-session for a key change (round 2, friction).

## 13. Out of scope

- **Protocol negotiation — the real fix for the kitty family.** Pushing kitty's "disambiguate" flag (`CSI > 1 u`) or xterm's `modifyOtherKeys` would make Shift+Enter work unconfigured on kitty, foot, Ghostty, WezTerm and Alacritty — and would re-encode Esc (`CSI 27 u`), every Ctrl chord (`CSI 99 ; 5 u` for Ctrl+C) and every Alt chord, which prompt_toolkit 3.0.52 does not read and cannot be taught through its exact-match table (a synthesised key carries the raw sequence as its text, so an Alt+letter would type garbage). It needs a CSI u reader of its own, pushed and popped on the box's lifecycle with a crash-safe reset. Until it lands, §8's table is the stopgap, and the README says so; the cost falls on every kitty-family user, so it is recommended for the next minor — the owner schedules it. `ctrl-enter` joins the vocabulary with it.
- More vocabulary (function keys, further Ctrl letters).
- A one-time flash when the backslash rule first fires.
- The phone, Discord and Zed.

## 14. Release note (CHANGELOG, 0.18.1)

Added: a line break in the terminal box (`\` then Enter anywhere; Shift+Enter on a terminal set up to send it, with the one-line setup per terminal in the README and the one for your terminal at `/help`) and `terminal.keys`, the chords for the nudge and the line break, machine-level only. Changed: Enter after a line ending in a single `\` now breaks the line instead of submitting; `/help` names the two keys; the box's legend names the configured nudge chord and the `!` prefix. Both proposed by @eode in #167 and #168 — with thanks, and with a note on why the evdev route was not taken (§12 here).

## 15. Iterations

**Round 1** (2026-10-15; design REVISE, security SHIP, friction REVISE; Sonnet-class critics from the repo's own `subagents/*.yaml`):
- Both majors were the both-lists refusal against one-at-a-time `components set` → DEC21-precedence (§5), §8 reordered to one command.
- `ctrl-enter` dropped: no one-line README route (friction), and its xterm sequence override turned Ctrl+Enter from Enter into a dead key for `modifyOtherKeys` users (security) → §5, §13.
- Ctrl+G / Ctrl+O rationale was wrong about prompt_toolkit's emacs defaults → reworded with the real bindings and the shadowing rule (§2, §5).
- Private cache name guarded, premise pinned by a test, table snapshot fixture (design, security) → §7, §10.
- `/help` block trimmed to the two rows that answer the issues, rendered by `keys.py`, passed as a prepared string, only while the box is live (friction, design) → §6.7, §7.
- Echo indent dropped: a multi-row text already echoes as rows (friction, security) → §6.5.
- `\\` defined, `!text` in the legend, `\ then enter` wording (friction) → §6.1, §6.6.
- Unconfigured Shift+Enter still submits on the kitty family: said plainly in §2, G5, §8's first sentence and §13; the terminal is sensed from the environment for `/help` → DEC21-sense-the-terminal.
- `alt-enter` as a newline chord with the menu open behaves as the nudge does (design) → §6.2.
- G5 softened for the trailing `\` (design) → §3, §14.
- One vocabulary source, `config/chords.py` (design) → §7.
- `InputRouted.text_preview` whitespace collapse (design) → §6.5.
- SECURITY.md lead-in clause and the loader docstring paragraph; the restated-defaults warning named (security, design) → §5.
- Ground-truth nits: `repl.py:604` / `self._harness`; where the strip happens (security) → §2.
- Owner's notes recorded, not adopted: teach terminals to send `\` then CR (breaks mid-line, §7); nudge-wins precedence with a warning (explicit-wins is quieter and reversible, §5).

**Round 2** (2026-10-15; design SHIP, security SHIP, friction REVISE on one item):
- `/help` for a terminal the harness cannot name pointed at a README with no row for it (friction, MAJOR) → the three first-row forms in §6.7: nothing about Shift+Enter there, the Alt+Enter route instead; the named-terminal row now says Shift+Enter submits until the terminal is set up.
- Wire sequences installed only when `shift-enter` is configured, so the xterm form stays Enter for everyone else (design) → §7.
- `shift_enter_seen` only on a newline fire; `keys_help_state()` accessor (design) → §7.
- `TMUX` checked first in `detect_terminal` (design, friction) → §7, §10.
- Hand-written both-lists config is order-sensitive; said in the README (design) → §5, §8.
- `mode="after"` validator, round-trip test, dump-and-reload never a config path (design, security's note) → §5, §10.
- Windows Terminal row scoped to WSL (design) → §8.
- Refusal names a command (friction) → §5.
- A change applies at the next start, and the receipt says so (friction) → DEC21-next-start.
- Loader docstring lists the section, not its leaves (security) → §5, §9.
- Owner's notes recorded, not adopted: an idle-legend notice naming the kitty-family setup line once per session (a line on every start until configured; `/help` carries it instead); a `components set` receipt that names the list precedence emptied (the working legend shows it).

**Round 3** (2026-10-15; **SHIP from all three**; eight minors folded in):
- The unnamed-terminal route was `newline alt-enter` — the chord #168's wrapper eats, and it cost the nudge chord (friction) → the route is `newline ctrl-o` (or `ctrl-g`), §6.7, §8.
- The Shift+Enter clause and the setup line shown only when `shift-enter` is a newline chord (design, friction) → §6.7.
- The refusal's example built from the colliding chord and a free one, with "then retry" (friction) → §5.
- The validator reads `model_fields_set` before it writes and writes with `object.__setattr__`; the set is asserted unchanged (design) → §5, §10.
- `shift_enter_seen` lives on the channel, set through an `on_newline_chord` callback, so a box suspend-and-restart keeps it (design) → §7, §10.
- The receipt clause sits on the `wrote:` line and `--json` gains `"applies": "next-start"` (design, friction) → §5, §9, §10.
- Owner's notes recorded, not adopted: remembering a prior `shift-enter` arrival across sessions (state on disk to save one conditional line); hot-rebuilding the box on a keys change (DEC21-next-start stands).
