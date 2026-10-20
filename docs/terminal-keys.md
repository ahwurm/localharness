# Terminal keys: line breaks, submit and the nudge

**Shift+Enter submits until your terminal is set up to send it; `\` then Enter always breaks the line.** Alt+Enter, or `!` before the text, sends the line into the running turn. `terminal.keys` changes the chords, and `/help` names your terminal's setup line. To send a message that ends in one `\`, type a space after it.

How to make Shift+Enter send `ESC [ 13 ; 2 u`, which the harness reads as a line break:

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

A terminal with no such binding (GNOME Terminal, Konsole, xterm), or anything under tmux: `\` then Enter, or give a chord that every terminal sends to the line break. It is one command, and Alt+Enter stays the nudge:

```bash
localharness components set terminal.keys.newline ctrl-o
```

If you would rather have Alt+Enter break the line, `localharness components set terminal.keys.newline alt-enter` does that in one command too. A list you set wins over one left at its default ([0.18.1](../CHANGELOG.md#0181--2026-10-15)), so it takes Alt+Enter from the nudge (the `!` prefix remains), and `localharness components set terminal.keys.nudge ctrl-o` gives the nudge a chord again. A change applies at the next `localharness start`.
