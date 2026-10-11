"""The one incumbent sentence builder (Phase 5 D-14, D-29, D-32, D-45): every form and suffix
verbatim, every shown value control-safe, the unit inside the suggested command shell-quoted."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from localharness.cli import workspace
from localharness.cli.workspace import (
    Holder, incumbent_sentence, incumbent_subject, shown_path, window_clause,
)


@dataclass(frozen=True)
class State:
    """A stand-in for units.UnitState: the builder reads only these two attributes."""
    load_state: str = "loaded"
    unit_file_state: str = "disabled"


S = State()
UNIT = Holder("llama-server.service", 1234, "llama-server", 1000, S)


def _no_control(text: str) -> bool:
    return all(ord(c) >= 0x20 and ord(c) != 0x7F for c in text)


class TestAttachForm:
    def test_the_attach_sentence_verbatim(self):
        assert incumbent_sentence(UNIT, port=8080, form="attach") == \
            "llama-server.service holds :8080; the harness attaches"

    def test_no_unit_names_the_pid_and_command(self):
        h = Holder(None, 4321, "llama-server", 1000)
        assert incumbent_sentence(h, port=8080, form="attach") == \
            "pid 4321 (llama-server) holds :8080; the harness attaches"

    def test_no_unit_and_no_command(self):
        assert incumbent_subject(Holder(None, 4321, None, 1000), port=8080) == "pid 4321"

    def test_another_uid_with_no_visible_pid(self):
        h = Holder(None, None, None, 0)
        assert incumbent_subject(h, port=8080) == "uid 0's process on :8080"
        assert incumbent_sentence(h, port=8080, form="attach") == \
            "uid 0's process on :8080 holds :8080; the harness attaches"


class TestSuffixes:
    def test_unit_file_not_checked(self):
        h = Holder("llama-server.service", 1234, "llama-server", 1000, None)
        assert incumbent_sentence(h, port=8080, form="attach") == \
            "llama-server.service (unit file not checked) holds :8080; the harness attaches"

    def test_no_unit_file_by_that_name(self):
        h = Holder("run-u7.service", 1, "x", 1000, State(load_state="not-found"))
        assert incumbent_sentence(h, port=8080, form="attach") == \
            "run-u7.service (no unit file by that name) holds :8080; the harness attaches"

    def test_transient(self):
        h = Holder("run-u7.service", 1, "x", 1000, State(unit_file_state="transient"))
        assert incumbent_sentence(h, port=8080, form="attach") == \
            "run-u7.service (transient) holds :8080; the harness attaches"

    def test_an_enabled_or_static_unit_has_no_suffix(self):
        for ufs in ("enabled", "static", "disabled"):
            h = Holder("a.service", 1, "x", 1000, State(unit_file_state=ufs))
            assert incumbent_subject(h, port=1) == "a.service"


class TestImperativeForm:
    def test_stop_for_a_disabled_unit(self):
        assert incumbent_sentence(UNIT, port=8080, form="imperative", served="qwen-27b",
                                  target="flash-next") == (
            "qwen-27b is served by llama-server.service (pid 1234), not by the harness; "
            "systemctl --user stop llama-server.service, then /model flash-next")

    def test_disable_now_for_an_enabled_unit(self):
        h = Holder("llama-server.service", 1234, "llama-server", 1000,
                   State(unit_file_state="enabled"))
        assert incumbent_sentence(h, port=8080, form="imperative", served="qwen-27b",
                                  target="flash-next") == (
            "qwen-27b is served by llama-server.service (pid 1234), not by the harness; "
            "systemctl --user disable --now llama-server.service, then /model flash-next")

    def test_a_unit_with_no_pid_has_no_pid_part(self):
        h = Holder("llm.service", None, None, 1000, S)
        assert incumbent_sentence(h, port=8080, form="imperative", served="a", target="b") == (
            "a is served by llm.service, not by the harness; "
            "systemctl --user stop llm.service, then /model b")

    def test_suffix_rides_in_the_subject_but_not_the_command(self):
        h = Holder("run-u7.service", 5, "x", 1000, State(unit_file_state="transient"))
        assert incumbent_sentence(h, port=8080, form="imperative", served="a", target="b") == (
            "a is served by run-u7.service (transient) (pid 5), not by the harness; "
            "systemctl --user stop run-u7.service, then /model b")

    def test_no_unit_says_stop_it(self):
        h = Holder(None, 4321, "llama-server", 1000)
        assert incumbent_sentence(h, port=8080, form="imperative", served="qwen-27b",
                                  target="flash-next") == (
            "qwen-27b is served by pid 4321 (llama-server), not by the harness; "
            "stop it, then /model flash-next")

    def test_another_uid(self):
        h = Holder(None, None, None, 0)
        assert incumbent_sentence(h, port=8080, form="imperative", served="a", target="b") == (
            "a is served by uid 0's process on :8080, not by the harness; stop it, then /model b")

    def test_an_unknown_form_is_refused(self):
        with pytest.raises(ValueError):
            incumbent_sentence(UNIT, port=8080, form="shout")


class TestWindowClause:
    def test_running_unit(self):
        clause = window_clause(65536, 114688, running=True)
        assert incumbent_sentence(UNIT, port=8080, form="attach", window=clause) == (
            "llama-server.service serves -c 65536; the harness launches -c 114688 once it stops "
            "· server.context: 65536 keeps it")

    def test_stopped_unit(self):
        h = Holder("llm.service", None, None, 1000, S)
        clause = window_clause(65536, 114688, running=False)
        assert incumbent_sentence(h, port=8080, form="attach", window=clause) == (
            "llm.service sets -c 65536; the first start launches -c 114688 "
            "· server.context: 65536 keeps it")

    def test_unknown_fit_reads_with_no_c(self):
        assert window_clause(65536, None, running=True) == (
            "serves -c 65536; the harness launches with no -c once it stops "
            "· server.context: 65536 keeps it")
        assert window_clause(65536, None, running=False) == (
            "sets -c 65536; the first start launches with no -c "
            "· server.context: 65536 keeps it")

    def test_the_window_sentence_is_one_line(self):
        s = incumbent_sentence(UNIT, port=8080, form="attach",
                               window=window_clause(1, 2, running=True))
        assert "\n" not in s and "the harness attaches" not in s


class TestControlSafety:
    BAD = "x\x1b[2Ky\nz"
    SHOWN = "x\\x1b[2Ky\\x0az"

    def test_unit_command_served_and_target_are_escaped(self):
        for holder in (Holder(self.BAD, 1, "c", 1, S), Holder(None, 1, self.BAD, 1)):
            for form in ("attach", "imperative"):
                s = incumbent_sentence(holder, port=8080, form=form,
                                       served=self.BAD, target=self.BAD)
                assert _no_control(s)
                assert self.SHOWN in s

    def test_served_and_target_in_the_attach_window_path(self):
        s = incumbent_sentence(Holder(self.BAD, None, None, 1, S), port=1, form="attach",
                               window=window_clause(1, 2, running=False))
        assert _no_control(s) and self.SHOWN in s

    def test_a_unit_with_a_space_is_shlex_quoted_in_the_command(self):
        h = Holder("a b.service", 1, "x", 1, S)
        s = incumbent_sentence(h, port=1, form="imperative", served="s", target="t")
        assert "systemctl --user stop 'a b.service', then /model t" in s

    def test_an_escape_in_the_unit_is_quoted_after_escaping(self):
        h = Holder("a\x1bb.service", 1, "x", 1, S)
        s = incumbent_sentence(h, port=1, form="imperative", served="s", target="t")
        assert _no_control(s)
        assert "stop 'a\\x1bb.service'" in s

    def test_a_uid_is_never_a_raw_control_byte(self):
        assert _no_control(incumbent_subject(Holder(None, None, None, 0), port=1))


class TestShownPath:
    def test_home_prefix_is_tilde(self):
        assert shown_path(Path.home() / "models" / "a.gguf") == "~/models/a.gguf"

    def test_home_itself(self):
        assert shown_path(Path.home()) == "~"

    def test_outside_home_is_unchanged(self):
        assert shown_path("/srv/models/a.gguf") == "/srv/models/a.gguf"

    def test_a_sibling_sharing_the_prefix_is_not_abbreviated(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/al")
        assert shown_path("/home/alice/x") == "/home/alice/x"
        assert shown_path("/home/al/x") == "~/x"

    def test_escape_is_shown_not_printed(self):
        out = shown_path("/m/\x1b[2K/a\n.gguf")
        assert out == "/m/\\x1b[2K/a\\x0a.gguf"
        assert _no_control(out)


def test_visible_text_escapes_every_character_that_reorders_or_hides_text():
    """IN-05: line and paragraph separators, the Arabic letter mark, zero-width characters, the BOM."""
    for ch, shown in (("\u2028", "\\u2028"), ("\u2029", "\\u2029"), ("\u061c", "\\u061c"),
                      ("\u200b", "\\u200b"), ("\u200c", "\\u200c"), ("\u200d", "\\u200d"),
                      ("\ufeff", "\\ufeff"), ("\u202e", "\\u202e"), ("\u2066", "\\u2066"),
                      ("\x1b", "\\x1b"), ("\x85", "\\x85")):
        assert workspace.visible_text(f"a{ch}b") == f"a{shown}b"
    assert workspace.visible_text("naïve — ✓ 日本語") == "naïve — ✓ 日本語"


def test_builder_lives_beside_visible_text():
    assert workspace.visible_text("a\x1b") == "a\\x1b"
