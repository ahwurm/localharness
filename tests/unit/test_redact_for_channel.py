"""`model_ops.redact_for_channel`: the one redaction every channel applies to a launch tail (D-22, SEC-04).

It masks the secrets the config holds (a SecretStr anywhere, a peer's included), the key a key file
supplied, `Bearer` tokens, llama-server's `api_keys: ****wxyz` startup line and URL credentials; the
rest of the text is untouched, and redaction never raises into a channel.
"""
from __future__ import annotations

import pytest
from pydantic import SecretStr

from localharness.cli import model_ops
from localharness.cli.model_ops import redact_for_channel
from localharness.config.models import EndpointRef, HarnessConfig, OrgConfig, ProviderConfig
from localharness.config.redact import SECRET_MASK

CONFIG_KEY = "sk-config-4f2a91"
PEER_KEY = "sk-peer-88c1d7"
FILE_KEY = "from-file-123"


def _harness(provider_key: str = CONFIG_KEY, peer_key: str | None = PEER_KEY) -> HarnessConfig:
    peers = []
    if peer_key is not None:
        peers.append(EndpointRef(name="flash-next", base_url="http://127.0.0.1:8081/v1",
                                 provider_type="llamacpp", api_key=peer_key))
    return HarnessConfig(
        provider=ProviderConfig(provider_type="llamacpp", base_url="http://127.0.0.1:8080/v1",
                                default_model="m", available_models=["m"], api_key=provider_key),
        org=OrgConfig(default_model="m"),
        extra_endpoints=peers,
    )


TAIL = (
    "srv  load: using key {config}\n"
    "peer key was {peer}\n"
    "file key {file}\n"
    "request header Authorization: Bearer zzz\n"
    "api_keys: ****9876\n"
    "listening on http://user:pw@127.0.0.1:8080/v1\n"
    "slot 0 is processing [task 1]"
)


def _tail() -> str:
    return TAIL.format(config=CONFIG_KEY, peer=PEER_KEY, file=FILE_KEY)


def test_every_secret_form_a_tail_can_carry_is_masked_and_the_rest_is_unchanged():
    out = redact_for_channel(_tail(), _harness(), SecretStr(FILE_KEY))

    for planted in (CONFIG_KEY, PEER_KEY, FILE_KEY, "zzz", "9876", "user:pw"):
        assert planted not in out, planted
    assert SECRET_MASK in out
    assert out == (
        f"srv  load: using key {SECRET_MASK}\n"
        f"peer key was {SECRET_MASK}\n"
        f"file key {SECRET_MASK}\n"
        f"request header Authorization: Bearer {SECRET_MASK}\n"
        f"api_keys: {SECRET_MASK}\n"
        "listening on http://127.0.0.1:8080/v1\n"
        "slot 0 is processing [task 1]")


@pytest.mark.parametrize("key_file_key", [None, FILE_KEY, SecretStr(FILE_KEY), ""])
def test_the_key_file_key_may_be_a_secret_str_a_str_or_absent(key_file_key):
    out = redact_for_channel(f"key {FILE_KEY} and {CONFIG_KEY}", _harness(), key_file_key)
    assert CONFIG_KEY not in out
    assert (FILE_KEY in out) == (key_file_key in (None, ""))


def test_the_literal_none_is_never_masked():
    harness = _harness(provider_key="none", peer_key=None)
    text = "none of the slots is busy; Bearer abc"
    assert redact_for_channel(text, harness) == f"none of the slots is busy; Bearer {SECRET_MASK}"


def test_a_harness_that_cannot_be_dumped_still_gets_the_key_and_the_patterns():
    class Broken:
        def model_dump(self, **kw):
            raise RuntimeError("cannot dump")

    out = redact_for_channel(f"{FILE_KEY} Bearer abc api_keys: ****1234 http://u:p@h/", Broken(),
                             SecretStr(FILE_KEY))
    assert out == f"{SECRET_MASK} Bearer {SECRET_MASK} api_keys: {SECRET_MASK} http://h/"


def test_a_stand_in_without_model_dump_does_not_raise():
    assert redact_for_channel("Bearer abc", object()) == f"Bearer {SECRET_MASK}"


def test_text_already_scrubbed_at_the_source_passes_through_unchanged():
    scrubbed = (f"Bearer {SECRET_MASK}\napi_keys: {SECRET_MASK}\nllama.cpp exited during startup.\n"
                "(see log /home/u/.localharness/serve.log, ref 1a2b3c4d)")
    assert redact_for_channel(scrubbed, _harness()) == scrubbed


def test_a_config_secret_the_source_scrub_missed_is_still_masked():
    text = f"Bearer {SECRET_MASK}\nlate line with {CONFIG_KEY}\n(see log /x/serve.log, ref 1a2b3c4d)"
    out = redact_for_channel(text, _harness())
    assert CONFIG_KEY not in out
    assert out.endswith("(see log /x/serve.log, ref 1a2b3c4d)")


def test_text_with_nothing_secret_in_it_is_returned_as_it_was():
    text = "llama.cpp exited during startup. Log tail:\nerror: failed to load model '/m/a.gguf'"
    assert redact_for_channel(text, _harness()) == text


def test_the_channel_redaction_uses_the_logged_key_patterns():
    """The patterns are the shared ones (config/redact.py), so a form added there reaches channels."""
    from localharness.config import redact
    assert model_ops.redact is redact
    assert len(redact.LOGGED_KEY_PATTERNS) >= 2
