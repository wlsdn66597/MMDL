"""Round 3 only. Matched one-call ERA SFT with explicit structure, evidence and EOS supervision."""
import argparse
from pathlib import Path
import re

import train_round as base


TITLES = ("Evidence:", "Rule:", "Application:", "Final answer:")
TITLE_BONUS = 3.0
END_BONUS = 3.0
BASE_SCRIPT = "a7a2b819fec2208faf91e0f905f83e5e13964d958e61302a13802c86c35b80c1"
ordinary_weights = base.focus_weights


def title_spans(text):
    """Only author-controlled line-leading section labels count as structural targets."""
    matches = list(re.finditer(r"(?m)^(Evidence:|Rule:|Application:|Final answer:)(?= )", text))
    if tuple(m.group() for m in matches) != TITLES:
        raise ValueError("Expected exactly four ERA labels in target order")
    return [m.span() for m in matches]


def structural_weights(feature, report, batch, processor, hp, arm):
    positions, weights = ordinary_weights(feature, report, batch, processor, hp, arm)
    if arm == "control":
        return positions, weights
    text = feature["response"]
    spans = title_spans(text)
    encoded = processor.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    assert len(positions) == len(encoded.offset_mapping) + 1
    assert batch["labels"][0, positions[-1]].item() == report["assistant_end_token_id"]
    seen = set()
    for i, (a, b) in enumerate(encoded.offset_mapping):
        hits = {j for j, (start, end) in enumerate(spans) if a < end and b > start}
        if hits:
            weights[i] += hp["title_bonus"]
            seen.update(hits)
    assert len(seen) == len(spans)
    weights[-1] += hp["end_bonus"]
    return positions, weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", required=True, type=int, choices=(3,))
    parser.add_argument("--arm", required=True, choices=("control","candidate"))
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    if base.sha(Path(base.__file__)) != BASE_SCRIPT:
        raise RuntimeError("Frozen common training implementation has changed")
    hp = dict(base.ROUNDS[3], name="structure_and_links",
              title_bonus=TITLE_BONUS, end_bonus=END_BONUS,
              wrapper_sha256=base.sha(Path(__file__)))
    base.ROUNDS[3] = hp
    base.focus_weights = structural_weights
    # train_round's merge path normally authenticates the saved adapter; also bind this
    # branch to the training recipe that produced it before reusing or merging any state.
    folder = base.ROOT/"models"/"r3"/args.arm
    if (folder/"identity.json").exists() and base.read(folder/"identity.json")["hyperparameters"] != hp:
        raise RuntimeError("Existing round 3 state belongs to a different recipe")
    if args.merge and base.read(folder/"manifest.json")["identity"]["hyperparameters"] != hp:
        raise RuntimeError("Cannot merge a round 3 adapter from a different recipe")
    base.run(args)


if __name__ == "__main__":
    main()
