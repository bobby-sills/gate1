"""
backend.py -- THE ONLY FILE CLAUDE CODE SHOULD EDIT.

Four functions. Everything else in this repo depends on them behaving exactly as the
docstrings state. Do not change any signature. Do not add parameters. If a signature
seems wrong for your inference stack, stop and ask -- the callers encode experimental
invariants that a signature change would silently break.

Implement in this order, committing after each:
    1. load_pool
    2. sample_closed_book + deterministic_features
    3. render_prompts        (parity check -- human must read the output)
    4. next_token_distributions
"""

import ast
import difflib
import hashlib
import json
import os
import sys
from dataclasses import dataclass

import numpy as np

import gate1


# ======================================================================================
# 1. DATA LOADING
# ======================================================================================

@dataclass
class RawInstance:
    """One question with both context conditions. Field names are fixed."""
    qid: str                       # stable unique id
    subject_qid: str               # Wikidata QID of the SUBJECT entity (e.g. "Q60")
    question: str                  # question text, no context
    gold_aliases: list[str]        # factual answer + all aliases, non-empty
    cf_aliases: list[str]          # counterfactual answer + aliases, non-empty
    factual_context: str           # passage supporting the factual answer (FiQA)
    counterfactual_context: str    # passage supporting the counterfactual (ConFiQA)


# --------------------------------------------------------------------------------------
# Loader notes -- read these before changing anything here.
#
# NO FiQA FILE EXISTS. github.com/byronBBL/Context-DPO ships only ConFiQA-{QA,MR,MC}.json.
# There is no separate factual-context dataset to join against. ConFiQA rows carry BOTH
# conditions natively (orig_context / cf_context), which is what PROTOCOL 1.2 means by
# "the document-correctness axis is inherited rather than constructed". So there is no
# join step: factual_context comes from orig_context of the same row. `fiqa_path` is
# accepted to keep the signature and the documented command line intact, and is otherwise
# unused; the loader says so loudly so nobody believes a join happened.
#
# NO question id EXISTS EITHER. `qid` is synthesized as a hash of
# (question, orig_triple, cf_triple). It is deterministic across runs and processes,
# which resume in `labels` and `decode` depends on.
#
# DOCUMENT = full context, not context_piece. orig_context / cf_context are the
# multi-sentence paragraphs (~518 chars mean), matching Context-DPO's own evaluation
# setting. Known cost: some cf_context paragraphs still mention the factual answer
# elsewhere in the passage, which makes the resistance cell easier for every arm alike.
# Reported by load_pool as `cf_context_mentions_gold` -- a data property, not a filter.
# --------------------------------------------------------------------------------------

QID_LEN = 16


def _aliases(row: dict, which: str) -> list[str]:
    """Answer plus its alias list. ConFiQA leaves `*_alias` empty for ~21% of rows, so
    the answer string itself must always lead the list."""
    raw = [row.get(f"{which}_answer")] + list(row.get(f"{which}_alias") or [])
    seen, out = set(), []
    for a in raw:
        if isinstance(a, str) and a.strip() and a not in seen:
            seen.add(a)
            out.append(a.strip())
    return out


def _subject_qid(row: dict) -> str:
    """First element of orig_triple, e.g. "('Q29021224', 'P86', 'Q608628')" -> Q29021224."""
    try:
        triple = ast.literal_eval(row["orig_triple"])
    except (ValueError, SyntaxError, KeyError, TypeError):
        return ""
    if not isinstance(triple, (list, tuple)) or not triple:
        return ""
    subj = str(triple[0]).strip()
    return subj if subj.startswith("Q") and subj[1:].isdigit() else ""


def _make_qid(row: dict) -> str:
    payload = "||".join(str(row.get(k, "")) for k in ("question", "orig_triple", "cf_triple"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:QID_LEN]


def _alias_collision(gold: list[str], cf: list[str]) -> bool:
    """Filter 2. Substring in EITHER direction after gate1.normalize, not equality.

    The matcher used for labels (Phase 2) and scoring (Phase 5) is substring-based, so a
    cf alias that merely *contains* or is *contained in* a gold alias -- "America" vs
    "United States of America" -- makes alias_match fire on the wrong answer and silently
    scores a resistance failure as a success. The filter has to be exactly as loose as the
    matcher or invariant #3 does not hold.
    """
    g = {gate1.normalize(a) for a in gold}
    c = {gate1.normalize(a) for a in cf}
    g.discard("")
    c.discard("")
    return any(x in y or y in x for x in g for y in c)


def _context_diff_is_entity_only(orig_ctx: str, cf_ctx: str,
                                 gold: list[str], cf: list[str]) -> bool:
    """Filter 3. The two passages must differ ONLY where the target entity was swapped.

    Whitespace-token diff; every non-equal span must read as the gold answer on the
    factual side and as the counterfactual answer on the counterfactual side. Anything
    else means the generator rewrote more than the entity, and the document-correctness
    axis is no longer the only thing that changed between the two conditions.

    Strict on purpose. It costs rows where the entity is welded into a compound
    ("States-based" / "Kingdom-based"), and it catches genuinely broken rows -- one cf
    passage in ConFiQA-QA degenerates into "African-American" repeated ~200 times.
    """
    a, b = orig_ctx.split(), cf_ctx.split()
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    spans = [(" ".join(a[i1:i2]), " ".join(b[j1:j2]))
             for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"]
    if not spans:
        return False          # identical passages: no counterfactual at all
    return all(gate1.alias_match(o, gold) and gate1.alias_match(n, cf) for o, n in spans)


def load_pool(confiqa_qa_path: str, fiqa_path: str) -> tuple[list[RawInstance], dict]:
    """Load ConFiQA-QA and FiQA, join on question id, filter.

    ConFiQA ships: orig_triple, cf_triple, orig_answer, cf_answer, orig_alias,
    cf_alias, orig_context_piece, cf_context_piece, question.
    FiQA has the same questions with factual contexts.

    USE THE QA SUBSET ONLY. MR and MC are multi-hop; "does the model know this fact"
    is not well-posed when the answer requires composing several facts.

    Filters, applied in this order:
      1. drop rows missing subject_qid
      2. drop rows where any cf_alias normalizes to any gold_alias (an alias collision
         silently converts a resistance instance into an agreement instance)
      3. drop rows where the two contexts differ in more than the target entity

    Use gate1.normalize() for filter 2 so the matcher and the filter agree.

    Returns:
        (instances, attrition) where attrition is an ordered dict of
        {"loaded": N, "after_filter_1": N, ...} for the writeup.

    VERIFY BEFORE MOVING ON: print the attrition chain and 3 full instances. A human
    reads them. Do not proceed on a silent pass.
    """
    if "confiqa" not in confiqa_qa_path.lower():
        print(f"NOTE: {confiqa_qa_path} does not look like a ConFiQA file.", file=sys.stderr)
    if "mr" in confiqa_qa_path.lower().replace("confiqa", "") or \
       "mc" in confiqa_qa_path.lower().replace("confiqa", ""):
        raise ValueError(f"{confiqa_qa_path} looks like the MR or MC subset. "
                         "PROTOCOL 1.1: QA subset only -- MR and MC are multi-hop and "
                         "the knowledge premise does not hold there.")

    print("=" * 78, file=sys.stderr)
    print("NO JOIN WAS PERFORMED. Context-DPO ships no FiQA file; ConFiQA-QA rows carry "
          "both\nconditions natively, so factual_context comes from `orig_context` of the "
          f"same row.\n`fiqa_path`={fiqa_path!r} is unused.", file=sys.stderr)
    print("=" * 78, file=sys.stderr)

    with open(confiqa_qa_path) as f:
        rows = json.load(f)

    attrition = {
        "loaded": len(rows),
        "fiqa_join": "not performed -- no FiQA file exists in Context-DPO; "
                     "factual_context taken from ConFiQA `orig_context`",
        "document_field": "orig_context / cf_context (full passage, not context_piece)",
    }

    # ---- filter 1: subject_qid present (plus basic well-formedness) -------------------
    stage1, drop_no_qid, drop_malformed = [], 0, 0
    for r in rows:
        subj = _subject_qid(r)
        if not subj:
            drop_no_qid += 1
            continue
        gold, cf = _aliases(r, "orig"), _aliases(r, "cf")
        if not (r.get("question") and gold and cf
                and r.get("orig_context") and r.get("cf_context")):
            drop_malformed += 1
            continue
        stage1.append((r, subj, gold, cf))
    attrition["dropped_missing_subject_qid"] = drop_no_qid
    attrition["dropped_malformed"] = drop_malformed
    attrition["after_filter_1"] = len(stage1)

    # ---- filter 2: gold/counterfactual alias collision --------------------------------
    stage2 = [t for t in stage1 if not _alias_collision(t[2], t[3])]
    attrition["dropped_alias_collision"] = len(stage1) - len(stage2)
    attrition["after_filter_2"] = len(stage2)

    # ---- filter 3: contexts differ only by the target entity --------------------------
    stage3 = [t for t in stage2
              if _context_diff_is_entity_only(t[0]["orig_context"], t[0]["cf_context"],
                                              t[2], t[3])]
    attrition["dropped_context_diff"] = len(stage2) - len(stage3)
    attrition["after_filter_3"] = len(stage3)

    # ---- build, dropping any qid hash collision ---------------------------------------
    instances, seen = [], set()
    dup = 0
    for r, subj, gold, cf in stage3:
        qid = _make_qid(r)
        if qid in seen:
            dup += 1
            continue
        seen.add(qid)
        instances.append(RawInstance(
            qid=qid,
            subject_qid=subj,
            question=r["question"].strip(),
            gold_aliases=gold,
            cf_aliases=cf,
            factual_context=r["orig_context"].strip(),
            counterfactual_context=r["cf_context"].strip(),
        ))
    attrition["dropped_duplicate_qid"] = dup
    attrition["final"] = len(instances)

    # ---- reported properties, not filters ---------------------------------------------
    attrition["unique_subject_qids"] = len({i.subject_qid for i in instances})
    attrition["cf_context_mentions_gold"] = sum(
        gate1.alias_match(i.counterfactual_context, i.gold_aliases) for i in instances)
    return instances, attrition


# ======================================================================================
# MODEL PLUMBING  (shared by sections 2, 3 and 4)
# ======================================================================================

# The system prompt is IDENTICAL for both passes. The document is the only thing that
# ever differs between them -- see render_prompts and the parity artifact.
SYSTEM_PROMPT = ("You are a helpful assistant. Answer the question with the short "
                 "factual answer only. Do not explain.")

# Llama-3.1's chat template stamps "Today Date: <date>" into the system block, taking
# the value from `date_string` (default: today). Left to default, the rendered prompt
# changes between the `labels` run and the `decode` run, and again after midnight --
# so cached generations would no longer correspond to the prompt that produced them.
# Freezing it makes every pass reproducible. It is identical in both passes either way,
# so parity is unaffected; this is about resumability, not about parity.
DATE_STRING = "26 Jul 2024"          # the template's own default, pinned

TEMPERATURE = 0.7                    # PROTOCOL 2.1
MAX_NEW_TOKENS = 32                  # PROTOCOL 2.1

_MODEL = None
_TOK = None


def _load():
    """Lazy singleton. Imported at call time so `pool`, `sanity`, `cells`, `sweep` and
    `analyze` -- all CPU-only phases -- never touch torch."""
    global _MODEL, _TOK
    if _MODEL is not None:
        return _TOK, _MODEL

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = gate1.MODEL                       # pre-registered in gate1.py, not tunable
    _TOK = AutoTokenizer.from_pretrained(name)

    kwargs = {"dtype": torch.bfloat16, "device_map": "auto"}
    if os.environ.get("GATE1_LOAD_8BIT") == "1":
        # T4 fallback documented in COLAB.md. Slower, and the logits differ slightly
        # from bf16 -- do not mix 8-bit and bf16 units within one experiment.
        from transformers import BitsAndBytesConfig
        kwargs = {"quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                  "device_map": "auto"}
        print("GATE1_LOAD_8BIT=1: loading in 8-bit.", file=sys.stderr)
    try:
        _MODEL = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    except TypeError:                        # transformers < 4.56 spells it torch_dtype
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        _MODEL = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    _MODEL.eval()
    return _TOK, _MODEL


def _build_prompt(question: str, context: str | None) -> str:
    """THE single prompt builder. Every pass in this file goes through it, so the parity
    artifact in section 3 describes what sections 2 and 4 actually ran -- a parity check
    against a prompt nothing else uses would be worthless.

    context=None -> closed book. Otherwise the document block is prepended to the user
    turn and is the ONLY difference between the two renderings.
    """
    tok, _ = _load()
    user = f"Question: {question}" if context is None else \
           f"Document:\n{context}\n\nQuestion: {question}"
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                   date_string=DATE_STRING)


def _encode(prompt: str):
    """apply_chat_template already emits <|begin_of_text|>; add_special_tokens=False
    stops the tokenizer adding a second BOS."""
    tok, model = _load()
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    return ids.to(model.device)


def _question_seed(question: str, seed: int) -> int:
    """Stable across processes. Python's hash() is salted per interpreter, so using it
    here would silently break resume -- a re-run would draw different samples."""
    h = hashlib.sha256(question.encode("utf-8")).hexdigest()[:8]
    return (seed ^ int(h, 16)) % (2**31 - 1)


# ======================================================================================
# 2. GENERATION AND FEATURES
# ======================================================================================

def sample_closed_book(question: str, n: int, seed: int) -> list[str]:
    """n independent generations, NO document, temperature 0.7, max_new_tokens=32.

    Must use the same chat template as decoding. `seed` makes the run reproducible;
    derive per-sample seeds from it however you like, but the same (question, seed)
    must give the same list on a re-run, because resume depends on it.

    Returns: n answer strings, prompt stripped.
    """
    import torch

    tok, model = _load()
    ids = _encode(_build_prompt(question, None))          # None -> no document
    torch.manual_seed(_question_seed(question, seed))

    with torch.inference_mode():
        out = model.generate(
            ids,
            do_sample=True,
            temperature=TEMPERATURE,
            # Llama-3.1 ships generation_config with temperature=0.6, top_p=0.9. Those
            # would silently apply on top of the pre-registered temperature, so both
            # filters are disabled explicitly: PROTOCOL 2.1 registers temperature only.
            top_p=1.0,
            top_k=0,
            num_return_sequences=n,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tok.eos_token_id,
        )
    return [tok.decode(seq[ids.shape[1]:], skip_special_tokens=True).strip()
            for seq in out]


def deterministic_features(question: str) -> tuple[float, float]:
    """Greedy, NO document, NO sampling. Returns (entropy, max_prob) of the next-token
    distribution at the FIRST answer position.

    Entropy in nats over the full vocabulary.

    THIS MUST BE A SEPARATE FORWARD PASS from sample_closed_book. Do not derive these
    numbers from the sampled generations, and do not fold the two into one call to save
    time. The knowledge label comes from sampling, which is itself a confidence
    measurement; sharing computation between label and predictor handicaps the entropy
    baseline and inflates the headline result. This is the single most important
    instruction in this file.
    """
    import torch

    _, model = _load()
    # Its own forward pass, on its own freshly-encoded prompt, with no cache shared with
    # sample_closed_book and nothing carried over from it. Deliberately redundant compute.
    ids = _encode(_build_prompt(question, None))
    with torch.inference_mode():
        logits = model(ids).logits[0, -1, :].float()      # first answer position

    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    entropy = float(-(p * logp).sum())                    # nats, full vocabulary
    return entropy, float(p.max())


# ======================================================================================
# 3. PROMPT PARITY (Llama-3.1 specific -- do not skip)
# ======================================================================================

def render_prompts(question: str, context: str) -> tuple[str, str]:
    """Return the two fully-rendered prompt strings, exactly as the model sees them:
        (closed_book_prompt, with_context_prompt)

    Llama-3.1's chat template injects a date-cutoff line into the system prompt in some
    transformers versions. If the two passes end up with different system prompts, then
    p_theta and p_ctx are not the same distribution over the same conditioning, and every
    contrastive subtraction downstream measures template drift alongside document effect.

    The two strings must differ ONLY by the document block.

    Do NOT write an automated equality assertion and report "parity OK". Write both
    strings to outputs/prompt_parity.txt for 10 instances and tell the human to read
    them. Whitespace-insensitive diffs hide exactly the failure this check exists for.
    """
    raise NotImplementedError


# ======================================================================================
# 4. TWO-PASS LOGITS  (the decoding primitive)
# ======================================================================================

def next_token_distributions(
    question: str,
    context: str,
    generated_prefix: list[int],
) -> tuple["np.ndarray", "np.ndarray"]:
    """The core primitive. Two forward passes, same generated prefix.

    Returns (log_p_theta, log_p_ctx), each a float32 array of shape [vocab_size]:
        log_p_theta = log P(. | question, generated_prefix)              no document
        log_p_ctx   = log P(. | question, context, generated_prefix)     with document

    LOG probabilities, normalized (logsumexp == 0), NOT raw logits. gate1 combines them
    as (1 - tau) * log_p_theta + tau * log_p_ctx, which is only correct for normalized
    log-probs.

    Both passes use the same chat template and the same generated prefix; they differ
    only by the document block, per render_prompts.

    PERFORMANCE: gate1 calls this once per (instance, prefix) and caches on the prefix,
    so tau costs nothing extra. Keep KV caches for both passes across calls within one
    instance's decode -- that is where the real speedup is. Do not add caching logic
    inside gate1; it already has a cache and two layers would diverge.

    INSTRUMENT FIRST: before any full sweep, run 100 instances and report wall-clock
    per generated token and cache hit rate. The compute estimates in the protocol are
    arithmetic, not measurements, and have been wrong before. If throughput implies more
    than ~3 GPU-hours for Phase 4, stop and report rather than shrinking the grids.
    """
    raise NotImplementedError
