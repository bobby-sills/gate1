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
import re
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
    relation: str                  # ConFiQA predicate, e.g. "composer" -- see FUNCTIONAL


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

# Minimum NORMALIZED length for an alias to be kept. Measured after gate1.normalize,
# not on the raw string: "U.S." is 4 raw characters but normalizes to "us", and it is
# the normalized form that gate1.alias_match substring-tests. See _clean_aliases.
MIN_ALIAS_CHARS = 3


# --------------------------------------------------------------------------------------
# Relation classification -- filters 5 and 6, added after the Phase 2 label read.
#
# FUNCTIONAL: one true object per subject. ONE_TO_MANY: the subject genuinely has
# several and ConFiQA picked one, so "does the model know this fact" is not well posed
# -- the same well-posedness argument PROTOCOL 1.1 uses to exclude the multi-hop
# subsets. This is not an assumption: measured on the full 4207-row pool, ONE_TO_MANY
# rows are labelled `unknown` 63% of the time against 34% for FUNCTIONAL, with an
# identical majority-class share (24%), which is what a label measuring "did the model
# name the particular one ConFiQA picked" looks like.
#
# TEMPORAL: functional at any instant, but the holder changes over time, so a model
# that knows a different-era holder scores wrong. Excluded from Gate 1 and tagged, not
# discarded -- these are the legitimate-update case for a possible Gate 2.
#
# DROP: structurally defective in ConFiQA. See the note on each.
# --------------------------------------------------------------------------------------
FUNCTIONAL = (
    "spouse", "composer", "country of citizenship", "director", "sport", "religion",
    "native language", "performer", "official language", "capital", "continent",
    "language of work or name", "headquarters location", "currency", "author",
)
ONE_TO_MANY = (
    "award received", "genre", "member of", "field of work", "member of sports team",
    "employer", "record label", "ethnic group", "work location", "position played",
    "position held", "developer", "owned by", "founded by",
    "country of origin",   # co-productions; Wikidata lists all of them
    "creator",             # multiple credited creators
)
TEMPORAL = (
    "head of state", "head of government", "chairperson", "head coach",
    "chief executive officer",
)
DROP_RELATIONS = {
    "follows": "54% of rows name the gold answer in the question -- the template renders "
               "(X, follows, Y) in both directions inconsistently",
    "country": "43% tautologous -- 'What country is Panama?' -> 'Panama'",
    "location": "too vague to be well posed",
    "capital of": "4 of 5 rows self-answering",
}

RELATION_CLASS = {}
for _r in FUNCTIONAL:
    RELATION_CLASS[_r] = "FUNCTIONAL"
for _r in ONE_TO_MANY:
    RELATION_CLASS[_r] = "ONE_TO_MANY"
for _r in TEMPORAL:
    RELATION_CLASS[_r] = "TEMPORAL"
for _r in DROP_RELATIONS:
    RELATION_CLASS[_r] = "DROP"


def _relation(row: dict) -> str:
    """orig_path_labeled is "[('Bad Boys for Life', 'composer', 'Lorne Balfe')]"."""
    try:
        return str(ast.literal_eval(row["orig_path_labeled"])[0][1])
    except (ValueError, SyntaxError, IndexError, KeyError, TypeError):
        return "?"


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


def _clean_aliases(aliases: list[str]) -> list[str]:
    """Drop alias entries that make gate1.alias_match fire on arbitrary text.

    alias_match tests `normalize(alias) in normalize(prediction)`, so two classes of
    entry shipped by ConFiQA are pathological:

      A. entries that normalize to "" -- emoji flags and currency symbols ('🇯🇵', '£',
         '$', '⚾'). `"" in anything` is True, so a single such alias makes EVERY
         generation score correct: n_correct=10, label `known` regardless of the model,
         and accuracy 1.0 in every cell for every arm. 165 rows in the pool.
      B. very short entries -- 'US', 'fr', 'ru', 'ja', 'or' (Odia), 'SS'. These are
         substrings of ordinary English prose. 294 rows.

    Together they scored a non-answer as correct in 10.3% of the pool.

    The primary answer (index 0) is never dropped, so the list stays non-empty. Rows
    whose primary answer is itself degenerate are removed by filter 1 instead -- an
    unscoreable answer is unusable in either direction, not repairable by keeping it.
    """
    return [aliases[0]] + [a for a in aliases[1:]
                           if len(gate1.normalize(a)) >= MIN_ALIAS_CHARS]


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


def _answer_leaks(context: str, aliases: list[str]) -> bool:
    """Filter 4. Does this passage contain a complete alias of the OTHER condition's
    answer, i.e. can a model score correct by copying the document it should be
    resisting (or contradicting the document it should be following)?

    POST-HOC FILTER -- not one of the three in the docstring above. Added after the
    Phase 1 checkpoint, on evidence: ConFiQA swapped only the full-name mention when it
    built the counterfactual, so passages routinely retain material supporting the other
    answer. Symmetric by design: gold surviving in cf_context corrupts `resistance`,
    cf surviving in factual_context corrupts `agreement`/`correction`.

    Fragment survivals ("...Balfe's score" where gold is "Lorne Balfe") are NOT filtered.
    alias_match cannot fire on them, so they cannot turn into a scored-correct answer.
    Only complete-alias survivals are dropped.
    """
    return gate1.alias_match(context, aliases)


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
      4. POST-HOC: drop rows where the other condition's answer survives in the passage
      5. POST-HOC: keep FUNCTIONAL relations only (see RELATION_CLASS)
      6. POST-HOC: drop rows whose question contains its own answer

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
    stage1, drop_no_qid, drop_malformed, drop_unscoreable = [], 0, 0, 0
    n_alias_trimmed, n_aliases_dropped = 0, 0
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
        # An answer that normalizes to fewer than MIN_ALIAS_CHARS is unscoreable: it
        # substring-matches arbitrary text, and keeping it would poison filters 2 and 4
        # as well as Phase 2 labelling. It cannot be repaired by trimming aliases.
        if min(len(gate1.normalize(gold[0])), len(gate1.normalize(cf[0]))) < MIN_ALIAS_CHARS:
            drop_unscoreable += 1
            continue
        clean_gold, clean_cf = _clean_aliases(gold), _clean_aliases(cf)
        n_aliases_dropped += (len(gold) - len(clean_gold)) + (len(cf) - len(clean_cf))
        n_alias_trimmed += (len(clean_gold) < len(gold)) or (len(clean_cf) < len(cf))
        stage1.append((r, subj, clean_gold, clean_cf))
    attrition["dropped_missing_subject_qid"] = drop_no_qid
    attrition["dropped_malformed"] = drop_malformed
    attrition["dropped_unscoreable_answer"] = drop_unscoreable
    attrition["alias_hygiene_min_normalized_chars"] = MIN_ALIAS_CHARS
    attrition["rows_with_aliases_trimmed"] = n_alias_trimmed
    attrition["alias_entries_dropped"] = n_aliases_dropped
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

    # ---- filter 4 (POST-HOC): the other condition's answer survives in the passage ----
    gold_in_cf = sum(1 for t in stage3 if _answer_leaks(t[0]["cf_context"], t[2]))
    cf_in_factual = sum(1 for t in stage3 if _answer_leaks(t[0]["orig_context"], t[3]))
    stage4 = [t for t in stage3
              if not _answer_leaks(t[0]["cf_context"], t[2])
              and not _answer_leaks(t[0]["orig_context"], t[3])]
    attrition["filter_4_status"] = ("POST-HOC, added after the Phase 1 checkpoint -- "
                                    "not one of the three pre-registered filters")
    attrition["dropped_gold_alias_in_cf_context"] = gold_in_cf
    attrition["dropped_cf_alias_in_factual_context"] = cf_in_factual
    attrition["dropped_filter_4_total"] = len(stage3) - len(stage4)
    attrition["after_filter_4"] = len(stage4)

    # ---- filter 5 (POST-HOC): keep FUNCTIONAL relations only --------------------------
    by_class = {}
    for t in stage4:
        by_class.setdefault(RELATION_CLASS.get(_relation(t[0]), "UNCLASSIFIED"), []).append(t)
    stage5 = by_class.get("FUNCTIONAL", [])
    attrition["filter_5_status"] = ("POST-HOC, added after the Phase 2 label read -- "
                                    "not one of the three pre-registered filters")
    attrition["filter_5_class_counts"] = {k: len(v) for k, v in sorted(by_class.items())}
    attrition["dropped_non_functional_relation"] = len(stage4) - len(stage5)
    attrition["after_filter_5"] = len(stage5)
    if by_class.get("UNCLASSIFIED"):
        unk = sorted({_relation(t[0]) for t in by_class["UNCLASSIFIED"]})
        print(f"WARNING: {len(unk)} relation(s) not in RELATION_CLASS, dropped as "
              f"non-FUNCTIONAL: {unk}", file=sys.stderr)

    # ---- filter 6 (POST-HOC): the question contains its own answer ---------------------
    # ONE GLOBAL RULE, deliberately the same matcher as labels and scoring (invariant #3).
    # If alias_match fires on the question text itself no knowledge is being measured.
    # 156/4207 pool-wide; concentrated in relations filter 5 already removes, but the
    # rule is applied everywhere rather than per relation.
    stage6 = [t for t in stage5 if not gate1.alias_match(t[0]["question"], t[2])]
    attrition["filter_6_status"] = "POST-HOC, one global rule -- gold alias in question"
    attrition["dropped_self_answering_question"] = len(stage5) - len(stage6)
    attrition["after_filter_6"] = len(stage6)

    # ---- build, dropping any qid hash collision ---------------------------------------
    instances, seen = [], set()
    dup = 0
    for r, subj, gold, cf in stage6:
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
            relation=_relation(r),
        ))
    attrition["dropped_duplicate_qid"] = dup
    attrition["final"] = len(instances)

    # ---- reported properties, not filters ---------------------------------------------
    attrition["unique_subject_qids"] = len({i.subject_qid for i in instances})
    # Fragment survivals that filter 4 deliberately leaves in: a >=4-char word from a
    # gold alias still present in cf_context ("...Balfe's score"). alias_match cannot
    # fire on these, so they cannot become a scored-correct answer, but they do make the
    # counterfactual passage internally inconsistent. Reported, not filtered.
    frag = 0
    for i in instances:
        words = {w for a in i.gold_aliases for w in gate1.normalize(a).split() if len(w) >= 4}
        if words & set(gate1.normalize(i.counterfactual_context).split()):
            frag += 1
    attrition["cf_context_retains_gold_fragment"] = frag
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


def _load_tok():
    """Tokenizer only. Keeps `parity` off the GPU: rendering prompts needs the chat
    template, not 16GB of weights, so the human can do the Phase 2.7 read in a CPU
    session."""
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained(gate1.MODEL)
    return _TOK


def _load():
    """Lazy singleton. Imported at call time so `pool`, `sanity`, `cells`, `sweep` and
    `analyze` -- all CPU-only phases -- never touch torch."""
    global _MODEL
    if _MODEL is not None:
        return _TOK, _MODEL

    import torch
    from transformers import AutoModelForCausalLM

    name = gate1.MODEL                       # pre-registered in gate1.py, not tunable
    _load_tok()

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
    tok = _load_tok()
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
    # Same builder as sections 2 and 4 -- these ARE the prompts those passes run, not a
    # reconstruction of them. No assertion here on purpose: gate1.phase_parity writes
    # both strings out and stops, and a human reads them.
    return _build_prompt(question, None), _build_prompt(question, context)


# ======================================================================================
# 4. TWO-PASS LOGITS  (the decoding primitive)
# ======================================================================================

class _TwoPassCache:
    """KV state for ONE (question, context) pair, both passes.

    This is the cache invariant #8 refers to. gate1.phase_decode iterates
    sorted(work), so every tau for one (qid, context_kind) arrives consecutively: the
    prompt prefill is paid once per instance and reused across the whole tau grid, and
    only the generated suffix is rolled back between them.

    gate1 does NOT cache here -- it dedups whole decode units by (qid, context_kind,
    tau). Two cache layers would diverge; this one is the only place KV state lives.
    """

    def __init__(self):
        self.key = None
        self.caches = None          # (theta, ctx) DynamicCache pair
        self.prompt_lens = None     # (theta, ctx) prompt token counts
        self.gen: list[int] = []    # generated tokens currently in the caches
        self.last = None            # (log_p_theta, log_p_ctx) at the current position
        self.prompt_last = None     # ... at the end of the prompt, i.e. prefix == []
        self.stats = {"prefills": 0, "forward_steps": 0, "calls": 0, "hits": 0,
                      "rollbacks": 0}


_CACHE = _TwoPassCache()


def _forward(ids, cache, past_len):
    """One incremental forward pass. Returns float32 normalized log-probs at the last
    position. `ids` is only the NEW tokens; `cache` holds everything before them."""
    import torch

    _, model = _load()
    total = past_len + ids.shape[1]
    mask = torch.ones((1, total), dtype=torch.long, device=model.device)
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=mask, past_key_values=cache,
                    use_cache=True)
    # log_softmax, not raw logits: gate1 forms (1-tau)*log_p_theta + tau*log_p_ctx,
    # which is only a valid unnormalized log-density if both terms are normalized.
    logp = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
    return out.past_key_values, logp.cpu().numpy().astype(np.float32)


def _reset(question: str, context: str):
    """Prefill both passes for a new (question, context). The expensive step: the
    document is ~150 tokens, the generated suffix is at most 32."""
    import torch
    from transformers import DynamicCache

    ids_theta = _encode(_build_prompt(question, None))
    ids_ctx = _encode(_build_prompt(question, context))
    c_theta, lp_theta = _forward(ids_theta, DynamicCache(), 0)
    c_ctx, lp_ctx = _forward(ids_ctx, DynamicCache(), 0)

    _CACHE.key = (question, context)
    _CACHE.caches = (c_theta, c_ctx)
    _CACHE.prompt_lens = (ids_theta.shape[1], ids_ctx.shape[1])
    _CACHE.gen = []
    _CACHE.last = (lp_theta, lp_ctx)
    # Kept for the whole instance: gate1 restarts at prefix == [] for every tau, and
    # this is the position it lands on. Without it, each restart would re-prefill.
    _CACHE.prompt_last = (lp_theta, lp_ctx)
    _CACHE.stats["prefills"] += 1
    del torch


def _rollback(n_keep: int) -> bool:
    """Crop both caches back to n_keep generated tokens. Returns False if the installed
    transformers cannot crop, in which case the caller re-prefills."""
    theta, ctx = _CACHE.caches
    if not (hasattr(theta, "crop") and hasattr(ctx, "crop")):
        return False
    pt, pc = _CACHE.prompt_lens
    theta.crop(pt + n_keep)
    ctx.crop(pc + n_keep)
    _CACHE.gen = _CACHE.gen[:n_keep]
    # `last` described the position we just cropped away. Anything that returned it now
    # would hand back a distribution from a different prefix -- silently, and only for
    # the second and later tau of each instance.
    _CACHE.last = None
    _CACHE.stats["rollbacks"] += 1
    return True


def cache_stats() -> dict:
    """Instrumentation for the HANDOFF Task 4 checkpoint. Not called by gate1."""
    return dict(_CACHE.stats)


def is_eos(token_id: int) -> bool:
    """Stop condition for gate1.greedy_decode. Llama-3.1 ends turns with <|eot_id|>
    (128009), not only <|end_of_text|>, so the generation_config list is the authority."""
    tok, model = _load()
    ids = getattr(model.generation_config, "eos_token_id", None) or tok.eos_token_id
    if isinstance(ids, int):
        ids = [ids]
    return int(token_id) in set(ids)


def detokenize(token_ids: list[int]) -> str:
    """Generated tokens -> the answer string gate1 scores with alias_match."""
    tok, _ = _load()
    return tok.decode(token_ids, skip_special_tokens=True).strip()


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
    prefix = list(generated_prefix)
    _CACHE.stats["calls"] += 1

    if _CACHE.key != (question, context):
        _reset(question, context)

    # How much of the cached suffix the request still agrees with.
    common = 0
    for a, b in zip(_CACHE.gen, prefix):
        if a != b:
            break
        common += 1

    if common == len(_CACHE.gen) == len(prefix) and _CACHE.last is not None:
        _CACHE.stats["hits"] += 1
        return _CACHE.last                      # same position as the previous call

    if not prefix:
        # Start of a decode -- gate1 lands here once per tau. The prompt-final position
        # is already known, so roll the suffix off and answer from prefill.
        if _CACHE.gen and not _rollback(0):
            _reset(question, context)
        _CACHE.stats["hits"] += 1
        return _CACHE.prompt_last

    # Keep at most len(prefix)-1 tokens, so at least one token is always fed and `last`
    # is always recomputed for the position actually being asked about. Re-feeding one
    # token is far cheaper than re-prefilling the document.
    keep = min(common, len(prefix) - 1)
    if len(_CACHE.gen) > keep and not _rollback(keep):
        _reset(question, context)               # cannot crop -> rebuild from the prompt
        keep = 0

    new = prefix[keep:]

    import torch

    _, model = _load()
    ids = torch.tensor([new], dtype=torch.long, device=model.device)
    theta, ctx = _CACHE.caches
    pt, pc = _CACHE.prompt_lens
    theta, lp_theta = _forward(ids, theta, pt + keep)
    ctx, lp_ctx = _forward(ids, ctx, pc + keep)

    _CACHE.caches = (theta, ctx)
    _CACHE.gen = prefix
    _CACHE.last = (lp_theta, lp_ctx)
    _CACHE.stats["forward_steps"] += 1
    return _CACHE.last


# ======================================================================================
# 5. GATE 2 -- RESIDUAL-STREAM EXTRACTION  (see GATE2_PROTOCOL.md Phase A)
# ======================================================================================

# Wikidata labels carry disambiguating parentheticals the question drops:
# "Shaft (2019 film)" appears as "the film Shaft (2019)". Stripping a single trailing
# parenthetical recovers 7.6% of the pool on top of the 91.5% that match verbatim.
_TRAILING_PAREN = re.compile(r"\s*\([^()]*\)\s*$")


def _subject_char_span(prompt: str, question: str, subject: str):
    """Character span of `subject` inside the QUESTION LINE of `prompt`.

    Scoped to the question line on purpose: the subject string can also occur in the
    system block or, in the with-document rendering, in the passage. Position (2) is
    defined as the entity mention the question is asking about, so an earlier incidental
    occurrence is the wrong token.

    Returns (start, end, tier) or (None, None, "unresolved"). `tier` is recorded per
    question -- GATE2_PROTOCOL Phase A reports the mix, because a sudden shift toward the
    looser tiers is how a tokenizer or template change would show up.
    """
    base = prompt.rfind(question)
    if base < 0 or not subject:
        return None, None, "unresolved"
    line = prompt[base:base + len(question)]

    stripped = _TRAILING_PAREN.sub("", subject).strip()
    for cand, tier in ((subject, "verbatim"), (stripped, "paren-stripped")):
        if cand and (i := line.find(cand)) >= 0:
            return base + i, base + i + len(cand), tier
    if stripped and (i := line.lower().find(stripped.lower())) >= 0:
        return base + i, base + i + len(stripped), "casefold"
    return None, None, "unresolved"


def hidden_states(question: str, subject: str | None = None) -> dict:
    """Context-free forward pass capturing the residual stream at EVERY layer.

    Returns
        h_final    float16 [n_layers, hidden] -- final question token, pre-generation
        h_subject  float16 [n_layers, hidden] or None -- last token of the subject span
        entropy    float   next-token entropy in nats, full vocabulary
        max_prob   float   next-token max probability
        span_tier  str     how the subject span was located; "unresolved" -> h_subject None
        n_tokens   int     prompt length in tokens

    THIS IS A THIRD FORWARD PASS. It does not reuse and must not be merged into
    sample_closed_book or deterministic_features. Gate 1 invariant #2 forbids merging the
    label pass with the predictor pass because sharing computation between them handicaps
    the entropy baseline; this pass runs after both are on disk and shares computation
    with neither, so it cannot move a label or a baseline. See GATE2_PROTOCOL.md Phase A.

    `entropy` and `max_prob` come out of this pass for free and are NOT used downstream as
    features -- gate2 reads those from labels.jsonl. They are returned so gate2 can check
    them against labels.jsonl, which is a byte-level proof that the extraction prompt is
    the Phase 2 prompt. That check is stronger than re-reading prompt_parity.txt, and it
    costs one softmax.
    """
    import torch

    tok, model = _load()
    prompt = _build_prompt(question, None)              # None -> closed book, as Phase 2
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False,
              return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    ids = enc.input_ids.to(model.device)

    with torch.inference_mode():
        out = model(ids, output_hidden_states=True)

    logits = out.logits[0, -1, :].float()
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()

    # hidden_states is a tuple of length n_layers+1: embeddings, then one per block.
    # The embedding layer is kept -- it is the null control for the C4 depth curve, and a
    # probe that reads at layer 0 is reading the tokenizer, not the model.
    hs = torch.stack([h[0] for h in out.hidden_states])  # [n_layers, seq, hidden]

    start, end, tier = (None, None, "no-subject")
    if subject is not None:
        start, end, tier = _subject_char_span(prompt, question, subject)

    h_subject = None
    if start is not None:
        # LAST token overlapping the span. Llama's BPE splits entity names across several
        # tokens and the entity representation is assembled at the final one.
        hit = [i for i, (a, b) in enumerate(offsets) if a < end and b > start]
        if hit:
            h_subject = hs[:, hit[-1], :].float().cpu().numpy().astype("float16")
        else:
            tier = "unresolved"                          # span fell in a zero-width offset

    return {
        "h_final": hs[:, -1, :].float().cpu().numpy().astype("float16"),
        "h_subject": h_subject,
        "entropy": float(-(p * logp).sum()),
        "max_prob": float(p.max()),
        "span_tier": tier,
        "n_tokens": int(ids.shape[1]),
    }


# ======================================================================================
# 6. GATE 2b -- ANSWER-TOKEN STATES  (see GATE2B_PROTOCOL.md Phase A)
# ======================================================================================

def _entropies(logits):
    """Row-wise entropy in nats over the full vocabulary, and the row-wise max prob."""
    import torch
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    return (-(p * logp).sum(dim=-1), p.max(dim=-1).values, logp)


def answer_states(question: str, max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    """Stage 1 of the Gate 2b mechanism: greedy-complete the closed-book answer, then read
    the residual stream at the ANSWER tokens.

    Returns
        h_first / h_last / h_mean   float16 [n_layers, hidden], or None if no answer span
        answer                      str    the decoded answer, specials stripped
        n_answer_tokens             int    m
        entropy / max_prob          float  at the final QUESTION token -- the parity pair
        first_token_is_argmax       bool   see below
        b2_first / b2_last / b2_mean  float  H(p_theta) at each read position
        b3_mean_logprob             float  (1/m) sum_j log p(a_j)
        n_tokens                    int    prompt length

    THIS IS A FOURTH FORWARD PASS and it must not be merged into any of the others. Gate 1
    invariant #2 forbids sharing computation between the label pass and the predictor pass;
    this one runs after both are on disk and shares computation with neither, exactly as
    `hidden_states` does. The ten Phase 2 samples that Gate 2b's b4 needs are obtained by a
    SEPARATE call to `sample_closed_book`, not from here.

    Two structural points, both registered in GATE2B_PROTOCOL Phase A:

    The second pass is TEACHER-FORCED over [prompt + answer] rather than harvested from
    `generate`. `generate` computes a hidden state at a position only when it feeds that
    position back in, so when the answer ends in EOS the state at the last answer token is
    never computed -- p2 would be silently undefined for exactly the questions where the
    model finished cleanly. One extra pass of length n+m removes that whole class of bug.

    `first_token_is_argmax` checks that the greedy answer's first token is the argmax of
    the distribution `deterministic_features` measured at the final question token. That is
    the sense in which stage 1 "continues" that pass, and it is the assertion that would
    catch a generation_config override silently perturbing the decode.
    """
    import torch

    tok, model = _load()
    prompt = _build_prompt(question, None)              # None -> closed book, as Phase 2
    ids = _encode(prompt)
    n = ids.shape[1]

    # ---- stage 1: greedy completion --------------------------------------------------
    # temperature/top_p from Llama-3.1's shipped generation_config are inert under
    # do_sample=False (transformers builds the warpers only in sample mode), unlike in
    # sample_closed_book where they had to be disabled explicitly.
    with torch.inference_mode():
        gen = model.generate(ids, do_sample=False, max_new_tokens=max_new_tokens,
                             pad_token_id=tok.eos_token_id)

    # Truncate at the first special token rather than filtering them out: a special in the
    # middle ends the answer, and keeping what follows would splice a second turn onto it.
    specials = set(tok.all_special_ids)
    core = []
    for t in gen[0, n:].tolist():
        if t in specials:
            break
        core.append(t)
    m = len(core)

    if m == 0:
        # No usable answer span. Dropped from Gate 2b entirely and counted -- NOT
        # backfilled with the question-final token, which would quietly turn the row into
        # a Gate 2 row and destroy the position comparison. GATE2B_PROTOCOL invariant 4.
        with torch.inference_mode():
            logits = model(ids).logits[0, -1:, :].float()
        ent, mx, _ = _entropies(logits)
        return {"h_first": None, "h_last": None, "h_mean": None, "answer": "",
                "n_answer_tokens": 0, "entropy": float(ent[0]), "max_prob": float(mx[0]),
                "first_token_is_argmax": False, "b2_first": float("nan"),
                "b2_last": float("nan"), "b2_mean": float("nan"),
                "b3_mean_logprob": float("nan"), "n_tokens": int(n)}

    # ---- stage 2: one teacher-forced pass over prompt + answer ------------------------
    full = torch.cat([ids, torch.tensor([core], device=ids.device, dtype=ids.dtype)], dim=1)
    with torch.inference_mode():
        out = model(full, output_hidden_states=True)

    # hidden_states is length n_layers+1: embeddings, then one per block. The embedding
    # layer is kept as the null control for the depth curve, as in `hidden_states`.
    hs = torch.stack([h[0] for h in out.hidden_states])   # [n_layers, n+m, hidden]

    # Positions, 0-based into `full`: a_j sits at n+j-1. The distribution produced AT a
    # position is over the NEXT token, so the slice below starts one position early.
    #   slice index 0   -> at t_n,     distribution over a_1   (this is b1 / Phase 2)
    #   slice index j   -> at a_j,     distribution over a_{j+1}
    #   slice index m   -> at a_m,     distribution over whatever follows the answer
    logits = out.logits[0, n - 1:n + m, :].float()        # [m+1, V]
    ent, mx, logp = _entropies(logits)

    tgt = torch.tensor(core, device=logits.device)
    tok_logp = logp[:m].gather(1, tgt.unsqueeze(1)).squeeze(1)   # log p(a_j), j=1..m

    def h(sl):
        return sl.float().cpu().numpy().astype("float16")

    return {
        "h_first": h(hs[:, n, :]),
        "h_last": h(hs[:, n + m - 1, :]),
        "h_mean": h(hs[:, n:n + m, :].mean(dim=1)),
        "answer": tok.decode(core, skip_special_tokens=True).strip(),
        "n_answer_tokens": int(m),
        # The parity pair: recomputed at the final QUESTION token, so it must reproduce
        # what labels.jsonl holds. b1 itself is read from labels.jsonl, not from here.
        "entropy": float(ent[0]),
        "max_prob": float(mx[0]),
        "first_token_is_argmax": bool(int(logits[0].argmax()) == core[0]),
        # b2 at each read position. b2_first is one token PAST b1 and must not equal it.
        "b2_first": float(ent[1]) if m >= 1 else float("nan"),
        "b2_last": float(ent[m]),
        "b2_mean": float(ent[1:m + 1].mean()),
        "b3_mean_logprob": float(tok_logp.mean()),
        "n_tokens": int(n),
    }
