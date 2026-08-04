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

from dataclasses import dataclass


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
