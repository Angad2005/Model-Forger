#!/usr/bin/env python3
"""
dataset_audit.py - audit a JSONL training dataset and write a markdown report.

Usage:
    python dataset_audit.py data.jsonl
    python dataset_audit.py data.jsonl --max-len 4096 --eval-file eval.jsonl --clusters 25
    python dataset_audit.py data.jsonl --hf-tokenizer meta-llama/Llama-3.1-8B --ppl-model gpt2

Supported record formats (auto-detected):
    {"messages": [{"role": "...", "content": "..."}]}            (also "conversations" with from/value)
    {"prompt": "...", "completion"|"response": "..."}
    {"instruction": "...", "input": "...", "output": "..."}
    {"prompt": "...", "chosen": "...", "rejected": "..."}        (preference data)
    {"text": "..."}

Required: Python 3.8+ (stdlib only for core checks).
Optional (each unlocks more checks, skipped gracefully if missing):
    pip install tiktoken                 # token counts (default tokenizer)
    pip install transformers torch       # --hf-tokenizer, --ppl-model
    pip install datasketch               # near-duplicate detection
    pip install sentence-transformers scikit-learn   # topic clusters / coverage
    pip install matplotlib               # histogram PNGs

Outputs (in --out-dir, default ./audit_out):
    report.md              the full report
    flagged.jsonl          every flagged line with reasons
    review_sample.jsonl    random sample to read by hand / feed to an LLM judge
    length_hist.png        token length histogram (if matplotlib is installed)
"""
import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
from collections import Counter, defaultdict

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

ROLE_MAP = {"human": "user", "gpt": "assistant", "bot": "assistant", "model": "assistant",
            "ai": "assistant", "system": "system", "user": "user", "assistant": "assistant",
            "tool": "tool", "function": "tool"}

REFUSAL_PATTERNS = [
    r"as an ai (language )?model", r"i('m| am) (sorry|unable)", r"i cannot (assist|help|provide)",
    r"i can't (assist|help|provide)", r"as a large language model", r"i do not have (personal )?(opinions|feelings)",
    r"openai", r"i apologi[sz]e, but",
]
REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.I)
HTML_RE = re.compile(r"</?(div|span|p|br|html|body|script|style|table|td|tr|a|img)\b[^>]*>", re.I)
MOJIBAKE_RE = re.compile(r"(Ã.|â€|ï¿½|\ufffd)")
TEMPLATE_RE = re.compile(r"(\{\{.*?\}\}|<\|.*?\|>|\[INST\]|\[/INST\]|### (Instruction|Response))")


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, int(round(p / 100 * (len(sorted_vals) - 1))))
    return sorted_vals[i]


def stats_line(vals):
    if not vals:
        return "n/a"
    s = sorted(vals)
    return (f"mean={statistics.mean(s):.1f} median={pct(s, 50)} p5={pct(s, 5)} p95={pct(s, 95)} "
            f"p99={pct(s, 99)} min={s[0]} max={s[-1]}")


def ascii_hist(vals, bins=12, width=40):
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    if lo == hi:
        return f"{lo}: {'#' * width} ({len(vals)})"
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        counts[min(bins - 1, int((v - lo) / step))] += 1
    m = max(counts)
    lines = []
    for i, c in enumerate(counts):
        a, b = lo + i * step, lo + (i + 1) * step
        lines.append(f"{a:>8.0f}-{b:<8.0f} | {'#' * int(width * c / m):<{width}} {c}")
    return "\n".join(lines)


def to_text(x):
    """Coerce content (str / list of parts / dict / None) to a string."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        parts = []
        for p in x:
            if isinstance(p, dict):
                parts.append(str(p.get("text", "")))
            else:
                parts.append(str(p))
        return "\n".join(parts)
    if isinstance(x, dict):
        return str(x.get("text", x.get("content", "")))
    return str(x)


def norm_turns(msgs):
    turns = []
    for m in msgs:
        if not isinstance(m, dict):
            turns.append(("?", to_text(m)))
            continue
        role = m.get("role", m.get("from", "?"))
        content = m.get("content", m.get("value"))
        turns.append((ROLE_MAP.get(str(role).lower(), str(role).lower()), to_text(content)))
    return turns


def normalize(rec):
    """Return (kind, turns, extras) or None if schema is unrecognised."""
    if not isinstance(rec, dict):
        return None
    for key in ("messages", "conversations"):
        if isinstance(rec.get(key), list):
            return "chat", norm_turns(rec[key]), {}
    if "chosen" in rec and "rejected" in rec:
        prompt = to_text(rec.get("prompt"))
        ch, rj = rec["chosen"], rec["rejected"]
        if isinstance(ch, list):
            turns = norm_turns(ch)
            rj_text = to_text(norm_turns(rj)[-1][1]) if isinstance(rj, list) and rj else to_text(rj)
        else:
            turns = [("user", prompt), ("assistant", to_text(ch))]
            rj_text = to_text(rj)
        return "preference", turns, {"rejected": rj_text}
    if "prompt" in rec and ("completion" in rec or "response" in rec):
        return "pc", [("user", to_text(rec["prompt"])),
                      ("assistant", to_text(rec.get("completion", rec.get("response"))))], {}
    if "instruction" in rec and ("output" in rec or "response" in rec):
        user = to_text(rec["instruction"])
        if rec.get("input"):
            user += "\n" + to_text(rec["input"])
        return "alpaca", [("user", user), ("assistant", to_text(rec.get("output", rec.get("response"))))], {}
    if "text" in rec:
        return "text", [("text", to_text(rec["text"]))], {}
    return None


def prompt_text(turns):
    return "\n".join(t for r, t in turns if r != "assistant")


def response_text(turns):
    return "\n".join(t for r, t in turns if r == "assistant")


def full_text(turns):
    return "\n".join(t for _, t in turns)


def md5(s):
    return hashlib.md5(s.encode("utf-8", "ignore")).hexdigest()


def norm_ws(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def words(s):
    return re.findall(r"\w+", s.lower())


def script_of(ch):
    o = ord(ch)
    if 0x0900 <= o <= 0x097F: return "devanagari"
    if 0x0A00 <= o <= 0x0A7F: return "gurmukhi"
    if 0x0B80 <= o <= 0x0DFF: return "other_indic"
    if 0x4E00 <= o <= 0x9FFF or 0x3040 <= o <= 0x30FF: return "cjk"
    if 0xAC00 <= o <= 0xD7AF: return "hangul"
    if 0x0600 <= o <= 0x06FF: return "arabic"
    if 0x0400 <= o <= 0x04FF: return "cyrillic"
    if ch.isalpha(): return "latin"
    return None


def script_profile(s):
    c = Counter(x for x in (script_of(ch) for ch in s) if x)
    tot = sum(c.values())
    return {k: v / tot for k, v in c.items()} if tot else {}


def make_token_counter(args):
    if args.hf_tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.hf_tokenizer)
            return (lambda s: len(tok.encode(s, add_special_tokens=False))), f"HF:{args.hf_tokenizer}"
        except Exception as e:
            print(f"[warn] HF tokenizer failed ({e}); falling back.", file=sys.stderr)
    try:
        import tiktoken
        enc = tiktoken.get_encoding(args.tiktoken_encoding)
        return (lambda s: len(enc.encode(s, disallowed_special=()))), f"tiktoken:{args.tiktoken_encoding}"
    except Exception:
        print("[warn] tiktoken not installed; using whitespace word count x1.3 as a rough token estimate.",
              file=sys.stderr)
        return (lambda s: int(len(s.split()) * 1.3)), "approx (words*1.3)"


# ----------------------------------------------------------------------------
# Main audit
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path")
    ap.add_argument("--out-dir", default="audit_out")
    ap.add_argument("--max-len", type=int, default=4096, help="training context length in tokens")
    ap.add_argument("--tiktoken-encoding", default="cl100k_base")
    ap.add_argument("--hf-tokenizer", default=None, help="HF tokenizer name/path (overrides tiktoken)")
    ap.add_argument("--eval-file", default=None, help="JSONL eval/benchmark set to check contamination against")
    ap.add_argument("--clusters", type=int, default=20)
    ap.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--no-embed", action="store_true", help="skip embedding/clustering")
    ap.add_argument("--near-dup-threshold", type=float, default=0.8)
    ap.add_argument("--max-near-dup", type=int, default=100000, help="cap examples used for MinHash")
    ap.add_argument("--ppl-model", default=None, help="HF causal LM for perplexity on a sample (e.g. gpt2)")
    ap.add_argument("--ppl-sample", type=int, default=200)
    ap.add_argument("--review-sample", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    count_tokens, tok_name = make_token_counter(args)

    R = []          # report lines
    flags = defaultdict(list)  # line_no -> [reasons]

    def H(title):
        R.append(f"\n## {title}\n")

    def P(s=""):
        R.append(s)

    # ------------------------------------------------------------------
    # 1. Structure & validity
    # ------------------------------------------------------------------
    records = []  # dict(line, kind, turns, extras, raw)
    bad_json, unknown_schema, blank_lines = [], [], 0
    schema_sigs = Counter()
    empty_fields = 0

    with open(args.path, "r", encoding="utf-8", errors="replace") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                blank_lines += 1
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad_json.append(ln)
                flags[ln].append("invalid_json")
                continue
            if isinstance(rec, dict):
                schema_sigs[tuple(sorted(rec.keys()))] += 1
            n = normalize(rec)
            if n is None:
                unknown_schema.append(ln)
                flags[ln].append("unknown_schema")
                continue
            kind, turns, extras = n
            records.append({"line": ln, "kind": kind, "turns": turns, "extras": extras, "raw": rec})

    total_lines = len(records) + len(bad_json) + len(unknown_schema)
    H("1. Structure and validity")
    P(f"- File: `{args.path}`")
    P(f"- Non-blank lines: **{total_lines}**, usable records: **{len(records)}**")
    P(f"- Invalid JSON lines: **{len(bad_json)}** {bad_json[:10] if bad_json else ''}")
    P(f"- Unrecognised schema: **{len(unknown_schema)}** {unknown_schema[:10] if unknown_schema else ''}")
    P(f"- Blank lines: {blank_lines}")
    P(f"- Detected formats: {dict(Counter(r['kind'] for r in records))}")
    P("- Key sets (schema consistency):")
    for sig, c in schema_sigs.most_common(6):
        P(f"  - {list(sig)}: {c}")
    if len(schema_sigs) > 1:
        P("  - **WARNING: more than one key set present; check for inconsistent records.**")

    role_issues = Counter()
    trunc_suspects = 0
    for r in records:
        t, ln = r["turns"], r["line"]
        if not t:
            flags[ln].append("no_turns")
            role_issues["no_turns"] += 1
            continue
        for role, content in t:
            if content is None or not str(content).strip():
                empty_fields += 1
                flags[ln].append(f"empty_{role}_content")
        if r["kind"] != "text":
            roles = [x[0] for x in t]
            if roles[-1] != "assistant":
                role_issues["last_turn_not_assistant"] += 1
                flags[ln].append("last_turn_not_assistant")
            core = [x for x in roles if x not in ("system", "tool")]
            if core and core[0] != "user":
                role_issues["first_turn_not_user"] += 1
                flags[ln].append("first_turn_not_user")
            for a, b in zip(core, core[1:]):
                if a == b:
                    role_issues["consecutive_same_role"] += 1
                    flags[ln].append("consecutive_same_role")
                    break
            if "assistant" not in roles:
                role_issues["no_assistant_turn"] += 1
                flags[ln].append("no_assistant_turn")
            if any(x not in ("system", "user", "assistant", "tool") for x in roles):
                role_issues["unknown_role"] += 1
                flags[ln].append("unknown_role")
            resp = response_text(t).rstrip()
            if resp and len(resp) > 200 and not re.search(r"[.!?\"')\]}`>\u0964\u3002]$|```$", resp):
                trunc_suspects += 1
                flags[ln].append("possibly_truncated_response")
    P(f"- Empty / null content fields: **{empty_fields}**")
    P(f"- Role-structure issues: {dict(role_issues) if role_issues else 'none'}")
    P(f"- Responses (>200 chars) not ending in terminal punctuation (possible truncation): **{trunc_suspects}**")

    if not records:
        P("\nNo usable records; stopping.")
        write_outputs(args, R, flags, records)
        return

    # ------------------------------------------------------------------
    # 2. Size and length distribution
    # ------------------------------------------------------------------
    H("2. Size and length distribution")
    tot_tokens, prompt_toks, resp_toks, ratios, over = [], [], [], [], []
    for r in records:
        pt = count_tokens(prompt_text(r["turns"]))
        rt = count_tokens(response_text(r["turns"]))
        if r["kind"] == "text":
            pt, rt = 0, count_tokens(full_text(r["turns"]))
        tt = count_tokens(full_text(r["turns"]))
        r["tok"], r["ptok"], r["rtok"] = tt, pt, rt
        tot_tokens.append(tt); prompt_toks.append(pt); resp_toks.append(rt)
        if pt > 0 and rt > 0:
            ratios.append(rt / pt)
        if tt > args.max_len:
            over.append(r["line"]); flags[r["line"]].append(f"exceeds_max_len_{args.max_len}")
    P(f"- Tokenizer: `{tok_name}`")
    P(f"- Examples: **{len(records)}**, total tokens: **{sum(tot_tokens):,}**")
    P(f"- Total tokens/example: {stats_line(tot_tokens)}")
    P(f"- Prompt tokens: {stats_line(prompt_toks)}")
    P(f"- Response tokens: {stats_line(resp_toks)}")
    if ratios:
        sr = sorted(ratios)
        P(f"- Response/prompt token ratio: median={pct(sr, 50):.2f} p5={pct(sr, 5):.2f} p95={pct(sr, 95):.2f}")
        short_resp = sum(1 for r in records if r["kind"] != "text" and r["rtok"] < 5)
        P(f"- Responses under 5 tokens: **{short_resp}**")
    P(f"- Examples over max length ({args.max_len}): **{len(over)}** "
      f"({100 * len(over) / len(records):.1f}%)")
    P("\nToken-length histogram:\n```\n" + ascii_hist(tot_tokens) + "\n```")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
        ax[0].hist(tot_tokens, bins=50); ax[0].set_title("Total tokens / example")
        ax[0].axvline(args.max_len, color="r", ls="--")
        ax[1].hist(resp_toks, bins=50); ax[1].set_title("Response tokens / example")
        fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "length_hist.png"), dpi=120)
        P("\nSaved `length_hist.png`.")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 3. Duplicates and contamination
    # ------------------------------------------------------------------
    H("3. Duplicates and contamination")
    by_full, by_prompt = defaultdict(list), defaultdict(list)
    for r in records:
        by_full[md5(norm_ws(full_text(r["turns"])))].append(r["line"])
        if r["kind"] != "text":
            by_prompt[md5(norm_ws(prompt_text(r["turns"])))].append(r["line"])
    exact_groups = [v for v in by_full.values() if len(v) > 1]
    exact_extra = sum(len(v) - 1 for v in exact_groups)
    for g in exact_groups:
        for ln in g[1:]:
            flags[ln].append(f"exact_duplicate_of_line_{g[0]}")
    P(f"- Exact duplicates (whitespace/case-normalised): **{exact_extra}** extra copies in "
      f"{len(exact_groups)} groups ({100 * exact_extra / len(records):.2f}%)")
    conflicts = []
    resp_by_line = {r["line"]: md5(norm_ws(response_text(r["turns"]))) for r in records}
    for h, lns in by_prompt.items():
        if len(lns) > 1 and len({resp_by_line[l] for l in lns}) > 1:
            conflicts.append(lns)
    dup_prompts = sum(1 for v in by_prompt.values() if len(v) > 1)
    P(f"- Prompts appearing more than once: **{dup_prompts}**; of those, with *different* responses: "
      f"**{len(conflicts)}** (may be fine, may be conflicting labels)")
    for g in conflicts[:3]:
        P(f"  - lines {g[:6]}")

    try:
        from datasketch import MinHash, MinHashLSH
        cand = records[: args.max_near_dup]
        lsh = MinHashLSH(threshold=args.near_dup_threshold, num_perm=64)
        near_pairs = 0
        near_lines = set()
        for r in cand:
            w = words(full_text(r["turns"]))
            sh = {" ".join(w[i:i + 5]) for i in range(max(1, len(w) - 4))}
            m = MinHash(num_perm=64)
            for s in sh:
                m.update(s.encode("utf-8"))
            hits = lsh.query(m)
            if hits:
                near_pairs += 1
                near_lines.add(r["line"])
                flags[r["line"]].append(f"near_duplicate_of_line_{hits[0]}")
            try:
                lsh.insert(str(r["line"]), m)
            except ValueError:
                pass
        # note: hits are keys (str line numbers)
        P(f"- Near-duplicates (5-gram MinHash, Jaccard >= {args.near_dup_threshold}, "
          f"checked {len(cand)} examples): **{len(near_lines)}** "
          f"({100 * len(near_lines) / len(cand):.2f}%) beyond the first occurrence")
    except ImportError:
        P("- Near-duplicate check skipped (`pip install datasketch`).")

    if args.eval_file:
        ev_prompts, ev_ngrams = set(), set()
        with open(args.eval_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    n = normalize(json.loads(line))
                except Exception:
                    continue
                if not n:
                    continue
                pt = prompt_text(n[1]) if n[0] != "text" else full_text(n[1])
                ev_prompts.add(md5(norm_ws(pt)))
                w = words(pt)
                for i in range(len(w) - 7):
                    ev_ngrams.add(hash(" ".join(w[i:i + 8])))
        exact_c, ngram_c = 0, 0
        for r in records:
            pt = prompt_text(r["turns"]) if r["kind"] != "text" else full_text(r["turns"])
            if md5(norm_ws(pt)) in ev_prompts:
                exact_c += 1; flags[r["line"]].append("eval_contamination_exact_prompt")
                continue
            w = words(pt)
            grams = [hash(" ".join(w[i:i + 8])) for i in range(len(w) - 7)]
            if grams and sum(g in ev_ngrams for g in grams) / len(grams) > 0.5:
                ngram_c += 1; flags[r["line"]].append("eval_contamination_8gram")
        P(f"- Eval contamination vs `{args.eval_file}`: exact prompt matches **{exact_c}**, "
          f">50% 8-gram overlap **{ngram_c}**")
    else:
        P("- Eval contamination check skipped (pass `--eval-file`).")

    # ------------------------------------------------------------------
    # 4. Diversity and coverage
    # ------------------------------------------------------------------
    H("4. Diversity and coverage")
    all_prompts = [prompt_text(r["turns"]) if r["kind"] != "text" else full_text(r["turns"]) for r in records]
    all_resps = [response_text(r["turns"]) if r["kind"] != "text" else "" for r in records]

    def distinct_n(texts, n, cap=2_000_000):
        seen, tot = set(), 0
        for t in texts:
            w = words(t)
            for i in range(len(w) - n + 1):
                seen.add(hash(" ".join(w[i:i + n]))); tot += 1
                if tot >= cap:
                    break
            if tot >= cap:
                break
        return len(seen) / tot if tot else 0

    P("| Field | distinct-1 | distinct-2 | distinct-3 |")
    P("|---|---|---|---|")
    P(f"| prompts | {distinct_n(all_prompts, 1):.3f} | {distinct_n(all_prompts, 2):.3f} | {distinct_n(all_prompts, 3):.3f} |")
    if any(all_resps):
        P(f"| responses | {distinct_n(all_resps, 1):.3f} | {distinct_n(all_resps, 2):.3f} | {distinct_n(all_resps, 3):.3f} |")
    openers = Counter(" ".join(words(p)[:3]) for p in all_prompts)
    P("\nMost common prompt openings (first 3 words) - high concentration = templated data:")
    for k, c in openers.most_common(10):
        P(f"- `{k}`: {c} ({100 * c / len(records):.1f}%)")
    resp_open = Counter(" ".join(words(x)[:4]) for x in all_resps if x)
    if resp_open:
        P("\nMost common response openings (first 4 words):")
        for k, c in resp_open.most_common(8):
            P(f"- `{k}`: {c} ({100 * c / len(records):.1f}%)")

    # script / language mix
    dom = Counter()
    mixed = 0
    for r, p in zip(records, all_prompts):
        prof = script_profile(p + " " + response_text(r["turns"]))
        if prof:
            top = max(prof, key=prof.get)
            dom[top] += 1
            if sum(1 for v in prof.values() if v > 0.15) > 1:
                mixed += 1
                r["mixed_script"] = True
    P(f"\nDominant script per example: {dict(dom.most_common())}")
    P(f"Examples mixing 2+ scripts (>15% each): **{mixed}** (expected for code-mixed data, suspicious otherwise)")

    # heuristic task-type tags
    def tag(p, resp):
        s = (p + " " + resp).lower()
        if "```" in s or re.search(r"\b(def |class |import |function |#include|public static)\b", s): return "code"
        if re.search(r"\b(translate|translation)\b", p.lower()): return "translation"
        if re.search(r"\b(summari[sz]e|summary|tl;dr)\b", p.lower()): return "summarization"
        if re.search(r"\b(solve|calculate|equation|prove|how many|probability)\b", p.lower()) or re.search(r"\d+\s*[\+\-\*/=]\s*\d+", p): return "math/reasoning"
        if re.search(r"\b(write|compose|story|poem|essay|email|letter)\b", p.lower()): return "creative/writing"
        if re.search(r"\b(classify|sentiment|label|categor)", p.lower()): return "classification"
        if "?" in p: return "qa/open-question"
        return "other"
    tags = Counter()
    for r, p, rs in zip(records, all_prompts, all_resps):
        r["tag"] = tag(p, rs); tags[r["tag"]] += 1
    P("\nHeuristic task-type distribution (keyword-based; rough - use an LLM/classifier for precision):")
    for k, c in tags.most_common():
        P(f"- {k}: {c} ({100 * c / len(records):.1f}%)")
    turns_dist = Counter(sum(1 for x in r["turns"] if x[0] == "user") for r in records if r["kind"] != "text")
    if turns_dist:
        P(f"\nUser turns per conversation: {dict(sorted(turns_dist.items())[:10])}")
    sys_frac = sum(1 for r in records if any(x[0] == "system" for x in r["turns"])) / len(records)
    P(f"Examples with a system prompt: {100 * sys_frac:.1f}%")

    # clustering
    if not args.no_embed:
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np
            sample_idx = list(range(len(records)))
            if len(sample_idx) > 50000:
                sample_idx = random.sample(sample_idx, 50000)
            texts = [all_prompts[i][:512] for i in sample_idx]
            model = SentenceTransformer(args.embed_model)
            emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
            k = max(2, min(args.clusters, len(texts) // 5))
            km = KMeans(n_clusters=k, n_init=3, random_state=args.seed).fit(emb)
            tf = TfidfVectorizer(stop_words="english", max_features=20000)
            X = tf.fit_transform(texts)
            terms = np.array(tf.get_feature_names_out())
            sizes = Counter(km.labels_)
            P(f"\n### Topic clusters (k={k}, {len(texts)} prompts, `{args.embed_model}`)\n")
            P("| Cluster | Size | % | Top terms | Example prompt |")
            P("|---|---|---|---|---|")
            for cl, sz in sizes.most_common():
                idxs = np.where(km.labels_ == cl)[0]
                top = terms[np.asarray(X[idxs].mean(axis=0)).ravel().argsort()[::-1][:8]]
                centre = idxs[np.argmax(emb[idxs] @ km.cluster_centers_[cl])]
                ex = texts[centre].replace("\n", " ").replace("|", "/")[:90]
                P(f"| {cl} | {sz} | {100 * sz / len(texts):.1f} | {', '.join(top)} | {ex} |")
            P("\nRead: very large clusters are over-represented; tiny clusters (or topics you expect but don't see) "
              "are where to add data. Compare against real user queries for the best signal.")
        except ImportError:
            P("\nTopic clustering skipped (`pip install sentence-transformers scikit-learn`).")

    # ------------------------------------------------------------------
    # 5. Quality heuristics
    # ------------------------------------------------------------------
    H("5. Quality heuristics")
    hc = Counter()
    rt_sorted = sorted(t for t in resp_toks if t > 0)
    lo_cut = pct(rt_sorted, 1) if rt_sorted else 0
    hi_cut = pct(rt_sorted, 99) if rt_sorted else 0
    for r, p, rs in zip(records, all_prompts, all_resps):
        ln = r["line"]
        target = rs if rs else p
        if REFUSAL_RE.search(target):
            hc["refusal/AI-boilerplate"] += 1; flags[ln].append("refusal_or_ai_boilerplate")
        if HTML_RE.search(target):
            hc["html_artifacts"] += 1; flags[ln].append("html_artifacts")
        if MOJIBAKE_RE.search(target):
            hc["mojibake/encoding"] += 1; flags[ln].append("mojibake")
        if TEMPLATE_RE.search(target):
            hc["template_leftovers"] += 1; flags[ln].append("template_leftovers")
        lines_ = [x.strip() for x in target.splitlines() if x.strip()]
        if len(lines_) >= 6 and len(set(lines_)) / len(lines_) < 0.5:
            hc["repeated_lines"] += 1; flags[ln].append("repeated_lines")
        w = words(target)
        if len(w) >= 40:
            tri = [" ".join(w[i:i + 3]) for i in range(len(w) - 2)]
            if len(set(tri)) / len(tri) < 0.5:
                hc["repetitive_ngrams"] += 1; flags[ln].append("repetitive_ngrams")
        if rs and norm_ws(rs) == norm_ws(p):
            hc["response_equals_prompt"] += 1; flags[ln].append("response_equals_prompt")
        if rs and r["rtok"] and (r["rtok"] <= lo_cut and lo_cut < 3 or r["rtok"] >= hi_cut and hi_cut > 0 and r["rtok"] > 3 * max(1, pct(rt_sorted, 50))):
            hc["response_length_outlier"] += 1; flags[ln].append("response_length_outlier")
        if r.get("mixed_script"):
            hc["mixed_script"] += 1
        if re.search(r"\s{6,}\S|\n{5,}", target):
            hc["whitespace_noise"] += 1; flags[ln].append("whitespace_noise")
    P("| Heuristic | Count | % |")
    P("|---|---|---|")
    for k, c in hc.most_common():
        P(f"| {k} | {c} | {100 * c / len(records):.2f} |")
    if not hc:
        P("| (nothing flagged) | 0 | 0 |")

    if args.ppl_model:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tk = AutoTokenizer.from_pretrained(args.ppl_model)
            lm = AutoModelForCausalLM.from_pretrained(args.ppl_model).eval()
            idxs = random.sample(range(len(records)), min(args.ppl_sample, len(records)))
            ppls = []
            for i in idxs:
                enc = tk(full_text(records[i]["turns"]), return_tensors="pt", truncation=True, max_length=512)
                if enc["input_ids"].shape[1] < 2:
                    continue
                with torch.no_grad():
                    loss = lm(**enc, labels=enc["input_ids"]).loss.item()
                ppl = math.exp(min(loss, 20))
                ppls.append((ppl, records[i]["line"]))
            ps = sorted(p for p, _ in ppls)
            P(f"\nPerplexity under `{args.ppl_model}` on {len(ppls)} sampled examples: "
              f"median={pct(ps, 50):.1f} p90={pct(ps, 90):.1f} max={ps[-1]:.1f}")
            P("Highest-perplexity lines (check for gibberish/noise): " +
              ", ".join(f"{ln} ({p:.0f})" for p, ln in sorted(ppls, reverse=True)[:8]))
        except Exception as e:
            P(f"\nPerplexity check failed: {e}")

    # ------------------------------------------------------------------
    # 6. Label / preference balance
    # ------------------------------------------------------------------
    H("6. Label and balance checks")
    label_keys = [k for k in ("label", "labels", "category", "class", "sentiment", "task", "source", "language", "lang")
                  if any(k in r["raw"] for r in records[:50] if isinstance(r["raw"], dict))]
    if label_keys:
        for k in label_keys:
            c = Counter(str(r["raw"].get(k)) for r in records if isinstance(r["raw"], dict))
            top = c.most_common(15)
            P(f"- `{k}`: {len(c)} distinct values; top: {top}")
            if len(c) > 1:
                imb = top[0][1] / max(1, min(c.values()))
                P(f"  - imbalance ratio (largest/smallest): {imb:.1f}x")
    else:
        P("- No obvious label/metadata columns found (looked for label, category, class, source, language, ...).")
    pref = [r for r in records if r["kind"] == "preference"]
    if pref:
        diffs = [count_tokens(response_text(r["turns"])) - count_tokens(r["extras"]["rejected"]) for r in pref]
        longer = sum(1 for d in diffs if d > 0) / len(diffs)
        same = sum(1 for r in pref if norm_ws(response_text(r["turns"])) == norm_ws(r["extras"]["rejected"]))
        P(f"- Preference pairs: {len(pref)}; chosen is longer in **{100 * longer:.1f}%** of pairs "
          f"(far from ~50% means length bias); identical chosen/rejected: **{same}**")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    flagged_lines = len(flags)
    reasons = Counter(re.sub(r"_(of_)?line_\d+$", "", x) for v in flags.values() for x in v)
    R.insert(0, f"# Dataset audit: `{os.path.basename(args.path)}`\n\n"
                f"**{len(records)}** usable records, **{flagged_lines}** lines flagged "
                f"({100 * flagged_lines / max(1, total_lines):.1f}%).\n\n"
                "Flag reasons: " + ", ".join(f"{k}={v}" for k, v in reasons.most_common()))

    # review sample
    sample = random.sample(records, min(args.review_sample, len(records)))
    with open(os.path.join(args.out_dir, "review_sample.jsonl"), "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps({"line": r["line"], "tag": r.get("tag"), "flags": flags.get(r["line"], []),
                                "record": r["raw"]}, ensure_ascii=False) + "\n")
    P("\n## Next steps\n")
    P("- Read `review_sample.jsonl` by hand (50-100 examples); this finds issues no metric will.")
    P("- Feed `review_sample.jsonl` (or a 200-500 sample) to an LLM judge with a rubric for correctness, "
      "helpfulness, instruction-following and format.")
    P("- Fix/remove flagged lines (`flagged.jsonl`) *before* adding new data, then use the cluster table to decide what to add.")

    write_outputs(args, R, flags, records)


def write_outputs(args, R, flags, records):
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(R) + "\n")
    with open(os.path.join(args.out_dir, "flagged.jsonl"), "w", encoding="utf-8") as f:
        for ln in sorted(flags):
            f.write(json.dumps({"line": ln, "reasons": flags[ln]}) + "\n")
    print("\n".join(R))
    print(f"\nWrote report.md, flagged.jsonl, review_sample.jsonl to {args.out_dir}/")


if __name__ == "__main__":
    main()