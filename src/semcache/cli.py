"""Command line for the semantic-cache study.

semcache evaluate                       # lexical baseline only (no download)
semcache evaluate --e5 --llm            # + multilingual-e5-small + Qwen verifier
semcache demo "Comment déclarer un sinistre ?" "Comment je déclare un sinistre ?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .cache import SemanticCache
from .embed import HashingEmbedder, TransformerEmbedder
from .evaluate import render, run


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="semcache", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--data-dir", type=Path, default=Path("data"))
    e.add_argument("--reports", type=Path, default=Path("reports"))
    e.add_argument("--e5", action="store_true", help="add intfloat/multilingual-e5-small")
    e.add_argument("--llm", default=None, help="verifier model, e.g. Qwen/Qwen2.5-1.5B-Instruct")
    e.add_argument("--gen-llm", default=None, help="generator model used to time a cache miss")
    d = sub.add_parser("demo", help="cache the first question, look up the others")
    d.add_argument("questions", nargs="+")
    d.add_argument("--accept", type=float, default=0.6)
    args = p.parse_args(argv)

    if args.cmd == "demo":
        cache = SemanticCache(HashingEmbedder(), accept=args.accept)
        cache.put(args.questions[0], f"<answer to: {args.questions[0]}>")
        for q in args.questions[1:]:
            r = cache.get(q)
            print(f"{r.route:9s} score={r.score:.3f}  {q}")
        return

    embedders = [HashingEmbedder()]
    if args.e5:
        embedders.append(TransformerEmbedder())
    llm = gen = None
    if args.llm:
        from .llm import LocalLLM

        llm = LocalLLM(args.llm)
    if args.gen_llm:
        from .llm import LocalLLM

        gen = llm if llm is not None and args.gen_llm == args.llm else LocalLLM(args.gen_llm)
    res = run(args.data_dir, args.reports, embedders, llm=llm, gen_llm=gen)
    print(render(res))


if __name__ == "__main__":
    main()
