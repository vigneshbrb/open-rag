"""Command-line entry point."""

import argparse
import sys


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ask questions about local documents.")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="Index TXT or text-based PDF files")
    ingest.add_argument("paths", nargs="+", help="Document paths")
    chat = commands.add_parser("chat", help="Ask multiple questions")
    ask = commands.add_parser("ask", help="Ask one question")
    ask.add_argument("question")
    for command in (chat, ask):
        command.add_argument("--provider", choices=("ollama", "openai"), default="ollama")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    if args.command == "ingest":
        from .documents import DocumentError
        from .index import DocumentIndex

        index = DocumentIndex()
        failed = False
        for path in args.paths:
            try:
                print(f"{index.index_file(path)}: {path}")
            except (DocumentError, OSError, ValueError) as exc:
                print(f"error: {exc}", file=sys.stderr)
                failed = True
            except Exception:
                print(f"error: cannot index {path}; check the local embedding model and index", file=sys.stderr)
                failed = True
        return 1 if failed else 0

    from .index import DocumentIndex
    from .providers import ProviderError
    from .qa import answer_question

    index = DocumentIndex()

    def run_question(question: str) -> bool:
        try:
            print(answer_question(index, question, args.provider))
            return True
        except (ValueError, ProviderError) as exc:
            print(f"error: {exc}", file=sys.stderr)
        except Exception:
            print("error: query failed; check the local embedding model and index", file=sys.stderr)
        return False

    if args.command == "ask":
        return 0 if run_question(args.question) else 1

    if index.count() == 0:
        print("error: No documents indexed. Run `python -m open_rag ingest <paths...>` first.", file=sys.stderr)
        return 1
    while True:
        try:
            question = input("Question (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit"}:
            return 0
        if question:
            run_question(question)
