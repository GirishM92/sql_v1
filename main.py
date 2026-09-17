import argparse
import sqlite3

from colorama import Fore, Style, init

import config
from db.schema_seed import seed_database
from graph import run_pipeline

init(autoreset=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Banking-grade agentic Text-to-SQL with schema guardrails.",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--ollama",
        action="store_true",
        help="Use Ollama (local). Shortcut for --provider ollama",
    )
    mode_group.add_argument(
        "--openai",
        action="store_true",
        help="Use OpenAI. Shortcut for --provider openai",
    )
    mode_group.add_argument(
        "--peek",
        action="store_true",
        help="Use Peek AI / HDFC LiteLLM. Shortcut for --provider peek",
    )
    mode_group.add_argument(
        "--provider",
        help=(
            "LLM provider: "
            + ", ".join(config.known_provider_ids())
            + ". Aliases: local, cloud, claude, gemini, compatible."
        ),
    )
    parser.add_argument(
        "--model",
        help="Override model name for the selected provider",
    )
    parser.add_argument(
        "--base-url",
        dest="base_url",
        help="Override API base URL (Ollama, Azure, OpenAI-compatible, vLLM, LM Studio)",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        help="Override API key for the selected provider",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=None,
        help="Module ids to scope the query (e.g. user_master lef base_report). Required.",
    )
    return parser.parse_args()


def apply_cli_config(args: argparse.Namespace) -> None:
    if args.ollama:
        config.set_provider("ollama")
    elif args.openai:
        config.set_provider("openai")
    elif args.peek:
        config.set_provider("peek")
    elif args.provider:
        config.set_provider(args.provider)

    if args.model:
        config.set_model(args.model)
    if args.base_url:
        config.set_base_url(args.base_url)
    if args.api_key:
        config.set_api_key(args.api_key)


def print_banner():
    print(f"\n{Fore.CYAN}{'=' * 60}")
    print("  Banking-Grade Agentic Text-to-SQL")
    print(f"  Provider: {config.get_mode_label()}")
    print(f"  Database: {config.DB_PATH}")
    print(f"{'=' * 60}{Style.RESET_ALL}\n")


def print_logs(logs: list[str]):
    for log in logs:
        if "FAILED" in log or "Blocked" in log:
            print(f"{Fore.RED}{log}{Style.RESET_ALL}")
        elif "PASSED" in log or "Selected tables" in log:
            print(f"{Fore.GREEN}{log}{Style.RESET_ALL}")
        elif log.startswith("["):
            print(f"{Fore.YELLOW}{log}{Style.RESET_ALL}")
        else:
            print(f"  {log}")


def execute_query(sql: str):
    conn = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(sql)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print(f"\n{Fore.YELLOW}Query returned 0 rows.{Style.RESET_ALL}")
        return

    headers = rows[0].keys()
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}

    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    separator = "-+-".join("-" * col_widths[h] for h in headers)
    print(f"\n{Fore.CYAN}{header_line}{Style.RESET_ALL}")
    print(separator)
    for row in rows:
        print(" | ".join(str(row[h]).ljust(col_widths[h]) for h in headers))
    print(f"\n{Fore.GREEN}({len(rows)} row(s) returned){Style.RESET_ALL}")


def main():
    args = parse_args()
    apply_cli_config(args)

    if not config.DB_PATH.exists():
        print(f"{Fore.YELLOW}Database not found. Seeding mock bank database...{Style.RESET_ALL}")
        seed_database()

    print_banner()
    from tools.modules import list_modules

    modules = list_modules()
    print("Available modules:")
    for m in modules:
        print(f"  - {m['id']}: {m['name']} → {', '.join(m.get('tables') or [])}")
    print()

    module_ids = args.modules
    if not module_ids:
        print(
            f"{Fore.YELLOW}Tip: pass --modules user_master "
            f"(or lef / base_report).{Style.RESET_ALL}"
        )
        raw = input(
            f"{Fore.WHITE}Modules (space-separated ids): {Style.RESET_ALL}"
        ).strip()
        module_ids = [x for x in raw.split() if x]
    if not module_ids:
        print(f"{Fore.RED}At least one module is required.{Style.RESET_ALL}")
        return

    print(f"Active modules: {', '.join(module_ids)}")
    print("Type your question in natural language. Type 'quit' to exit.\n")

    while True:
        try:
            user_query = input(f"{Fore.WHITE}You: {Style.RESET_ALL}").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_query:
            continue
        if user_query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        print(f"\n{Fore.CYAN}Processing...{Style.RESET_ALL}\n")
        try:
            result = run_pipeline(user_query, module_ids=module_ids)
        except Exception as exc:
            print(f"{Fore.RED}Pipeline error: {exc}{Style.RESET_ALL}\n")
            continue

        print_logs(result.get("agent_logs", []))

        usage = (result.get("token_usage") or {}).get("query") or {}
        if usage:
            print(
                f"{Fore.CYAN}Peek tokens: prompt={usage.get('prompt_tokens', 0)} "
                f"completion={usage.get('completion_tokens', 0)} "
                f"total={usage.get('total_tokens', 0)} "
                f"calls={usage.get('calls', 0)}{Style.RESET_ALL}"
            )

        if result.get("validation_passed") and result.get("final_sql"):
            sql = result["final_sql"]
            print(f"\n{Fore.GREEN}Generated SQL:{Style.RESET_ALL}")
            print(f"{Fore.WHITE}{sql}{Style.RESET_ALL}")

            answer = input(
                f"\n{Fore.YELLOW}Execute this query? (y/N): {Style.RESET_ALL}"
            ).strip().lower()
            if answer == "y":
                from db.utils import execute_select

                try:
                    execute_select(sql, allowed_tables=result.get("allowed_tables"))
                    execute_query(sql)
                except Exception as exc:
                    print(f"{Fore.RED}Execution error: {exc}{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}Query not executed (HITL gate).{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.RED}Could not generate a valid SQL query.{Style.RESET_ALL}")
            errors = result.get("validation_errors", [])
            if errors:
                for err in errors:
                    print(f"  - {err}")

        print()


if __name__ == "__main__":
    main()
