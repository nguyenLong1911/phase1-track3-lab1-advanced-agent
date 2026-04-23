from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print

from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.utils import load_dataset, save_jsonl

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dataset: str = "data/hotpot_mini.json",
    out_dir: str = "outputs/live_run",
    reflexion_attempts: int = 3,
) -> None:
    examples = load_dataset(dataset)
    print(f"[bold]Loaded {len(examples)} examples from [cyan]{dataset}[/cyan][/bold]")

    react = ReActAgent()
    reflexion = ReflexionAgent(max_attempts=reflexion_attempts)

    print("[yellow]Running ReAct agent...[/yellow]")
    react_records = []
    for i, example in enumerate(examples, 1):
        record = react.run(example)
        react_records.append(record)
        status = "[green]OK[/green]" if record.is_correct else "[red]FAIL[/red]"
        print(f"  {status} [{i:3d}/{len(examples)}] {example.qid}: {record.predicted_answer!r}")

    print("[yellow]Running Reflexion agent...[/yellow]")
    reflexion_records = []
    for i, example in enumerate(examples, 1):
        record = reflexion.run(example)
        reflexion_records.append(record)
        status = "[green]OK[/green]" if record.is_correct else "[red]FAIL[/red]"
        attempts_str = f"({record.attempts} attempt{'s' if record.attempts > 1 else ''})"
        print(f"  {status} [{i:3d}/{len(examples)}] {example.qid} {attempts_str}: {record.predicted_answer!r}")

    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)

    report = build_report(all_records, dataset_name=Path(dataset).name, mode="live")
    json_path, md_path = save_report(report, out_path)
    print(f"\n[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print("\n[bold]Summary:[/bold]")
    print(json.dumps(report.summary, indent=2))


if __name__ == "__main__":
    app()
