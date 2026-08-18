"""
Rich UI panel rendering for dataset summaries, inspection tables, tool results, agent responses, plan progress, and friendly error messages.
Refined with Arch-inspired theme badges and clean table formatting.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from proton_agent.context.dataset import DatasetContext

if TYPE_CHECKING:
    from proton_agent.agent.state import AgentResponse, Plan, PlanStep
    from proton_agent.tools.registry import ToolRegistry


def render_dataset_loaded(console: Console, dataset: DatasetContext) -> None:
    """Renders a success panel after loading a dataset."""
    text = Text()
    text.append("✓ Dataset loaded successfully\n\n", style="bold green")
    text.append("Dataset:  ", style="dim white")
    text.append(f"{dataset.name}\n", style="bold bright_cyan")
    text.append("Rows:     ", style="dim white")
    text.append(f"{dataset.rows:,}\n", style="bold white")
    text.append("Columns:  ", style="dim white")
    text.append(f"{dataset.columns_count:,}", style="bold white")

    panel = Panel(
        text,
        title="[bold green]▲ Data Ingest[/bold green]",
        border_style="bright_green",
        padding=(0, 2),
        expand=False,
    )
    console.print(panel)


def render_test_dataset_loaded(console: Console, dataset: DatasetContext) -> None:
    """Renders a success panel after loading a test dataset specifically for model evaluation."""
    text = Text()
    text.append("✓ Test Dataset loaded successfully for model evaluation\n\n", style="bold green")
    text.append("Test Dataset: ", style="dim white")
    text.append(f"{dataset.name}\n", style="bold bright_cyan")
    text.append("Rows:         ", style="dim white")
    text.append(f"{dataset.rows:,}\n", style="bold white")
    text.append("Columns:      ", style="dim white")
    text.append(f"{dataset.columns_count:,}", style="bold white")

    panel = Panel(
        text,
        title="[bold green]▲ Model Test Data Ingest[/bold green]",
        border_style="bright_green",
        padding=(0, 2),
        expand=False,
    )
    console.print(panel)


def render_dataset_inspection(console: Console, dataset: DatasetContext) -> None:
    """Renders comprehensive dataset inspection details including columns and preview."""
    info_table = Table.grid(padding=(0, 2))
    info_table.add_column(style="dim white")
    info_table.add_column(style="bold bright_white")

    kb_memory = dataset.memory_usage_bytes / 1024
    info_table.add_row("File Path:", str(dataset.filepath))
    info_table.add_row("Dimensions:", f"{dataset.rows:,} rows × {dataset.columns_count:,} columns")
    info_table.add_row("Total Missing:", f"{dataset.total_missing:,} values")
    info_table.add_row("Memory Footprint:", f"{kb_memory:.1f} KB")

    summary_panel = Panel(
        info_table,
        title=f"[bold bright_cyan]▲ Dataset Overview :: {dataset.name}[/bold bright_cyan]",
        border_style="cyan",
        padding=(0, 2),
        expand=False,
    )

    col_table = Table(
        title="[bold bright_white]▲ Schema & Missing Value Distribution[/bold bright_white]",
        header_style="bold bright_cyan",
        border_style="bright_blue",
        show_lines=False,
    )
    col_table.add_column("Column", style="bold cyan")
    col_table.add_column("Type", style="yellow")
    col_table.add_column("Non-Null Count", justify="right", style="green")
    col_table.add_column("Missing", justify="right")
    col_table.add_column("Missing %", justify="right")

    for col in dataset.get_column_info():
        missing_style = "bold red" if col["missing"] > 0 else "dim white"
        col_table.add_row(
            col["name"],
            col["dtype"],
            f"{col['non_null']:,}",
            f"[{missing_style}]{col['missing']:,}[/{missing_style}]",
            f"[{missing_style}]{col['missing_pct']:.1f}%[/{missing_style}]",
        )

    head_df = dataset.head(5)
    preview_table = Table(
        title="[bold bright_white]▲ Data Preview (Head 5 Rows)[/bold bright_white]",
        header_style="bold bright_cyan",
        border_style="bright_blue",
        show_lines=True,
    )
    for col_name in head_df.columns:
        preview_table.add_column(str(col_name), overflow="ellipsis")
    for _, row in head_df.iterrows():
        preview_table.add_row(*[str(val) for val in row.values])

    console.print()
    console.print(summary_panel)
    console.print(col_table)
    console.print(preview_table)
    console.print()


def render_quality_audit(console: Console, data: dict[str, Any]) -> None:
    """Renders data quality health score and warnings."""
    score = data.get("quality_score", 100.0)
    status = data.get("status", "Good")
    warnings = data.get("warnings", [])

    score_color = "bright_green" if score >= 90 else ("yellow" if score >= 70 else "red")

    text = Text()
    text.append("Health Score: ", style="bold white")
    text.append(f"{score}% ", style=f"bold {score_color}")
    text.append(f"({status})\n\n", style="dim white")

    if not warnings:
        text.append("✓ Dataset is clean with no data quality warnings!", style="bold green")
    else:
        text.append("Quality Warnings Detected:\n", style="bold yellow")
        for w in warnings:
            text.append(f"  • {w}\n", style="yellow")

    panel = Panel(
        text,
        title="[bold bright_cyan]▲ Data Quality Audit[/bold bright_cyan]",
        border_style=score_color,
        padding=(0, 2),
        expand=False,
    )
    console.print(panel)


def render_missing_values(console: Console, data: dict[str, Any]) -> None:
    """Renders missing value analysis results."""
    text = Text()
    total_missing = data.get("total_missing", 0)
    pct = data.get("overall_missing_pct", 0.0)
    cols_count = data.get("columns_with_missing_count", 0)

    if total_missing == 0:
        text.append("✓ No missing values detected in any column!", style="bold green")
        panel = Panel(text, title="[bold green]▲ Missing Values[/bold green]", border_style="green", padding=(0, 2), expand=False)
        console.print(panel)
        return

    text.append(f"Total Missing Values:  {total_missing:,}\n", style="bold red")
    text.append(f"Overall Missing Rate:  {pct:.1f}%\n", style="yellow")
    text.append(f"Affected Columns:      {cols_count}", style="bold white")

    panel = Panel(text, title="[bold red]▲ Missing Values Analysis[/bold red]", border_style="red", padding=(0, 2), expand=False)
    console.print(panel)

    cols_with_missing = data.get("columns_with_missing", [])
    if cols_with_missing:
        table = Table(title="[bold]Columns With Missing Data[/bold]", header_style="bold cyan", border_style="bright_blue")
        table.add_column("Column", style="bold cyan")
        table.add_column("Type", style="yellow")
        table.add_column("Missing Count", justify="right", style="bold red")
        table.add_column("Missing %", justify="right", style="bold red")

        for col in cols_with_missing:
            table.add_row(col["name"], col["dtype"], f"{col['missing']:,}", f"{col['missing_pct']:.1f}%")

        console.print(table)


def render_duplicates(console: Console, data: dict[str, Any]) -> None:
    """Renders duplicate row detection results."""
    text = Text()
    dup_rows = data.get("duplicate_rows", 0)
    dup_pct = data.get("duplicate_pct", 0.0)
    total_rows = data.get("total_rows", 0)

    if dup_rows == 0:
        text.append("✓ No duplicate rows detected in dataset!", style="bold green")
        panel = Panel(text, title="[bold green]▲ Duplicates Analysis[/bold green]", border_style="green", padding=(0, 2), expand=False)
    else:
        text.append(f"Duplicate Rows:  {dup_rows:,} / {total_rows:,}\n", style="bold red")
        text.append(f"Duplicate Rate:  {dup_pct:.1f}%", style="yellow")
        panel = Panel(text, title="[bold yellow]▲ Duplicates Analysis[/bold yellow]", border_style="yellow", padding=(0, 2), expand=False)

    console.print(panel)


def render_data_types(console: Console, data: dict[str, Any]) -> None:
    """Renders categorized column data types."""
    table = Table(title="[bold bright_white]▲ Data Types Categorization[/bold bright_white]", header_style="bold bright_cyan", border_style="cyan")
    table.add_column("Category", style="bold cyan")
    table.add_column("Count", justify="right", style="bold white")
    table.add_column("Columns", style="yellow")

    num = data.get("numerical_columns", [])
    cat = data.get("categorical_columns", [])
    bool_cols = data.get("boolean_columns", [])
    dt = data.get("datetime_columns", [])

    table.add_row("Numerical", str(len(num)), ", ".join(num) if num else "None")
    table.add_row("Categorical", str(len(cat)), ", ".join(cat) if cat else "None")
    table.add_row("Boolean", str(len(bool_cols)), ", ".join(bool_cols) if bool_cols else "None")
    table.add_row("Datetime", str(len(dt)), ", ".join(dt) if dt else "None")

    console.print(table)


def render_outliers(console: Console, data: dict[str, Any]) -> None:
    """Renders IQR outlier analysis."""
    tot = data.get("total_outliers", 0)
    by_col = data.get("outliers_by_column", {})

    if tot == 0:
        text = Text("✓ No IQR statistical outliers detected!", style="bold green")
        panel = Panel(text, title="[bold green]▲ Outliers Analysis (IQR)[/bold green]", border_style="green", padding=(0, 2), expand=False)
        console.print(panel)
        return

    table = Table(title=f"[bold bright_white]▲ IQR Outlier Detection ({tot:,} Total Outliers)[/bold bright_white]", header_style="bold bright_cyan", border_style="bright_blue")
    table.add_column("Column", style="bold cyan")
    table.add_column("Outlier Count", justify="right", style="bold red")
    table.add_column("Outlier %", justify="right", style="bold yellow")
    table.add_column("IQR Bounds [Lower, Upper]", style="dim white")

    for col, stats in by_col.items():
        if stats["outlier_count"] > 0:
            table.add_row(
                col,
                f"{stats['outlier_count']:,}",
                f"{stats['outlier_pct']:.1f}%",
                f"[{stats['lower_bound']}, {stats['upper_bound']}]",
            )

    console.print(table)


def render_distribution(console: Console, data: dict[str, Any]) -> None:
    """Renders distribution shape metrics."""
    num_dist = data.get("numerical_distributions", {})
    if not num_dist:
        return

    table = Table(title="[bold bright_white]▲ Numerical Distribution Shapes & Skewness[/bold bright_white]", header_style="bold bright_cyan", border_style="cyan")
    table.add_column("Column", style="bold cyan")
    table.add_column("Mean", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Skewness", justify="right", style="bold yellow")
    table.add_column("Kurtosis", justify="right", style="dim white")
    table.add_column("Shape", style="bold white")

    for col, d in num_dist.items():
        shape_style = "green" if "Symmetric" in d["shape"] else "yellow"
        table.add_row(
            col,
            f"{d['mean']:.2f}",
            f"{d['median']:.2f}",
            f"{d['std']:.2f}",
            f"{d['skewness']:+.2f}",
            f"{d['kurtosis']:+.2f}",
            f"[{shape_style}]{d['shape']}[/{shape_style}]",
        )

    console.print(table)


def render_statistics(console: Console, data: dict[str, Any]) -> None:
    """Renders numerical statistics table."""
    stats = data.get("statistics") or data.get("numerical_summary", {})
    if not stats:
        console.print(f"[yellow]{data.get('message', 'No numerical statistics available.')}[/yellow]")
        return

    table = Table(title="[bold bright_white]▲ Descriptive Statistics (Numerical Columns)[/bold bright_white]", header_style="bold bright_cyan", border_style="bright_blue")
    table.add_column("Column", style="bold cyan")
    table.add_column("Count", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("25%", justify="right")
    table.add_column("Median (50%)", justify="right")
    table.add_column("75%", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Skew", justify="right")

    for col_name, s in stats.items():
        cnt = s.get("count", 0)
        mean_val = s.get("mean", 0.0)
        std_val = s.get("std", 0.0)
        min_val = s.get("min", 0.0)
        q25_val = s.get("q25", s.get("25%", 0.0))
        med_val = s.get("median", s.get("50%", 0.0))
        q75_val = s.get("q75", s.get("75%", 0.0))
        max_val = s.get("max", 0.0)
        skew_val = s.get("skew", 0.0)

        table.add_row(
            col_name,
            f"{int(cnt):,}",
            f"{float(mean_val):.2f}",
            f"{float(std_val):.2f}",
            f"{float(min_val):.2f}",
            f"{float(q25_val):.2f}",
            f"{float(med_val):.2f}",
            f"{float(q75_val):.2f}",
            f"{float(max_val):.2f}",
            f"{float(skew_val):.2f}",
        )

    console.print(table)


def render_correlation(console: Console, data: dict[str, Any]) -> None:
    """Renders correlation matrix and top pairwise correlations."""
    top_pairs = data.get("top_correlations", [])
    if not top_pairs:
        console.print(f"[yellow]{data.get('message', 'No correlation data available.')}[/yellow]")
        return

    table = Table(title="[bold bright_white]▲ Top Feature Correlations[/bold bright_white]", header_style="bold bright_cyan", border_style="bright_blue")
    table.add_column("Feature 1", style="bold cyan")
    table.add_column("Feature 2", style="bold cyan")
    table.add_column("Pearson Correlation", justify="right", style="bold yellow")

    for pair in top_pairs:
        corr_val = pair["correlation"]
        style = "bold red" if abs(corr_val) >= 0.7 else ("bold yellow" if abs(corr_val) >= 0.4 else "white")
        table.add_row(pair["feature_1"], pair["feature_2"], f"[{style}]{corr_val:+.4f}[/{style}]")

    console.print(table)


def render_plan(console: Console, plan: Plan) -> None:
    """Renders structured plan checklist."""
    text = Text()
    text.append("Goal: ", style="dim white")
    text.append(f"{plan.goal}\n\n", style="bold bright_white")

    for step in plan.steps:
        status = "[bold green]✓[/bold green]" if step.completed else "[dim white]○[/dim white]"
        text.append(f"{status} Step {step.step_number}: ", style="bold cyan")
        text.append(f"{step.description} ", style="white" if step.completed else "dim white")
        text.append(f"({step.tool_name})\n", style="dim cyan")

    panel = Panel(text, title="[bold bright_cyan]▲ Agent Execution Plan[/bold bright_cyan]", border_style="cyan", padding=(0, 2), expand=False)
    console.print(panel)


def render_plan_summary(console: Console, summary: str) -> None:
    """Renders key findings summary from plan execution."""
    if not summary:
        return
    text = Text(summary)
    panel = Panel(text, title="[bold green]▲ Key Findings Summary[/bold green]", border_style="green", padding=(0, 2), expand=False)
    console.print(panel)


def render_clean_preview(console: Console, data: dict[str, Any]) -> None:
    """Renders a dry-run preview of planned cleaning transformations — no data is changed yet."""
    changes = data.get("pending_changes", [])
    shape = data.get("initial_shape", [0, 0])

    header = Text()
    header.append("⚠  Dry-Run Preview — No data has been changed yet\n\n", style="bold yellow")
    header.append(f"Dataset shape: {shape[0]:,} rows × {shape[1]:,} columns\n", style="dim white")
    header.append(f"Planned transformations: {len(changes)}\n", style="dim white")

    console.print(Panel(header, title="[bold yellow]▲ Clean Dataset — Preview[/bold yellow]", border_style="yellow", padding=(0, 2), expand=False))

    if not changes:
        console.print("[bold green]✓ Dataset already clean — no transformations needed.[/bold green]\n")
        return

    table = Table(header_style="bold bright_cyan", border_style="yellow", show_lines=True)
    table.add_column("#", style="dim white", width=3)
    table.add_column("Transformation", style="bold cyan")
    table.add_column("Impact", style="white")

    for i, change in enumerate(changes, 1):
        table.add_row(
            str(i),
            change.get("description", ""),
            change.get("impact", ""),
        )
    console.print(table)


def render_clean_result(console: Console, data: dict[str, Any]) -> None:
    """Renders dataset transformation and cleaning results."""
    text = Text()
    msg = data.get("message") or data.get("transformation") or "Data transformation applied."
    text.append(f"✓ {msg}\n\n", style="bold green")

    if "initial_shape" in data and "final_shape" in data:
        init_r, init_c = data["initial_shape"]
        fin_r, fin_c = data["final_shape"]
        text.append(f"Dimensions: {init_r:,} × {init_c:,}  →  ", style="dim white")
        text.append(f"{fin_r:,} × {fin_c:,}\n\n", style="bold bright_white")

    transformations = data.get("transformations_applied", [])
    if transformations:
        text.append("Transformations Applied:\n", style="bold bright_cyan")
        for t in transformations:
            text.append(f"  • {t}\n", style="white")

    panel = Panel(text, title="[bold green]▲ Data Transformation & Cleaning[/bold green]", border_style="green", padding=(0, 2), expand=False)
    console.print(panel)


def render_ml_result(console: Console, data: dict[str, Any]) -> None:
    """Renders model training & evaluation metrics."""
    text = Text()
    text.append(f"✓ {data.get('message', 'Model trained successfully.')}\n\n", style="bold green")

    text.append("Model:     ", style="dim white")
    text.append(f"{data.get('model_name', 'Model')}\n", style="bold bright_cyan")
    text.append("Target:    ", style="dim white")
    text.append(f"{data.get('target_column')}\n", style="bold white")
    text.append("Task:      ", style="dim white")
    text.append(f"{data.get('problem_type', '').title()}\n\n", style="yellow")

    text.append("Evaluation Metrics:\n", style="bold bright_white")
    for m_name, val in data.get("metrics", {}).items():
        text.append(f"  • {m_name.replace('_', ' ').title()}: ", style="dim white")
        text.append(f"{val}\n", style="bold green")

    best_params = data.get("best_params", {})
    if best_params:
        text.append("\n🔧 Best Hyperparameters (Tuned):\n", style="bold bright_yellow")
        for param, val in best_params.items():
            text.append(f"  • {param}: ", style="dim white")
            text.append(f"{val}\n", style="bold yellow")

    panel = Panel(text, title="[bold bright_cyan]▲ Machine Learning Model Training[/bold bright_cyan]", border_style="cyan", padding=(0, 2), expand=False)
    console.print(panel)



def render_model_comparison(console: Console, data: dict[str, Any]) -> None:
    """Renders model comparison matrix and selection rationale."""
    best = data.get("best_model", "Best Model")
    rationale = data.get("selection_rationale", "")

    table = Table(title=f"[bold bright_white]▲ Model Performance Comparison (Best: {best})[/bold bright_white]", header_style="bold bright_cyan", border_style="bright_blue")
    table.add_column("Model Algorithm", style="bold cyan")
    table.add_column("Primary Metric", justify="right", style="bold yellow")
    table.add_column("Secondary Metric", justify="right", style="dim white")

    for item in data.get("comparison_matrix", []):
        m_name = item["model_name"]
        metrics = item.get("metrics", {})
        metric_str = ", ".join([f"{k}: {v}" for k, v in list(metrics.items())[:1]])
        sec_str = ", ".join([f"{k}: {v}" for k, v in list(metrics.items())[1:2]])
        style = "bold green" if m_name == best else "white"
        table.add_row(f"[{style}]{m_name}[/{style}]", f"[{style}]{metric_str}[/{style}]", sec_str)

    console.print(table)

    if rationale:
        panel = Panel(Text(rationale, style="italic yellow"), title="[bold green]▲ Model Selection Rationale[/bold green]", border_style="green", padding=(0, 2), expand=False)
        console.print(panel)

    suggestion = data.get("suggestion_note", "")
    if suggestion:
        console.print(Panel(Text(suggestion, style="dim bright_cyan"), title="[bold bright_cyan]▲ Next Steps — Model Selection Guide[/bold bright_cyan]", border_style="cyan", padding=(0, 2), expand=False))

    saved = data.get("saved_best_path", "")
    if saved:
        console.print(f"[dim green]  💾 Best model auto-saved: {saved}[/dim green]")



def render_predictions(console: Console, data: dict[str, Any]) -> None:
    """Renders sample predictions table."""
    table = Table(title=f"[bold bright_white]▲ Sample Predictions ({data.get('model_name')})[/bold bright_white]", header_style="bold bright_cyan", border_style="cyan")
    table.add_column("Sample Row", justify="right", style="dim white")
    table.add_column("Predicted Output", style="bold green")

    for item in data.get("predictions", []):
        table.add_row(str(item["row"]), str(item["prediction"]))

    console.print(table)


def render_plot_result(console: Console, data: dict[str, Any]) -> None:
    """Renders plot generation path panel."""
    text = Text()
    msg = data.get("message", "Plot generated successfully.")
    path = data.get("plot_path", "")

    text.append(f"✓ {msg}\n\n", style="bold green")
    text.append("Saved Image Artifact: ", style="dim white")
    text.append(f"{path}", style="bold bright_cyan")

    panel = Panel(text, title="[bold bright_cyan]▲ Visualization Artifact[/bold bright_cyan]", border_style="cyan", padding=(0, 2), expand=False)
    console.print(panel)


def render_report_result(console: Console, data: dict[str, Any]) -> None:
    """Renders report generation path panel."""
    text = Text()
    msg = data.get("message", "Report generated successfully.")
    path = data.get("report_path", "")

    text.append(f"✓ {msg}\n\n", style="bold green")
    text.append("Saved Report Artifact: ", style="dim white")
    text.append(f"{path}", style="bold bright_cyan")

    panel = Panel(text, title="[bold green]▲ Data Science Report Artifact[/bold green]", border_style="green", padding=(0, 2), expand=False)
    console.print(panel)


def render_custom_qa(console: Console, response: AgentResponse) -> None:
    """Renders customized AI Data Science reasoning and Q&A answers."""
    explanation_text = response.explanation or "No answer generated."
    panel = Panel(
        Text(explanation_text, style="bright_white"),
        title="[bold bright_cyan]▲ Data Science Agent :: Dynamic AI Response[/bold bright_cyan]",
        border_style="bright_cyan",
        padding=(1, 2),
        expand=False,
    )
    console.print(panel)


def render_agent_response(console: Console, response: AgentResponse) -> None:
    """Renders agent execution badge, plan steps, observations, and key findings."""
    if response.selected_tool == "custom_llm_qa":
        render_custom_qa(console, response)
        return

    if not response.plan or not response.plan.steps:
        render_error(console, "Unrecognized Request", response.explanation)
        return

    # Check if execution failed on step 1 due to missing dataset
    failed_step = next((s for s in response.plan.steps if s.result and not s.result.success), None)
    if failed_step and not response.plan.is_complete:
        render_error(
            console,
            "No Dataset Loaded",
            failed_step.result.error or "No dataset is currently loaded. Load a CSV dataset first using 'load <filepath>'."
        )
        return

    # If multi-step plan, render Plan panel first
    if response.plan.total_steps > 1:
        render_plan(console, response.plan)
        # Render step tool outputs
        for step in response.plan.steps:
            if step.completed and step.result and step.result.success:
                tool_name = step.tool_name
                data = step.result.data
                if tool_name == "audit_data_quality":
                    render_quality_audit(console, data)
                elif tool_name == "detect_missing_values":
                    render_missing_values(console, data)
                elif tool_name == "detect_duplicates":
                    render_duplicates(console, data)
                elif tool_name == "inspect_data_types":
                    render_data_types(console, data)
                elif tool_name == "analyze_distribution":
                    render_distribution(console, data)
                elif tool_name == "detect_outliers":
                    render_outliers(console, data)
                elif tool_name in ("describe_dataset", "calculate_statistics"):
                    render_statistics(console, data)
                elif tool_name == "calculate_correlation":
                    render_correlation(console, data)
                elif tool_name in ("clean_dataset", "handle_missing_values", "encode_features", "scale_features"):
                    render_clean_result(console, data)
                elif tool_name == "train_model":
                    render_ml_result(console, data)
                elif tool_name == "compare_models":
                    render_model_comparison(console, data)
                elif tool_name == "predict":
                    render_predictions(console, data)
                elif tool_name == "explain_model":
                    render_explain_result(console, data)
                elif tool_name in ("export_model", "load_model", "export_predictions"):
                    render_export_result(console, data)
                elif tool_name in ("plot_distribution", "plot_correlation_matrix", "plot_feature_importance"):
                    render_plot_result(console, data)
                elif tool_name == "generate_report":
                    render_report_result(console, data)

        if response.summary:
            render_plan_summary(console, response.summary)
    else:
        # 1-step plan execution
        step = response.plan.steps[0]
        text = Text()
        text.append("Intent Identified → ", style="dim white")
        text.append(f"Tool Selected: {step.tool_name}", style="bold bright_cyan")

        console.print(Panel(text, title="[bold bright_cyan]▲ Agent Reasoning[/bold bright_cyan]", border_style="cyan", padding=(0, 2), expand=False))

        if step.result and step.result.success:
            tool_name = step.tool_name
            data = step.result.data
            if tool_name == "audit_data_quality":
                render_quality_audit(console, data)
            elif tool_name == "detect_missing_values":
                render_missing_values(console, data)
            elif tool_name == "detect_duplicates":
                render_duplicates(console, data)
            elif tool_name == "inspect_data_types":
                render_data_types(console, data)
            elif tool_name == "analyze_distribution":
                render_distribution(console, data)
            elif tool_name == "detect_outliers":
                render_outliers(console, data)
            elif tool_name in ("describe_dataset", "calculate_statistics"):
                render_statistics(console, data)
            elif tool_name == "calculate_correlation":
                render_correlation(console, data)
            elif tool_name in ("clean_dataset", "handle_missing_values", "encode_features", "scale_features"):
                render_clean_result(console, data)
            elif tool_name == "train_model":
                render_ml_result(console, data)
            elif tool_name == "compare_models":
                render_model_comparison(console, data)
            elif tool_name == "predict":
                render_predictions(console, data)
            elif tool_name == "explain_model":
                render_explain_result(console, data)
            elif tool_name in ("export_model", "load_model", "export_predictions"):
                render_export_result(console, data)
            elif tool_name in ("plot_distribution", "plot_correlation_matrix", "plot_feature_importance"):
                render_plot_result(console, data)
            elif tool_name == "generate_report":
                render_report_result(console, data)
        elif step.result and not step.result.success:
            render_error(console, "Tool Execution Error", step.result.error or "Unknown tool failure.")


def render_help(console: Console, registry: Optional[ToolRegistry] = None) -> None:
    """Renders a dynamic, beautifully categorized interactive command menu."""
    from proton_agent.tools import default_registry

    reg = registry or default_registry

    categories: dict[str, list[tuple[str, str]]] = {
        "📁 Data Ingestion": [
            ("load <filepath>", "Load a CSV dataset into the active session (e.g., 'load data.csv')."),
        ],
        "🤖 Autonomous AI Planning & EDA": [
            ("perform complete eda", "Decomposes analysis into a 7-step plan & executes complete EDA."),
            ("analyze this dataset", "Runs full dataset analysis and summarizes key findings."),
        ],
        "🔍 Data Quality & Inspection": [
            ("inspect", "View dataset shape, columns, memory usage, missing counts & head preview."),
            ("quality", "Audit data health, calculate quality score (0-100%), and list warnings."),
            ("missing", "Detect missing values and calculate per-column missingness statistics."),
            ("duplicates", "Detect duplicate rows and calculate duplication rates."),
            ("dtypes", "Categorize columns into Numerical, Categorical, Boolean, and Datetime."),
        ],
        "📊 Statistical Analysis & EDA": [
            ("outliers", "Detect statistical outliers using Interquartile Range (IQR)."),
            ("distribution", "Analyze numerical distribution skewness/kurtosis and categorical frequencies."),
            ("stats", "Compute mean, std, median, min, max, 25%, 75%, and skewness per column."),
            ("describe", "Generate comprehensive descriptive statistics (pandas df.describe())."),
            ("correlation", "Compute numerical Pearson correlation matrix & rank top feature pairs."),
        ],
    }

    known_tool_names = {
        "load_dataset",
        "inspect_dataset",
        "audit_data_quality",
        "detect_missing_values",
        "detect_duplicates",
        "inspect_data_types",
        "analyze_distribution",
        "detect_outliers",
        "calculate_statistics",
        "describe_dataset",
        "calculate_correlation",
    }
    extra_tools = []
    for tool in reg.list_tools():
        if tool.name not in known_tool_names:
            extra_tools.append((tool.name, tool.description))

    if extra_tools:
        categories["🛠 Additional Registered Tools"] = extra_tools

    categories["💬 Natural Language AI Queries"] = [
        ("  • 'perform complete eda'", "Triggers multi-step autonomous planning & EDA"),
        ("  • 'audit data quality'", "Calculates health score & quality warnings"),
        ("  • 'detect outliers'", "Identifies numerical IQR statistical outliers"),
        ("  • 'show distribution shapes'", "Profiles numerical skewness & categorical counts"),
        ("  • 'calculate feature correlations'", "Triggers correlation matrix computation"),
    ]

    categories["⚙️ System Controls"] = [
        ("help", "Show this interactive command menu."),
        ("clear", "Clear terminal screen output."),
        ("exit / quit", "Exit the interactive proton session."),
    ]

    console.print()
    header_panel = Panel(
        Text("▲ PROTON DATA SCIENCE AGENT :: COMMAND MENU", style="bold bright_cyan", justify="center"),
        border_style="bright_blue",
        padding=(0, 2),
        expand=False,
    )
    console.print(header_panel)

    for cat_title, items in categories.items():
        table = Table(
            title=f"[bold bright_white]{cat_title}[/bold bright_white]",
            header_style="bold bright_cyan",
            border_style="cyan",
            show_lines=False,
            padding=(0, 2),
        )
        table.add_column("Command / Query", style="bold cyan", min_width=32)
        table.add_column("Description", style="dim white")

        for cmd, desc in items:
            table.add_row(cmd, desc)

        console.print(table)

    console.print()


def render_error(console: Console, title: str, message: str) -> None:
    """Renders user-friendly error formatting per Section 20 guidelines."""
    text = Text()
    text.append(f"✗ {title}\n\n", style="bold red")
    text.append(message)

    panel = Panel(text, title="[bold red]▲ Alert[/bold red]", border_style="red", padding=(1, 2))
    console.print(panel)


def render_explain_result(console: Console, data: dict[str, Any]) -> None:
    """Renders confusion matrix, ROC AUC, per-class metrics, SHAP summary."""
    console.print(Panel(
        Text(f"✓ {data.get('message', 'Explanation complete.')}", style="bold green"),
        title="[bold bright_cyan]▲ Model Explainability[/bold bright_cyan]",
        border_style="cyan", padding=(0, 2), expand=False,
    ))

    # Confusion matrix as table
    cm_data = data.get("confusion_matrix")
    if cm_data:
        labels = cm_data.get("labels", [])
        matrix = cm_data.get("matrix", [])
        cm_table = Table(title="▲ Confusion Matrix", header_style="bold bright_cyan", border_style="blue")
        cm_table.add_column("Actual \\ Predicted", style="bold cyan")
        for lbl in labels:
            cm_table.add_column(str(lbl), justify="right")
        for i, row in enumerate(matrix):
            cm_table.add_row(str(labels[i]), *[str(v) for v in row])
        console.print(cm_table)

    if data.get("confusion_matrix_plot"):
        console.print(f"[dim]  📊 Confusion matrix saved: {data['confusion_matrix_plot']}[/dim]")
    if data.get("roc_curve_plot"):
        console.print(f"[dim]  📈 ROC curve saved: {data['roc_curve_plot']}[/dim]")

    if "roc_auc" in data:
        console.print(f"[bold bright_cyan]  ROC AUC Score:[/bold bright_cyan] [bold white]{data['roc_auc']}[/bold white]")

    report = data.get("classification_report")
    if report:
        console.print(Panel(Text(report, style="dim white"), title="[bold]▲ Per-Class Metrics[/bold]", border_style="dim", expand=False))

    shap = data.get("shap_feature_importance", {})
    if shap and "note" not in shap and "error" not in shap:
        shap_table = Table(title="▲ SHAP Feature Importance (Top 10)", header_style="bold bright_cyan", border_style="cyan")
        shap_table.add_column("Feature", style="bold cyan")
        shap_table.add_column("Mean |SHAP|", justify="right", style="bold yellow")
        for feat, val in shap.items():
            shap_table.add_row(feat, f"{val:.4f}")
        console.print(shap_table)
    elif shap:
        msg = shap.get("note") or shap.get("error", "")
        console.print(f"[dim yellow]  ℹ SHAP: {msg}[/dim yellow]")

    residuals = data.get("residuals_summary")
    if residuals:
        text = Text()
        text.append("Residuals Summary:\n", style="bold bright_white")
        text.append(f"  Mean Residual: {residuals.get('mean_residual', 'N/A')}\n", style="white")
        text.append(f"  Std Residual:  {residuals.get('std_residual', 'N/A')}\n", style="white")
        text.append(f"  Max |Error|:   {residuals.get('max_error', 'N/A')}\n", style="white")
        console.print(Panel(text, title="[bold]▲ Regression Diagnostics[/bold]", border_style="dim", expand=False))


def render_export_result(console: Console, data: dict[str, Any]) -> None:
    """Renders result for export model / export predictions / load model commands."""
    text = Text()
    text.append(f"{data.get('message', '✓ Export complete.')}\n", style="bold green")

    for key, label in [
        ("saved_path", "Saved to"),
        ("loaded_path", "Loaded from"),
        ("model_name", "Model"),
        ("target_column", "Target Column"),
        ("problem_type", "Task"),
        ("rows", "Rows exported"),
        ("prediction_column", "Prediction Column"),
    ]:
        if key in data and data[key] is not None:
            text.append(f"  {label}: ", style="dim white")
            text.append(f"{data[key]}\n", style="bold bright_cyan")

    console.print(Panel(text, title="[bold bright_cyan]▲ Export / Persistence[/bold bright_cyan]", border_style="cyan", padding=(0, 2), expand=False))


def render_config_panel(console: Console, session: Any, settings: Any) -> None:
    """Renders active configuration and session status."""
    api_key = getattr(settings, "gemini_api_key", "") or ""
    key_status = f"●  Active (…{api_key[-6:]})" if len(api_key) > 6 else ("●  Active" if api_key else "○  Not configured")

    text = Text()
    text.append("LLM Provider:    ", style="dim white")
    text.append(f"{getattr(settings, 'llm_provider', 'gemini').title()}\n", style="bold bright_cyan")
    text.append("Model:           ", style="dim white")
    text.append(f"{getattr(settings, 'gemini_model', 'N/A')}\n", style="bold white")
    text.append("Temperature:     ", style="dim white")
    text.append(f"{getattr(settings, 'temperature', 0.2)}\n", style="white")
    text.append("API Key:         ", style="dim white")
    text.append(f"{key_status}\n\n", style="bold green" if api_key else "bold red")

    text.append("Train Dataset:   ", style="dim white")
    if session.current_dataset:
        text.append(f"{session.current_dataset.name} ({session.current_dataset.rows:,} rows)\n", style="bold bright_cyan")
    else:
        text.append("None\n", style="dim white")

    text.append("Test Dataset:    ", style="dim white")
    if session.test_dataset:
        text.append(f"{session.test_dataset.name} ({session.test_dataset.rows:,} rows)\n", style="bold cyan")
    else:
        text.append("None\n", style="dim white")

    text.append("Active Model:    ", style="dim white")
    if session.current_model:
        model_name = session.current_model.get("name", "Unknown") if isinstance(session.current_model, dict) else "Loaded"
        text.append(f"{model_name}\n", style="bold yellow")
    else:
        text.append("None\n", style="dim white")

    text.append("\nOutput Dirs:     ", style="dim white")
    text.append(f"{getattr(settings, 'output_dir', 'outputs/')}\n", style="dim white")
    text.append("Session History: ", style="dim white")
    text.append(f"{len(session.history)} command(s)\n", style="white")

    console.print(Panel(text, title="[bold bright_cyan]▲ PROTON Configuration & Session Status[/bold bright_cyan]", border_style="cyan", padding=(0, 2), expand=False))
