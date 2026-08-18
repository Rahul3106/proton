"""
Agent: core orchestrator for the RECEIVE -> PLAN -> EXECUTE -> OBSERVE -> SUMMARIZE loop.
"""
from __future__ import annotations

from typing import Optional

from proton_agent.agent.executor import Executor
from proton_agent.agent.llm import LLMClient, get_default_llm_client
from proton_agent.agent.planner import Planner
from proton_agent.agent.state import AgentResponse, Plan
from proton_agent.context.session import SessionState
from proton_agent.tools import ToolRegistry, default_registry


class Agent:
    """Orchestrates natural language intent understanding, multi-step planning, and step-by-step tool execution."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        registry: Optional[ToolRegistry] = None,
    ) -> None:
        self.llm_client = llm_client or get_default_llm_client()
        self.registry = registry or default_registry
        self.planner = Planner(llm_client=self.llm_client, registry=self.registry)
        self.executor = Executor(registry=self.registry)

    def process_request(self, user_request: str, session: SessionState) -> AgentResponse:
        """Constructs and executes a Plan step-by-step, returning an AgentResponse with the executed Plan."""
        plan = self.planner.create_plan(user_request)

        if not plan.steps:
            return self._handle_custom_question(user_request, session)

        last_result = None
        selected_tool = None

        for step in plan.steps:
            selected_tool = step.tool_name
            step.result = self.executor.execute_tool(step.tool_name, session, user_request=user_request, **step.tool_kwargs)
            if step.result.success:
                step.completed = True
                last_result = step.result
            else:
                # Stop plan execution if a step fails
                break

        plan.is_complete = all(step.completed for step in plan.steps)

        summary_findings = self._summarize_plan_findings(plan)

        failed_step = next((s for s in plan.steps if s.result and not s.result.success), None)
        if failed_step:
            diagnosis = self._diagnose_failure(failed_step, session, user_request)
            if diagnosis:
                summary_findings = (summary_findings + f"\n\n🤖 AI Diagnostic & Recommended Action:\n{diagnosis}").strip()

        explanation = (
            f"Successfully executed {plan.completed_steps}/{plan.total_steps} plan step(s)."
            if plan.is_complete
            else f"Executed {plan.completed_steps}/{plan.total_steps} step(s). Encountered issue during step execution."
        )

        return AgentResponse(
            user_request=user_request,
            selected_tool=selected_tool,
            tool_result=last_result,
            explanation=explanation,
            plan=plan,
            summary=summary_findings,
        )

    def _diagnose_failure(self, step: PlanStep, session: SessionState, user_request: str) -> str:
        """Uses LLM reasoning to diagnose why a tool or step failed and provide actionable advice."""
        err_msg = step.result.error if step.result else "Unknown execution failure"
        ds_info = "No dataset loaded"
        if session.current_dataset and session.current_dataset.df is not None:
            ds = session.current_dataset
            ds_info = (
                f"Dataset: '{ds.name}' ({ds.rows:,} rows × {ds.columns_count} cols)\n"
                f"Columns: {', '.join(ds.columns[:25])}\n"
                f"Data types: {ds.dtypes}\n"
                f"Target column: {session.target_column or 'None selected'}"
            )

        prompt = (
            f"User Goal: \"{user_request}\"\n"
            f"Failed Step: \"{step.description}\" (Tool: '{step.tool_name}')\n"
            f"Execution Error: {err_msg}\n\n"
            f"Active Dataset Context:\n{ds_info}\n\n"
            "Diagnose why this occurred (e.g., target column data type mismatch, metadata URL column mistakenly chosen, unhandled missing values, or wrong model type). "
            "Give a concise, 2-3 sentence explanation of the root cause and advise the user on the exact command or model to select next."
        )

        try:
            return self.llm_client.generate_response(prompt)
        except Exception:
            return f"Step '{step.description}' failed: {err_msg}. Please verify your target column and feature data types."

    def _summarize_plan_findings(self, plan: Plan) -> str:
        """Aggregates key findings from completed plan steps."""
        findings = []

        for step in plan.steps:
            if not step.completed or not step.result or not step.result.success:
                continue

            data = step.result.data
            tool = step.tool_name

            if tool == "inspect_dataset":
                findings.append(f"• Dataset Overview: {data.get('rows', 0):,} rows × {data.get('columns_count', 0):,} columns ({data.get('memory_kb', 0)} KB)")
            elif tool == "audit_data_quality":
                score = data.get("quality_score", 100.0)
                status = data.get("status", "Good")
                findings.append(f"• Data Quality Score: {score}% ({status})")
            elif tool == "detect_missing_values":
                tot = data.get("total_missing", 0)
                if tot == 0:
                    findings.append("• Missing Data: No missing values detected")
                else:
                    findings.append(f"• Missing Data: {tot:,} missing value(s) across {data.get('columns_with_missing_count', 0)} column(s) ({data.get('overall_missing_pct', 0)}%)")
            elif tool == "detect_duplicates":
                dups = data.get("duplicate_rows", 0)
                if dups == 0:
                    findings.append("• Duplicates: No duplicate rows found")
                else:
                    findings.append(f"• Duplicates: {dups:,} duplicate row(s) detected ({data.get('duplicate_pct', 0)}%)")
            elif tool == "inspect_data_types":
                num = len(data.get("numerical_columns", []))
                cat = len(data.get("categorical_columns", []))
                findings.append(f"• Column Categorization: {num} numerical, {cat} categorical feature(s)")
            elif tool == "detect_outliers":
                total_out = data.get("total_outliers", 0)
                aff = len(data.get("affected_columns", []))
                if total_out == 0:
                    findings.append("• Outliers: No IQR outliers detected")
                else:
                    findings.append(f"• Outliers: {total_out:,} IQR outlier value(s) across {aff} numerical column(s)")
            elif tool == "analyze_distribution":
                num_dist = data.get("numerical_distributions", {})
                skewed = [c for c, d in num_dist.items() if "Symmetric" not in d.get("shape", "")]
                if skewed:
                    findings.append(f"• Distribution Skewness: {len(skewed)} column(s) display non-symmetric skewness ({', '.join(skewed[:3])})")
            elif tool == "calculate_correlation":
                top = data.get("top_correlations", [])
                if top:
                    strongest = top[0]
                    findings.append(f"• Strongest Correlation: {strongest['feature_1']} & {strongest['feature_2']} ({strongest['correlation']:+.4f})")

        return "\n".join(findings)

    def _handle_custom_question(self, user_request: str, session: SessionState) -> AgentResponse:
        """Handles custom/open-ended user questions using LLM reasoning over active dataset context."""
        from proton_agent.agent.state import PlanStep

        context_parts = [
            f"User Question: \"{user_request}\"",
            "You are an expert Data Science Agent assisting a user with their dataset.",
        ]

        if session.current_dataset:
            ds = session.current_dataset
            context_parts.extend([
                "\n--- ACTIVE DATASET CONTEXT ---",
                f"Dataset Name: {ds.name}",
                f"Dimensions: {ds.rows:,} rows × {ds.columns_count:,} columns",
                f"Columns ({len(ds.columns)}): {', '.join(ds.columns[:25])}",
                f"Data Types: {ds.dtypes}",
                f"Missing Values: {ds.total_missing:,} total missing cells",
            ])
            if session.target_column:
                context_parts.append(f"Target Column: {session.target_column}")
            if session.current_model:
                m = session.current_model
                context_parts.append(f"Active Model: {m['name']} (Metrics: {m['metrics']})")
            if session.transformations:
                context_parts.append(f"Applied Transformations: {session.transformations}")
        else:
            context_parts.append("\nNote: No dataset is currently loaded in this session.")

        context_parts.append("\nPlease provide a direct, concise, and helpful Data Science answer to the user's question.")
        prompt = "\n".join(context_parts)

        answer = self.llm_client.generate_response(prompt)

        custom_step = PlanStep(step_number=1, description="Custom LLM Data Science Reasoning", tool_name="custom_llm_qa")
        custom_step.completed = True

        return AgentResponse(
            user_request=user_request,
            selected_tool="custom_llm_qa",
            tool_result=None,
            explanation=answer,
            plan=Plan(goal=user_request, steps=[custom_step], is_complete=True),
            summary="",
        )
