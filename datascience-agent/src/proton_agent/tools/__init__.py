"""
Data Science Tools module.
Registers all Phase 4 & Phase 7 tools into default_registry.
"""
from proton_agent.tools.analysis import (
    AnalyzeDistributionTool,
    AuditDataQualityTool,
    CalculateCorrelationTool,
    CalculateStatisticsTool,
    DescribeDatasetTool,
    DetectOutliersTool,
    DriftDetectionTool,
)
from proton_agent.tools.base import BaseTool, ToolResult
from proton_agent.tools.data import (
    DetectDuplicatesTool,
    DetectMissingValuesTool,
    InspectDataTypesTool,
    InspectDatasetTool,
    LoadDatasetTool,
)
from proton_agent.tools.modeling import (
    CompareModelsTool,
    DetectProblemTypeTool,
    ExplainModelTool,
    ExplainPredictionTool,
    ExportModelTool,
    ExportPredictionsTool,
    LoadModelTool,
    PredictTool,
    TrainModelTool,
)
from proton_agent.tools.preprocessing import (
    CleanDatasetTool,
    EncodeFeaturesTool,
    HandleMissingValuesTool,
    ScaleFeaturesTool,
)
from proton_agent.tools.registry import ToolRegistry, default_registry
from proton_agent.tools.reporting import GenerateReportTool
from proton_agent.tools.visualization import (
    PlotCorrelationMatrixTool,
    PlotDistributionTool,
    PlotFeatureImportanceTool,
)

# Register Phase 4, Phase 7, Phase 8, Phase 9, Phase 11, Phase 12, and Phase F tools
for tool_cls in [
    LoadDatasetTool,
    InspectDatasetTool,
    DetectMissingValuesTool,
    DetectDuplicatesTool,
    InspectDataTypesTool,
    DescribeDatasetTool,
    CalculateStatisticsTool,
    CalculateCorrelationTool,
    DetectOutliersTool,
    AnalyzeDistributionTool,
    AuditDataQualityTool,
    DriftDetectionTool,
    CleanDatasetTool,
    HandleMissingValuesTool,
    EncodeFeaturesTool,
    ScaleFeaturesTool,
    DetectProblemTypeTool,
    TrainModelTool,
    CompareModelsTool,
    PredictTool,
    ExplainModelTool,
    ExplainPredictionTool,
    ExportModelTool,
    LoadModelTool,
    ExportPredictionsTool,
    PlotDistributionTool,
    PlotCorrelationMatrixTool,
    PlotFeatureImportanceTool,
    GenerateReportTool,
]:
    default_registry.register(tool_cls())

__all__ = [
    "AnalyzeDistributionTool",
    "AuditDataQualityTool",
    "BaseTool",
    "CalculateCorrelationTool",
    "CalculateStatisticsTool",
    "CleanDatasetTool",
    "CompareModelsTool",
    "DescribeDatasetTool",
    "DetectDuplicatesTool",
    "DetectMissingValuesTool",
    "DetectOutliersTool",
    "DetectProblemTypeTool",
    "DriftDetectionTool",
    "EncodeFeaturesTool",
    "ExplainModelTool",
    "ExplainPredictionTool",
    "ExportModelTool",
    "ExportPredictionsTool",
    "GenerateReportTool",
    "HandleMissingValuesTool",
    "InspectDataTypesTool",
    "InspectDatasetTool",
    "LoadDatasetTool",
    "LoadModelTool",
    "PlotCorrelationMatrixTool",
    "PlotDistributionTool",
    "PlotFeatureImportanceTool",
    "PredictTool",
    "ScaleFeaturesTool",
    "ToolRegistry",
    "ToolResult",
    "TrainModelTool",
    "default_registry",
]



