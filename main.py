"""
================================================================================
EduPulse Analytics: Student Academic Performance & Risk Intelligence System
================================================================================
MCA Semester 1 - Mini-Project Desktop Application
Tech Stack: Python 3, Tkinter (ttk), Pandas, NumPy, Matplotlib (FigureCanvasTkAgg)

Architecture:
- AcademicDataEngine: Vectorized data cleaning, feature engineering, and analytics.
- StatCard: Modern executive KPI status card with accent indicators.
- FilterPanel: Multi-dimensional interactive filtering sidebar.
- DataTable: Zebra-striped sortable Treeview data grid with conditional risk alerts.
- AnalyticsChartsView: Dynamic multi-subplot Matplotlib canvas embedded via TkAgg.
- EduPulseApp: Root window controller and application state orchestrator.
================================================================================
"""

from dataclasses import dataclass
from datetime import datetime
import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Tkinter GUI toolkit
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Matplotlib embedded backend
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.ticker as ticker


# ============================================================================
# 1. DESIGN TOKENS & SYSTEM SCHEMAS
# ============================================================================

@dataclass(frozen=True)
class UIColors:
    """Modern Enterprise UI Color Palette."""
    BG_WINDOW: str = "#F8FAFC"        # Light slate window background
    BG_CARD: str = "#FFFFFF"          # Crisp white card surface
    BG_PANEL: str = "#F1F5F9"         # Subtle gray for toolbars / controls
    BORDER_LIGHT: str = "#E2E8F0"     # Elegant light border tone
    
    TEXT_PRIMARY: str = "#0F172A"     # High-contrast deep slate
    TEXT_SECONDARY: str = "#475569"   # Muted body slate
    TEXT_MUTED: str = "#94A3B8"       # Subtle placeholder/caption text
    
    ACCENT_PRIMARY: str = "#2563EB"   # Royal Blue primary accent
    ACCENT_HOVER: str = "#1D4ED8"     # Deep Blue hover state
    
    RISK_HIGH: str = "#DC2626"        # Crimson Red (Critical Risk)
    RISK_HIGH_BG: str = "#FEE2E2"     # Soft Red row/badge highlight
    
    RISK_MODERATE: str = "#D97706"    # Amber Gold (Moderate Risk)
    RISK_MODERATE_BG: str = "#FEF3C7" # Soft Amber badge highlight
    
    RISK_LOW: str = "#059669"         # Emerald Green (Low Risk / Thriving)
    RISK_LOW_BG: str = "#D1FAE5"      # Soft Emerald badge highlight
    
    CHART_GRID: str = "#E2E8F0"       # Subtle chart grid line color


@dataclass(frozen=True)
class AnalyticsThresholds:
    """Institutional Early Warning Risk Benchmarks."""
    ATTENDANCE_MIN_BENCHMARK: float = 75.0   # 75% institutional attendance rule
    GPA_PROBATION_LIMIT: float = 2.00        # Academic probation cutoff
    GPA_HONORS_THRESHOLD: float = 3.75       # Dean's List / Honors cutoff
    ASSIGNMENT_MIN_RATE: float = 75.0        # Minimum assignment submission rate %
    
    # Composite Risk Index Boundaries (0 - 100 Scale)
    RISK_SCORE_HIGH_CUTOFF: float = 55.0     # Scores >= 55 are High Risk
    RISK_SCORE_MOD_CUTOFF: float = 30.0      # Scores 30 to < 55 are Moderate Risk


# Mandatory columns required by the data engine
REQUIRED_COLUMNS: List[str] = [
    "student_id",
    "student_name",
    "department",
    "semester",
    "attendance_rate",
    "study_hours_weekly",
    "assignment_completion_rate",
    "midterm_score",
    "quiz_average",
    "previous_gpa",
    "current_gpa"
]

# Column display mapping for Treeview table
COLUMN_DISPLAY_NAMES: Dict[str, str] = {
    "student_id": "Student ID",
    "student_name": "Student Name",
    "department": "Department",
    "semester": "Semester",
    "attendance_rate": "Attendance (%)",
    "study_hours_weekly": "Study Hrs/Wk",
    "assignment_completion_rate": "Assign. (%)",
    "midterm_score": "Midterm",
    "quiz_average": "Quiz Avg",
    "current_gpa": "GPA",
    "risk_score": "Risk Score",
    "risk_level": "Risk Category",
    "academic_status": "Academic Standing"
}


# ============================================================================
# 2. DATA PROCESSING ENGINE (PANDAS & NUMPY)
# ============================================================================

class DataEngineError(Exception):
    """Base exception for all DataEngine operational failures."""
    pass


class SchemaValidationError(DataEngineError):
    """Raised when an ingested CSV fails to meet required column specifications."""
    pass


class EmptyDatasetError(DataEngineError):
    """Raised when an ingested CSV contains no usable records."""
    pass


class AcademicDataEngine:
    """
    Core data engine implementing schema verification, vectorized data cleaning,
    composite risk scoring, multi-criteria filtering, and statistical aggregations.
    """

    def __init__(self, thresholds: Optional[AnalyticsThresholds] = None) -> None:
        self.thresholds = thresholds or AnalyticsThresholds()
        self._raw_df: Optional[pd.DataFrame] = None
        self._processed_df: Optional[pd.DataFrame] = None
        self._filtered_df: Optional[pd.DataFrame] = None

    @property
    def has_data(self) -> bool:
        """Check whether valid data is currently in memory."""
        return self._processed_df is not None and not self._processed_df.empty

    @property
    def full_data(self) -> pd.DataFrame:
        """Return a copy of the fully processed dataset."""
        if self._processed_df is None:
            return pd.DataFrame()
        return self._processed_df.copy()

    @property
    def current_data(self) -> pd.DataFrame:
        """Return the active dataset respecting applied filters."""
        if self._filtered_df is not None:
            return self._filtered_df.copy()
        if self._processed_df is not None:
            return self._processed_df.copy()
        return pd.DataFrame()

    def load_csv(self, file_path: str) -> Tuple[int, int]:
        """
        Load, validate, and process a CSV dataset from disk.

        Args:
            file_path: Path to the CSV file.

        Returns:
            Tuple of (total_raw_rows, clean_processed_rows).

        Raises:
            EmptyDatasetError: If file is empty.
            SchemaValidationError: If required columns are missing.
            DataEngineError: If parsing or I/O fails.
        """
        try:
            raw = pd.read_csv(file_path, skipinitialspace=True)
        except pd.errors.EmptyDataError:
            raise EmptyDatasetError("The selected CSV file is empty and contains no readable data.")
        except UnicodeDecodeError:
            try:
                raw = pd.read_csv(file_path, encoding="latin1", skipinitialspace=True)
            except pd.errors.EmptyDataError:
                raise EmptyDatasetError("The selected CSV file is empty and contains no readable data.")
            except Exception as exc:
                raise DataEngineError(f"File encoding error while reading '{file_path}': {exc}") from exc
        except Exception as exc:
            raise DataEngineError(f"Failed to read CSV file: {exc}") from exc

        if raw.empty:
            raise EmptyDatasetError("The selected CSV contains 0 rows.")

        # Normalize column headers
        raw.columns = [str(c).strip().lower() for c in raw.columns]

        # Verify presence of mandatory schema columns
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
        if missing_cols:
            raise SchemaValidationError(
                "Dataset is missing required columns:\n• " + "\n• ".join(missing_cols)
            )

        self._raw_df = raw
        self._processed_df = self._clean_and_engineer_features(raw)
        self._filtered_df = self._processed_df.copy()

        return len(raw), len(self._processed_df)

    def _clean_and_engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute vectorized cleaning, type coercion, and risk metric engineering.
        Uses NumPy vectorized operations for maximum performance.
        """
        df = df.copy()

        # Categorical and Identifier sanitation
        df["student_id"] = df["student_id"].astype(str).str.strip()
        df["student_name"] = df["student_name"].astype(str).str.strip()
        df["department"] = df["department"].fillna("General Studies").astype(str).str.strip()
        df["semester"] = df["semester"].fillna("Sem 1").astype(str).str.strip()

        # Numeric Columns list
        num_cols = [
            "attendance_rate",
            "study_hours_weekly",
            "assignment_completion_rate",
            "midterm_score",
            "quiz_average",
            "previous_gpa",
            "current_gpa"
        ]

        # Vectorized numeric coercion with column median fallback
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)

        # Enforce realistic domain boundaries via NumPy vector clipping
        df["attendance_rate"] = np.clip(df["attendance_rate"].to_numpy(), 0.0, 100.0)
        df["assignment_completion_rate"] = np.clip(df["assignment_completion_rate"].to_numpy(), 0.0, 100.0)
        df["midterm_score"] = np.clip(df["midterm_score"].to_numpy(), 0.0, 100.0)
        df["quiz_average"] = np.clip(df["quiz_average"].to_numpy(), 0.0, 100.0)
        df["study_hours_weekly"] = np.clip(df["study_hours_weekly"].to_numpy(), 0.0, 80.0)
        df["previous_gpa"] = np.clip(df["previous_gpa"].to_numpy(), 0.0, 4.0)
        df["current_gpa"] = np.clip(df["current_gpa"].to_numpy(), 0.0, 4.0)

        # -------------------------------------------------------------
        # Vectorized Academic Risk Score Calculation (0 - 100 Scale)
        # -------------------------------------------------------------
        # 1. Attendance penalty (0 to 40 pts)
        att = df["attendance_rate"].to_numpy()
        att_penalty = np.where(
            att < self.thresholds.ATTENDANCE_MIN_BENCHMARK,
            (self.thresholds.ATTENDANCE_MIN_BENCHMARK - att) * 1.2,
            np.where(att < 85.0, (85.0 - att) * 0.4, 0.0)
        )

        # 2. Exam Deficit penalty (0 to 35 pts)
        composite_exam = (df["midterm_score"].to_numpy() * 0.5) + (df["quiz_average"].to_numpy() * 0.5)
        exam_penalty = np.clip((85.0 - composite_exam) * 0.6, 0.0, 35.0)

        # 3. GPA Deficit penalty (0 to 30 pts)
        gpa = df["current_gpa"].to_numpy()
        gpa_penalty = np.clip((3.5 - gpa) * 12.0, 0.0, 30.0)

        # 4. Assignment Completion Deficit (0 to 15 pts)
        assign = df["assignment_completion_rate"].to_numpy()
        assign_penalty = np.where(
            assign < self.thresholds.ASSIGNMENT_MIN_RATE,
            (self.thresholds.ASSIGNMENT_MIN_RATE - assign) * 0.4,
            0.0
        )

        # Aggregate raw score and clamp to [0, 100]
        raw_risk = att_penalty + exam_penalty + gpa_penalty + assign_penalty
        risk_score = np.round(np.clip(raw_risk, 0.0, 100.0), 1)
        df["risk_score"] = risk_score

        # Vectorized Risk Categorization
        is_high = (risk_score >= self.thresholds.RISK_SCORE_HIGH_CUTOFF) | (att < 60.0) | (gpa < self.thresholds.GPA_PROBATION_LIMIT)
        is_mod = (~is_high) & ((risk_score >= self.thresholds.RISK_SCORE_MOD_CUTOFF) | (att < self.thresholds.ATTENDANCE_MIN_BENCHMARK) | (gpa < 2.5))

        risk_conditions = [is_high, is_mod]
        risk_choices = ["High Risk", "Moderate Risk"]
        df["risk_level"] = np.select(risk_conditions, risk_choices, default="Low Risk")

        # Vectorized Academic Standing categorization
        standing_conditions = [
            gpa >= self.thresholds.GPA_HONORS_THRESHOLD,
            gpa < self.thresholds.GPA_PROBATION_LIMIT
        ]
        standing_choices = ["Dean's List", "Probation"]
        df["academic_status"] = np.select(standing_conditions, standing_choices, default="Good Standing")

        # Rounding for UI presentation
        df["attendance_rate"] = np.round(df["attendance_rate"], 1)
        df["assignment_completion_rate"] = np.round(df["assignment_completion_rate"], 1)
        df["midterm_score"] = np.round(df["midterm_score"], 1)
        df["quiz_average"] = np.round(df["quiz_average"], 1)
        df["study_hours_weekly"] = np.round(df["study_hours_weekly"], 1)
        df["current_gpa"] = np.round(df["current_gpa"], 2)
        df["previous_gpa"] = np.round(df["previous_gpa"], 2)

        return df

    def apply_filters(
        self,
        department: Optional[str] = None,
        risk_level: Optional[str] = None,
        min_attendance: Optional[float] = None,
        search_query: Optional[str] = None
    ) -> pd.DataFrame:
        """Apply compound filtering across dataset using vectorized masks."""
        if self._processed_df is None:
            return pd.DataFrame()

        df = self._processed_df
        mask = np.ones(len(df), dtype=bool)

        if department and department.strip() and department != "All Departments":
            mask &= (df["department"].values == department.strip())

        if risk_level and risk_level.strip() and risk_level != "All Risk Tiers":
            mask &= (df["risk_level"].values == risk_level.strip())

        if min_attendance is not None and min_attendance > 0:
            mask &= (df["attendance_rate"].values >= min_attendance)

        if search_query and search_query.strip():
            query = search_query.strip().lower()
            id_match = df["student_id"].str.lower().str.contains(query, na=False).values
            name_match = df["student_name"].str.lower().str.contains(query, na=False).values
            mask &= (id_match | name_match)

        self._filtered_df = df[mask].copy()
        return self._filtered_df

    def get_summary_kpis(self) -> Dict[str, float]:
        """Compute aggregate summary KPIs from currently filtered data."""
        df = self.current_data
        if df.empty:
            return {
                "total_students": 0,
                "avg_gpa": 0.0,
                "high_risk_count": 0,
                "moderate_risk_count": 0,
                "total_at_risk": 0,
                "avg_attendance": 0.0,
                "avg_study_hours": 0.0
            }

        high_risk_count = int(np.sum(df["risk_level"].values == "High Risk"))
        moderate_risk_count = int(np.sum(df["risk_level"].values == "Moderate Risk"))

        return {
            "total_students": int(len(df)),
            "avg_gpa": float(np.round(df["current_gpa"].mean(), 2)),
            "high_risk_count": high_risk_count,
            "moderate_risk_count": moderate_risk_count,
            "total_at_risk": high_risk_count + moderate_risk_count,
            "avg_attendance": float(np.round(df["attendance_rate"].mean(), 1)),
            "avg_study_hours": float(np.round(df["study_hours_weekly"].mean(), 1))
        }

    def get_departments_list(self) -> List[str]:
        """Return unique sorted department list."""
        if self._processed_df is None or self._processed_df.empty:
            return []
        return sorted(self._processed_df["department"].dropna().unique().tolist())


# ============================================================================
# 3. REUSABLE UI COMPONENTS (TKINTER / TTK)
# ============================================================================

class StatCard(tk.Frame):
    """Modern executive KPI card with accent stripe, large metric, and subtitle."""

    def __init__(
        self,
        parent: tk.Widget,
        title: str,
        initial_value: str = "0",
        subtitle: str = "",
        accent_color: str = UIColors.ACCENT_PRIMARY,
        **kwargs
    ) -> None:
        super().__init__(
            parent,
            bg=UIColors.BG_CARD,
            highlightbackground=UIColors.BORDER_LIGHT,
            highlightthickness=1,
            padx=14,
            pady=10,
            **kwargs
        )
        # Left accent stripe
        self._stripe = tk.Frame(self, bg=accent_color, width=4)
        self._stripe.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        content_frame = tk.Frame(self, bg=UIColors.BG_CARD)
        content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._lbl_title = tk.Label(
            content_frame,
            text=title.upper(),
            font=("Segoe UI", 8, "bold"),
            fg=UIColors.TEXT_MUTED,
            bg=UIColors.BG_CARD,
            anchor="w"
        )
        self._lbl_title.pack(fill=tk.X)

        self._lbl_value = tk.Label(
            content_frame,
            text=initial_value,
            font=("Segoe UI", 18, "bold"),
            fg=UIColors.TEXT_PRIMARY,
            bg=UIColors.BG_CARD,
            anchor="w"
        )
        self._lbl_value.pack(fill=tk.X, pady=(2, 0))

        self._lbl_sub = tk.Label(
            content_frame,
            text=subtitle,
            font=("Segoe UI", 8),
            fg=UIColors.TEXT_SECONDARY,
            bg=UIColors.BG_CARD,
            anchor="w"
        )
        self._lbl_sub.pack(fill=tk.X)

    def update_value(self, value: str, subtitle: Optional[str] = None) -> None:
        """Update displayed metric and optional caption."""
        self._lbl_value.config(text=value)
        if subtitle is not None:
            self._lbl_sub.config(text=subtitle)


class FilterPanel(ttk.LabelFrame):
    """Interactive multi-attribute filter sidebar."""

    def __init__(
        self,
        parent: tk.Widget,
        on_filter_changed: Callable[[], None],
        on_reset: Callable[[], None],
        **kwargs
    ) -> None:
        super().__init__(parent, text="  Filter & Search Controls  ", padding=12, **kwargs)
        self.on_filter_changed = on_filter_changed
        self.on_reset = on_reset

        self._var_dept = tk.StringVar(value="All Departments")
        self._var_risk = tk.StringVar(value="All Risk Tiers")
        self._var_att_min = tk.DoubleVar(value=0.0)
        self._var_search = tk.StringVar(value="")

        self._create_widgets()

    def _create_widgets(self) -> None:
        # Search Entry
        ttk.Label(self, text="Search (ID / Name):", font=("Segoe UI", 9, "bold")).pack(fill=tk.X, pady=(2, 2))
        self.entry_search = ttk.Entry(self, textvariable=self._var_search)
        self.entry_search.pack(fill=tk.X, pady=(0, 10))
        self.entry_search.bind("<Return>", lambda event: self.on_filter_changed())

        # Department Dropdown
        ttk.Label(self, text="Department:", font=("Segoe UI", 9, "bold")).pack(fill=tk.X, pady=(2, 2))
        self.combo_dept = ttk.Combobox(self, textvariable=self._var_dept, state="readonly", values=["All Departments"])
        self.combo_dept.pack(fill=tk.X, pady=(0, 10))
        self.combo_dept.bind("<<ComboboxSelected>>", lambda event: self.on_filter_changed())

        # Risk Level Dropdown
        ttk.Label(self, text="Risk Category:", font=("Segoe UI", 9, "bold")).pack(fill=tk.X, pady=(2, 2))
        self.combo_risk = ttk.Combobox(
            self,
            textvariable=self._var_risk,
            state="readonly",
            values=["All Risk Tiers", "High Risk", "Moderate Risk", "Low Risk"]
        )
        self.combo_risk.pack(fill=tk.X, pady=(0, 10))
        self.combo_risk.bind("<<ComboboxSelected>>", lambda event: self.on_filter_changed())

        # Minimum Attendance Threshold Slider
        ttk.Label(self, text="Min Attendance Threshold:", font=("Segoe UI", 9, "bold")).pack(fill=tk.X, pady=(2, 2))
        slider_box = ttk.Frame(self)
        slider_box.pack(fill=tk.X, pady=(0, 10))

        self.lbl_att_val = ttk.Label(slider_box, text="0%", font=("Segoe UI", 9), width=7)
        self.lbl_att_val.pack(side=tk.RIGHT)

        self.scale_att = ttk.Scale(slider_box, from_=0.0, to=100.0, variable=self._var_att_min, command=self._on_slide)
        self.scale_att.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Action Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, pady=(6, 0))

        btn_apply = ttk.Button(btn_frame, text="Apply Filters", command=self.on_filter_changed, style="Accent.TButton")
        btn_apply.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        btn_reset = ttk.Button(btn_frame, text="Reset", command=self.reset_filters)
        btn_reset.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

    def _on_slide(self, val_str: str) -> None:
        try:
            val = float(val_str)
            self.lbl_att_val.config(text=f"{val:.0f}%")
        except ValueError:
            pass

    def update_departments(self, departments: List[str]) -> None:
        values = ["All Departments"] + sorted(departments)
        self.combo_dept["values"] = values
        if self._var_dept.get() not in values:
            self._var_dept.set("All Departments")

    def get_filter_values(self) -> Dict[str, Optional[object]]:
        return {
            "department": self._var_dept.get(),
            "risk_level": self._var_risk.get(),
            "min_attendance": self._var_att_min.get() if self._var_att_min.get() > 0 else None,
            "search_query": self._var_search.get().strip()
        }

    def reset_filters(self) -> None:
        self._var_dept.set("All Departments")
        self._var_risk.set("All Risk Tiers")
        self._var_att_min.set(0.0)
        self.lbl_att_val.config(text="0%")
        self._var_search.set("")
        self.on_reset()


class DataTable(ttk.Frame):
    """Zebra-striped sortable Treeview table with conditional risk highlighting."""

    def __init__(self, parent: tk.Widget, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self._sort_column: Optional[str] = None
        self._sort_reverse: bool = False
        self._current_df: pd.DataFrame = pd.DataFrame()

        self._create_table_widgets()

    def _create_table_widgets(self) -> None:
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        cols = list(COLUMN_DISPLAY_NAMES.keys())
        self.tree = ttk.Treeview(self, columns=cols, show="headings", selectmode="browse", style="Zebra.Treeview")

        for col_id, col_label in COLUMN_DISPLAY_NAMES.items():
            self.tree.heading(col_id, text=col_label, command=lambda c=col_id: self._sort_by_column(c))
            width = 110
            if col_id in ("student_name", "department"):
                width = 150
            elif col_id in ("student_id", "semester"):
                width = 80
            elif col_id in ("current_gpa", "risk_score", "attendance_rate"):
                width = 95
            self.tree.column(col_id, width=width, minwidth=60, anchor=tk.CENTER)

        self.tree.column("student_name", anchor=tk.W)
        self.tree.column("department", anchor=tk.W)

        vsb = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        hsb = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")

        # Row color tags
        self.tree.tag_configure("evenrow", background="#FFFFFF")
        self.tree.tag_configure("oddrow", background="#F8FAFC")
        self.tree.tag_configure("high_risk", background=UIColors.RISK_HIGH_BG, foreground="#991B1B")
        self.tree.tag_configure("moderate_risk", background=UIColors.RISK_MODERATE_BG, foreground="#92400E")

    def populate(self, df: pd.DataFrame) -> None:
        self._current_df = df.copy()
        for item in self.tree.get_children():
            self.tree.delete(item)

        if df.empty:
            return

        cols = list(COLUMN_DISPLAY_NAMES.keys())
        for idx, (_, row) in enumerate(df.iterrows()):
            vals = [row.get(col, "") for col in cols]
            risk = str(row.get("risk_level", ""))

            if risk == "High Risk":
                tag = "high_risk"
            elif risk == "Moderate Risk":
                tag = "moderate_risk"
            else:
                tag = "evenrow" if idx % 2 == 0 else "oddrow"

            self.tree.insert("", tk.END, values=vals, tags=(tag,))

    def _sort_by_column(self, col: str) -> None:
        if self._current_df.empty or col not in self._current_df.columns:
            return

        if self._sort_column == col:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = col
            self._sort_reverse = False

        sorted_df = self._current_df.sort_values(by=col, ascending=not self._sort_reverse, na_position="last")
        for col_id, col_label in COLUMN_DISPLAY_NAMES.items():
            if col_id == col:
                arrow = " ▼" if self._sort_reverse else " ▲"
                self.tree.heading(col_id, text=col_label + arrow)
            else:
                self.tree.heading(col_id, text=col_label)
        self.populate(sorted_df)


# ============================================================================
# 4. EMBEDDED MATPLOTLIB VISUALIZATIONS (FigureCanvasTkAgg)
# ============================================================================

class AnalyticsChartsView(ttk.Frame):
    """Three coordinated Matplotlib subplots embedded directly into Tkinter."""

    def __init__(self, parent: tk.Widget, thresholds: Optional[AnalyticsThresholds] = None, **kwargs) -> None:
        super().__init__(parent, **kwargs)
        self.thresholds = thresholds or AnalyticsThresholds()
        self._init_figure()

    def _init_figure(self) -> None:
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(11, 7), dpi=100, facecolor=UIColors.BG_WINDOW)
        gs = self.fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)
        self.ax_scatter = self.fig.add_subplot(gs[:, 0])
        self.ax_hist = self.fig.add_subplot(gs[0, 1])
        self.ax_bar = self.fig.add_subplot(gs[1, 1])

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _style_axis(self, ax: matplotlib.axes.Axes, title: str, xlabel: str, ylabel: str) -> None:
        ax.set_title(title, fontsize=10, fontweight="bold", color=UIColors.TEXT_PRIMARY, pad=10)
        ax.set_xlabel(xlabel, fontsize=8.5, color=UIColors.TEXT_SECONDARY, labelpad=5)
        ax.set_ylabel(ylabel, fontsize=8.5, color=UIColors.TEXT_SECONDARY, labelpad=5)
        ax.tick_params(colors=UIColors.TEXT_SECONDARY, labelsize=8)
        ax.set_facecolor(UIColors.BG_CARD)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(UIColors.BORDER_LIGHT)

        ax.grid(True, linestyle="--", alpha=0.5, color=UIColors.CHART_GRID)

    def render(self, df: pd.DataFrame) -> None:
        """Dynamically update and redraw all 3 subplots with current active data."""
        self.ax_scatter.clear()
        self.ax_hist.clear()
        self.ax_bar.clear()

        if df.empty:
            self.ax_scatter.text(
                0.5, 0.5, "No student records match active filters",
                ha="center", va="center", transform=self.ax_scatter.transAxes,
                color=UIColors.TEXT_MUTED, fontsize=11, fontweight="bold"
            )
            self.ax_scatter.axis("off")
            self.ax_hist.axis("off")
            self.ax_bar.axis("off")
            self.canvas.draw_idle()
            return

        # -------------------------------------------------------------
        # Subplot 1: Attendance vs. GPA Scatter Matrix & Risk Frontier
        # -------------------------------------------------------------
        self._style_axis(
            self.ax_scatter,
            title="Attendance vs. GPA Risk Matrix",
            xlabel="Attendance Rate (%)",
            ylabel="Current GPA (0.0 - 4.0)"
        )

        risk_cats = [
            ("Low Risk", UIColors.RISK_LOW, "o", 45, 0.7),
            ("Moderate Risk", UIColors.RISK_MODERATE, "s", 45, 0.8),
            ("High Risk", UIColors.RISK_HIGH, "^", 60, 0.9),
        ]

        for risk_tier, color, marker, size, alpha in risk_cats:
            sub = df[df["risk_level"] == risk_tier]
            if not sub.empty:
                self.ax_scatter.scatter(
                    sub["attendance_rate"], sub["current_gpa"],
                    c=color, label=f"{risk_tier} ({len(sub)})",
                    marker=marker, s=size, alpha=alpha, edgecolors="none"
                )

        # Benchmark reference lines
        self.ax_scatter.axvline(
            self.thresholds.ATTENDANCE_MIN_BENCHMARK,
            color=UIColors.RISK_HIGH, linestyle="--", linewidth=1.2, alpha=0.7,
            label=f"Min Attendance ({self.thresholds.ATTENDANCE_MIN_BENCHMARK:.0f}%)"
        )
        self.ax_scatter.axhline(
            self.thresholds.GPA_PROBATION_LIMIT,
            color=UIColors.RISK_HIGH, linestyle=":", linewidth=1.2, alpha=0.7,
            label=f"Probation GPA ({self.thresholds.GPA_PROBATION_LIMIT:.1f})"
        )

        # OLS regression trend line
        if len(df) >= 3:
            valid = df["attendance_rate"].notna() & df["current_gpa"].notna()
            x_vals = df.loc[valid, "attendance_rate"].values
            y_vals = df.loc[valid, "current_gpa"].values
            if len(np.unique(x_vals)) > 1:
                slope, intercept = np.polyfit(x_vals, y_vals, deg=1)
                x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
                y_line = np.clip(slope * x_line + intercept, 0.0, 4.0)
                self.ax_scatter.plot(
                    x_line, y_line, color="#475569", linestyle="-",
                    linewidth=1.5, alpha=0.85, label="Cohort Trendline"
                )

        self.ax_scatter.set_ylim(0.8, 4.2)
        self.ax_scatter.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

        # -------------------------------------------------------------
        # Subplot 2: Cohort GPA Distribution & Density
        # -------------------------------------------------------------
        self._style_axis(
            self.ax_hist,
            title="Cohort GPA Distribution",
            xlabel="Current GPA",
            ylabel="Student Count"
        )

        gpa_series = df["current_gpa"].dropna()
        if not gpa_series.empty:
            bins = np.linspace(1.0, 4.0, 16)
            self.ax_hist.hist(gpa_series, bins=bins, color="#3B82F6", edgecolor="#FFFFFF", alpha=0.75, rwidth=0.9)
            mean_gpa = gpa_series.mean()
            median_gpa = gpa_series.median()
            self.ax_hist.axvline(mean_gpa, color="#DC2626", linestyle="-", linewidth=1.5, label=f"Mean: {mean_gpa:.2f}")
            self.ax_hist.axvline(median_gpa, color="#059669", linestyle="--", linewidth=1.5, label=f"Median: {median_gpa:.2f}")
            self.ax_hist.axvspan(1.0, self.thresholds.GPA_PROBATION_LIMIT, color=UIColors.RISK_HIGH_BG, alpha=0.45, label="Probation Zone")
            self.ax_hist.set_xlim(0.9, 4.1)
            self.ax_hist.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

        # -------------------------------------------------------------
        # Subplot 3: Departmental Risk Profile (Stacked Bar Chart)
        # -------------------------------------------------------------
        self._style_axis(
            self.ax_bar,
            title="Risk Tier Distribution by Department",
            xlabel="",
            ylabel="Number of Students"
        )

        dept_summary = pd.crosstab(df["department"], df["risk_level"])
        for col in ["Low Risk", "Moderate Risk", "High Risk"]:
            if col not in dept_summary.columns:
                dept_summary[col] = 0

        dept_summary["Total"] = dept_summary["Low Risk"] + dept_summary["Moderate Risk"] + dept_summary["High Risk"]
        dept_summary = dept_summary.sort_values(by="Total", ascending=True)

        depts = dept_summary.index.tolist()
        clean_dept_names = [d if len(d) <= 15 else d[:13] + ".." for d in depts]
        low_vals = dept_summary["Low Risk"].values
        mod_vals = dept_summary["Moderate Risk"].values
        high_vals = dept_summary["High Risk"].values

        y_pos = np.arange(len(depts))
        self.ax_bar.barh(y_pos, low_vals, height=0.55, color=UIColors.RISK_LOW, label="Low", alpha=0.85)
        self.ax_bar.barh(y_pos, mod_vals, height=0.55, left=low_vals, color=UIColors.RISK_MODERATE, label="Moderate", alpha=0.85)
        self.ax_bar.barh(y_pos, high_vals, height=0.55, left=low_vals + mod_vals, color=UIColors.RISK_HIGH, label="High", alpha=0.85)

        self.ax_bar.set_yticks(y_pos)
        self.ax_bar.set_yticklabels(clean_dept_names, fontsize=8)
        self.ax_bar.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        self.ax_bar.legend(loc="lower right", fontsize=7.5, framealpha=0.9)

        self.canvas.draw_idle()


# ============================================================================
# 5. ROOT WINDOW APPLICATION ORCHESTRATOR
# ============================================================================

class EduPulseApp(tk.Tk):
    """
    Root desktop application window orchestrating views, state management,
    user interactions, and analytical rendering.
    """

    def __init__(self, initial_csv_path: Optional[str] = None) -> None:
        super().__init__()
        self.title("EduPulse Analytics | Student Academic Performance & Risk Intelligence")
        self.geometry("1300x840")
        self.minsize(1040, 680)
        self.configure(bg=UIColors.BG_WINDOW)

        self.thresholds = AnalyticsThresholds()
        self.data_engine = AcademicDataEngine(thresholds=self.thresholds)
        self.active_file_path: Optional[str] = None

        self._configure_ttk_theme()
        self._build_layout()
        self._build_menu()

        if initial_csv_path and os.path.exists(initial_csv_path):
            self.load_dataset(initial_csv_path)

    def _configure_ttk_theme(self) -> None:
        """Apply a sleek, modern enterprise design theme across all ttk widgets."""
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.style.configure(".", font=("Segoe UI", 9), background=UIColors.BG_WINDOW)
        self.style.configure("TFrame", background=UIColors.BG_WINDOW)
        self.style.configure("TLabelframe", background=UIColors.BG_CARD, foreground=UIColors.TEXT_PRIMARY)
        self.style.configure("TLabelframe.Label", font=("Segoe UI", 9, "bold"), background=UIColors.BG_CARD, foreground=UIColors.TEXT_PRIMARY)

        # Standard button
        self.style.configure(
            "TButton",
            font=("Segoe UI", 9),
            padding=(10, 5),
            background=UIColors.BG_PANEL,
            foreground=UIColors.TEXT_PRIMARY,
            borderwidth=1,
            relief="flat"
        )
        self.style.map("TButton", background=[("active", UIColors.BORDER_LIGHT)])

        # Primary Accent button
        self.style.configure(
            "Accent.TButton",
            font=("Segoe UI", 9, "bold"),
            padding=(12, 6),
            background=UIColors.ACCENT_PRIMARY,
            foreground="#FFFFFF",
            borderwidth=0
        )
        self.style.map("Accent.TButton", background=[("active", UIColors.ACCENT_HOVER)], foreground=[("active", "#FFFFFF")])

        # Treeview styling
        self.style.configure(
            "Zebra.Treeview",
            font=("Segoe UI", 9),
            rowheight=26,
            background="#FFFFFF",
            fieldbackground="#FFFFFF",
            foreground=UIColors.TEXT_PRIMARY,
            borderwidth=1
        )
        self.style.configure(
            "Zebra.Treeview.Heading",
            font=("Segoe UI", 9, "bold"),
            background=UIColors.BG_PANEL,
            foreground=UIColors.TEXT_PRIMARY,
            relief="flat",
            padding=(6, 4)
        )
        self.style.map("Zebra.Treeview", background=[("selected", "#DBEAFE")], foreground=[("selected", "#1E3A8A")])

        # Notebook tabs
        self.style.configure("TNotebook", background=UIColors.BG_WINDOW, borderwidth=0)
        self.style.configure(
            "TNotebook.Tab",
            font=("Segoe UI", 9, "bold"),
            padding=(16, 7),
            background=UIColors.BG_PANEL,
            foreground=UIColors.TEXT_SECONDARY
        )
        self.style.map("TNotebook.Tab", background=[("selected", UIColors.BG_CARD)], foreground=[("selected", UIColors.ACCENT_PRIMARY)])

    def _build_menu(self) -> None:
        """Construct native application menu bar."""
        menubar = tk.Menu(self)

        # File Menu
        menu_file = tk.Menu(menubar, tearoff=0)
        menu_file.add_command(label="Open CSV Dataset...", accelerator="Ctrl+O", command=self.on_open_file)
        menu_file.add_command(label="Load Sample Dataset", accelerator="Ctrl+L", command=self.on_load_sample)
        menu_file.add_command(label="Export Filtered Data...", accelerator="Ctrl+S", command=self.on_export_csv)
        menu_file.add_separator()
        menu_file.add_command(label="Exit", accelerator="Ctrl+Q", command=self.quit)
        menubar.add_cascade(label="File", menu=menu_file)

        # View Menu
        menu_view = tk.Menu(menubar, tearoff=0)
        menu_view.add_command(label="Visual Analytics View", command=lambda: self.notebook.select(0))
        menu_view.add_command(label="Student Records Table", command=lambda: self.notebook.select(1))
        menu_view.add_separator()
        menu_view.add_command(label="Reset All Filters", accelerator="Ctrl+R", command=self.on_reset_filters)
        menubar.add_cascade(label="View", menu=menu_view)

        # Help Menu
        menu_help = tk.Menu(menubar, tearoff=0)
        menu_help.add_command(label="Data Schema Reference", command=self._show_schema_dialog)
        menu_help.add_command(label="About EduPulse Analytics", command=self._show_about_dialog)
        menubar.add_cascade(label="Help", menu=menu_help)

        self.config(menu=menubar)

        # Keybindings
        self.bind("<Control-o>", lambda e: self.on_open_file())
        self.bind("<Control-l>", lambda e: self.on_load_sample())
        self.bind("<Control-s>", lambda e: self.on_export_csv())
        self.bind("<Control-q>", lambda e: self.quit())
        self.bind("<Control-r>", lambda e: self.on_reset_filters())

    def _build_layout(self) -> None:
        """Construct the responsive UI layout."""
        # Top Header Bar
        header = tk.Frame(self, bg=UIColors.BG_CARD, padx=20, pady=12, highlightthickness=1, highlightbackground=UIColors.BORDER_LIGHT)
        header.pack(fill=tk.X, side=tk.TOP)

        title_box = tk.Frame(header, bg=UIColors.BG_CARD)
        title_box.pack(side=tk.LEFT)

        tk.Label(
            title_box,
            text="🎓 EduPulse Analytics",
            font=("Segoe UI", 14, "bold"),
            fg=UIColors.TEXT_PRIMARY,
            bg=UIColors.BG_CARD
        ).pack(anchor="w")

        tk.Label(
            title_box,
            text="Student Academic Performance & Early Warning Risk Intelligence Dashboard",
            font=("Segoe UI", 9),
            fg=UIColors.TEXT_SECONDARY,
            bg=UIColors.BG_CARD
        ).pack(anchor="w")

        # Quick Action Buttons
        actions_box = tk.Frame(header, bg=UIColors.BG_CARD)
        actions_box.pack(side=tk.RIGHT)

        ttk.Button(actions_box, text="📂 Open CSV", command=self.on_open_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions_box, text="⚡ Load Sample", command=self.on_load_sample).pack(side=tk.LEFT, padx=4)
        ttk.Button(actions_box, text="💾 Export Filtered", command=self.on_export_csv).pack(side=tk.LEFT, padx=4)

        # KPI Summary Cards Bar
        cards_container = tk.Frame(self, bg=UIColors.BG_WINDOW, padx=20, pady=10)
        cards_container.pack(fill=tk.X, side=tk.TOP)

        for i in range(4):
            cards_container.columnconfigure(i, weight=1, uniform="kpi")

        self.card_total = StatCard(cards_container, title="Total Students", initial_value="0", subtitle="Active cohort enrollment", accent_color=UIColors.ACCENT_PRIMARY)
        self.card_total.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        self.card_gpa = StatCard(cards_container, title="Cohort Avg GPA", initial_value="0.00", subtitle="Scale 0.00 to 4.00", accent_color="#06B6D4")
        self.card_gpa.grid(row=0, column=1, sticky="ew", padx=6)

        self.card_risk = StatCard(cards_container, title="At-Risk Students", initial_value="0", subtitle="High & Moderate Risk", accent_color=UIColors.RISK_HIGH)
        self.card_risk.grid(row=0, column=2, sticky="ew", padx=6)

        self.card_att = StatCard(cards_container, title="Average Attendance", initial_value="0.0%", subtitle="Target benchmark: 75%", accent_color=UIColors.RISK_LOW)
        self.card_att.grid(row=0, column=3, sticky="ew", padx=(6, 0))

        # Main Body
        body = tk.Frame(self, bg=UIColors.BG_WINDOW, padx=20, pady=6)
        body.pack(fill=tk.BOTH, expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)

        # Left Filter Panel
        self.filter_panel = FilterPanel(body, on_filter_changed=self.on_filter_changed, on_reset=self.on_reset_filters, width=260)
        self.filter_panel.grid(row=0, column=0, sticky="ns", padx=(0, 12), pady=(0, 4))

        # Right Tabbed Notebook Workspace
        self.notebook = ttk.Notebook(body)
        self.notebook.grid(row=0, column=1, sticky="nsew")

        # Tab 1: Embedded Visual Analytics
        self.charts_view = AnalyticsChartsView(self.notebook, thresholds=self.thresholds)
        self.notebook.add(self.charts_view, text="  📊 Visual Analytics Dashboard  ")

        # Tab 2: Records Table
        table_container = ttk.Frame(self.notebook)
        table_container.rowconfigure(1, weight=1)
        table_container.columnconfigure(0, weight=1)

        tbl_info_bar = tk.Frame(table_container, bg=UIColors.BG_CARD, padx=10, pady=6, highlightthickness=1, highlightbackground=UIColors.BORDER_LIGHT)
        tbl_info_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        self.lbl_table_count = tk.Label(tbl_info_bar, text="Displaying: 0 students", font=("Segoe UI", 9, "bold"), fg=UIColors.TEXT_PRIMARY, bg=UIColors.BG_CARD)
        self.lbl_table_count.pack(side=tk.LEFT)

        tk.Label(
            tbl_info_bar,
            text="💡 Tip: Click column headers to sort | Red rows: High Risk | Amber: Moderate",
            font=("Segoe UI", 8),
            fg=UIColors.TEXT_MUTED,
            bg=UIColors.BG_CARD
        ).pack(side=tk.RIGHT)

        self.data_table = DataTable(table_container)
        self.data_table.grid(row=1, column=0, sticky="nsew")
        self.notebook.add(table_container, text="  📋 Student Records & Risk Roster  ")

        # Status Bar
        self.status_bar = tk.Frame(self, bg=UIColors.BG_PANEL, padx=16, pady=4, highlightthickness=1, highlightbackground=UIColors.BORDER_LIGHT)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        self.lbl_status_file = tk.Label(
            self.status_bar,
            text="No dataset loaded. Click 'Open CSV' or 'Load Sample' to begin.",
            font=("Segoe UI", 9),
            fg=UIColors.TEXT_SECONDARY,
            bg=UIColors.BG_PANEL
        )
        self.lbl_status_file.pack(side=tk.LEFT)

        self.lbl_status_meta = tk.Label(
            self.status_bar,
            text="Ready",
            font=("Segoe UI", 9),
            fg=UIColors.TEXT_MUTED,
            bg=UIColors.BG_PANEL
        )
        self.lbl_status_meta.pack(side=tk.RIGHT)

    def load_dataset(self, file_path: str) -> None:
        """Load and display dataset with comprehensive exception handling."""
        try:
            total_raw, clean_processed = self.data_engine.load_csv(file_path)
            self.active_file_path = file_path

            departments = self.data_engine.get_departments_list()
            self.filter_panel.update_departments(departments)
            self._refresh_all_views()

            filename = os.path.basename(file_path)
            self.lbl_status_file.config(text=f"Active File: {filename} | {clean_processed:,} records ingested successfully")
            self.lbl_status_meta.config(text=f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

        except SchemaValidationError as sve:
            messagebox.showerror("Schema Validation Error", f"CSV file violates required schema:\n\n{sve}", parent=self)
        except EmptyDatasetError as ede:
            messagebox.showwarning("Empty Dataset", f"Warning: {ede}", parent=self)
        except DataEngineError as dee:
            messagebox.showerror("Data Processing Error", f"Failed to parse dataset:\n\n{dee}", parent=self)
        except Exception as exc:
            messagebox.showerror("Unexpected Error", f"A critical error occurred:\n\n{exc}", parent=self)

    def _refresh_all_views(self) -> None:
        """Synchronize KPI cards, visual charts, and Treeview table with current data."""
        current_df = self.data_engine.current_data

        kpis = self.data_engine.get_summary_kpis()
        self.card_total.update_value(f"{kpis['total_students']:,}", "Active filtered records")
        self.card_gpa.update_value(f"{kpis['avg_gpa']:.2f}", "Cohort mean GPA")
        risk_pct = (kpis["total_at_risk"] / kpis["total_students"] * 100) if kpis["total_students"] > 0 else 0.0
        self.card_risk.update_value(
            f"{kpis['total_at_risk']} ({risk_pct:.1f}%)",
            f"High: {kpis['high_risk_count']} | Mod: {kpis['moderate_risk_count']}"
        )
        self.card_att.update_value(f"{kpis['avg_attendance']:.1f}%", "Benchmark: 75.0%")

        self.charts_view.render(current_df)
        self.data_table.populate(current_df)
        total_in_memory = len(self.data_engine.full_data)
        self.lbl_table_count.config(text=f"Displaying: {len(current_df):,} of {total_in_memory:,} students")

    def on_filter_changed(self) -> None:
        """Triggered when any filter dropdown, slider, or search input updates."""
        if not self.data_engine.has_data:
            return
        filter_params = self.filter_panel.get_filter_values()
        self.data_engine.apply_filters(**filter_params)
        self._refresh_all_views()
        count = len(self.data_engine.current_data)
        self.lbl_status_meta.config(text=f"Filtered: {count:,} records | {datetime.now().strftime('%H:%M:%S')}")

    def on_reset_filters(self) -> None:
        """Reset filters back to all records."""
        if not self.data_engine.has_data:
            return
        self.data_engine.apply_filters()
        self._refresh_all_views()
        self.lbl_status_meta.config(text=f"Filters reset | {datetime.now().strftime('%H:%M:%S')}")

    def on_open_file(self) -> None:
        """File open dialog handler."""
        initial_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = filedialog.askopenfilename(
            parent=self,
            title="Select Student Academic Records CSV",
            initialdir=initial_dir if os.path.exists(initial_dir) else os.getcwd(),
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if file_path:
            self.load_dataset(file_path)

    def on_load_sample(self) -> None:
        """Load bundled sample dataset or generate it on demand."""
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_students.csv")
        if not os.path.exists(sample_path):
            sample_path = create_sample_mock_csv(sample_path, num_records=350)
        self.load_dataset(sample_path)
        messagebox.showinfo("Sample Loaded", "Sample student dataset loaded successfully.", parent=self)

    def on_export_csv(self) -> None:
        """Export current filtered view to CSV file."""
        if not self.data_engine.has_data:
            messagebox.showwarning("Export Warning", "No data is currently loaded to export.", parent=self)
            return
        export_df = self.data_engine.current_data
        if export_df.empty:
            messagebox.showwarning("Export Warning", "The current filter matches 0 records. Nothing to export.", parent=self)
            return

        save_path = filedialog.asksaveasfilename(
            parent=self,
            title="Export Filtered Student Records",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not save_path:
            return
        try:
            export_df.to_csv(save_path, index=False)
            messagebox.showinfo(
                "Export Complete",
                f"Successfully exported {len(export_df):,} records to:\n{save_path}",
                parent=self
            )
        except Exception as exc:
            messagebox.showerror("Export Failed", f"Could not export data: {exc}", parent=self)

    def _show_schema_dialog(self) -> None:
        """Display reference dialog for required CSV headers."""
        schema_text = (
            "Required CSV Columns:\n\n"
            "• student_id: String (e.g. STU-1001)\n"
            "• student_name: String (Full Name)\n"
            "• department: String (Department / Program)\n"
            "• semester: String (e.g. Sem 1 - Sem 8)\n"
            "• attendance_rate: Float [0.0 - 100.0]\n"
            "• study_hours_weekly: Float [0.0 - 80.0]\n"
            "• assignment_completion_rate: Float [0.0 - 100.0]\n"
            "• midterm_score: Float [0.0 - 100.0]\n"
            "• quiz_average: Float [0.0 - 100.0]\n"
            "• previous_gpa: Float [0.00 - 4.00]\n"
            "• current_gpa: Float [0.00 - 4.00]\n"
        )
        messagebox.showinfo("CSV Schema Reference", schema_text, parent=self)

    def _show_about_dialog(self) -> None:
        """Display dialog with application info."""
        about_text = (
            "EduPulse Analytics v1.0.0\n"
            "Student Academic Performance & Early Warning Risk Intelligence\n\n"
            "MCA Semester 1 Mini-Project\n"
            "Core Stack: Python, Tkinter (ttk), Pandas, NumPy, and Matplotlib."
        )
        messagebox.showinfo("About EduPulse Analytics", about_text, parent=self)


# ============================================================================
# 6. STANDALONE SYNTHETIC DATA GENERATOR
# ============================================================================

def create_sample_mock_csv(target_path: str, num_records: int = 350) -> str:
    """Generate synthetic CSV dataset with realistic correlations."""
    os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
    np.random.seed(42)

    first_names = ["Aarav", "Elena", "Marcus", "Priya", "Liam", "Sophia", "Zack", "Chloe", "Chen", "Fatima", "Carlos", "Maya", "Lucas", "Noah", "Yuki", "Devon"]
    last_names = ["Patel", "Vance", "Chen", "Rodriguez", "Sharma", "Kowalski", "Nguyen", "Kim", "Al-Mansoor", "Dubois", "Santos", "Gupta", "Johansson"]
    departments = ["Computer Science", "Data Science & AI", "Mechanical Engineering", "Electrical Engineering", "Business Analytics", "Biotechnology"]
    dept_weights = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
    semesters = [f"Sem {i}" for i in range(1, 9)]

    student_ids = [f"STU-{1000 + i:04d}" for i in range(1, num_records + 1)]
    student_names = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(num_records)]
    chosen_depts = np.random.choice(departments, size=num_records, p=dept_weights)
    chosen_semesters = np.random.choice(semesters, size=num_records)

    # Latent academic engagement factor
    engagement = np.random.normal(0.0, 1.0, size=num_records)

    attendance = np.clip(np.round(80.0 + (engagement * 10.0) + np.random.normal(0, 7.0, num_records), 1), 42.0, 99.5)
    study_hrs = np.clip(np.round(16.0 + (engagement * 5.0) + np.random.normal(0, 4.0, num_records), 1), 3.0, 45.0)
    assign_rate = np.clip(np.round(78.0 + (engagement * 11.0) + np.random.normal(0, 6.0, num_records), 1), 40.0, 100.0)
    midterm = np.clip(np.round(72.0 + (engagement * 14.0) + (study_hrs * 0.4) + np.random.normal(0, 8.0, num_records), 1), 25.0, 99.0)
    quiz = np.clip(np.round((midterm * 0.7) + (assign_rate * 0.3) + np.random.normal(0, 5.0, num_records), 1), 30.0, 99.0)

    composite = (midterm * 0.4) + (quiz * 0.3) + (assign_rate * 0.3)
    current_gpa = np.clip(np.round(((composite - 40.0) / 60.0) * 2.8 + 1.2 + np.random.normal(0, 0.15, num_records), 2), 1.10, 4.00)
    prev_gpa = np.clip(np.round(current_gpa + np.random.normal(0, 0.25, num_records), 2), 1.00, 4.00)

    df = pd.DataFrame({
        "student_id": student_ids,
        "student_name": student_names,
        "department": chosen_depts,
        "semester": chosen_semesters,
        "attendance_rate": attendance,
        "study_hours_weekly": study_hrs,
        "assignment_completion_rate": assign_rate,
        "midterm_score": midterm,
        "quiz_average": quiz,
        "previous_gpa": prev_gpa,
        "current_gpa": current_gpa
    })
    df.to_csv(target_path, index=False)
    return target_path


# ============================================================================
# 7. EXECUTION ENTRY POINT
# ============================================================================

def main() -> None:
    """Launch EduPulse Analytics Desktop Application."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    sample_file = os.path.join(base_dir, "sample_students.csv")

    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        initial_csv = sys.argv[1]
    else:
        if not os.path.exists(sample_file):
            sample_file = create_sample_mock_csv(sample_file, num_records=350)
        initial_csv = sample_file

    app = EduPulseApp(initial_csv_path=initial_csv)
    app.mainloop()


if __name__ == "__main__":
    main()
