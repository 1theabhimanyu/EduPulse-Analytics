# EduPulse Analytics 🎓
### Student Academic Performance & Early Warning Risk Intelligence Dashboard
**MCA Semester 1 — Python Mini-Project**

---

## 📌 1. Project Overview
**EduPulse Analytics** is a production-grade, enterprise-styled desktop analytics application engineered in Python. It provides higher-education institutions, academic advisors, and faculty with a real-time data engine to monitor student cohorts, calculate composite academic risk indices, and visualize early-warning risk frontiers *before* end-term semester examinations.

### Key Highlights
- **100% Native Desktop App**: Built using Python 3, Tkinter with custom TTK styling, Pandas, NumPy, and embedded Matplotlib.
- **Strict OOP Architecture**: Complete separation of concerns between data processing (`AcademicDataEngine`) and presentation components (`StatCard`, `FilterPanel`, `DataTable`, `AnalyticsChartsView`, `EduPulseApp`).
- **High-Performance Vectorization**: No slow row-by-row iteration loops (`for iterrows()`). Data cleaning, boundary clipping, composite risk indexing, and demographic filtering use NumPy and Pandas vectorized masks.
- **Embedded Dynamic Visualizations**: Matplotlib charts embedded directly via `FigureCanvasTkAgg` without external popups, dynamically synchronizing whenever filters or new datasets are applied.
- **Production-Grade Error Handling**: Comprehensive validation for missing CSV files, unreadable encodings, empty datasets, missing mandatory columns, and corrupted values.

---

## 🏗️ 2. Architecture & Design Principles

```
┌────────────────────────────────────────────────────────────────────────┐
│                        EduPulseApp (Tk Root)                           │
│  - Window lifecycle, native menu bar, global shortcuts & event routing│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
       ┌────────────────────────────┼────────────────────────────┐
       ▼                            ▼                            ▼
┌──────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│  Executive KPI Cards │  │ FilterPanel Sidebar  │  │   Notebook (Tabs)    │
│  - Total Enrollment  │  │ - Search (ID / Name) │  │  Tab 1: Visual Charts│
│  - Cohort Mean GPA   │  │ - Department Dropdown│  │  Tab 2: Records Table│
│  - At-Risk Count (%) │  │ - Risk Tier Dropdown │  └──────────┬───────────┘
│  - Average Attendance│  │ - Attendance Slider  │             │
└──────────────────────┘  └──────────┬───────────┘             │
                                     │                         │
                                     ▼                         ▼
                  ┌──────────────────────────────────────────────────────┐
                  │             AcademicDataEngine (Pandas/NumPy)        │
                  │ - CSV Ingestion & Schema Verification                │
                  │ - Vectorized Imputation & Boundary Clipping          │
                  │ - Composite Risk Index Calculation (0 - 100 Scale)   │
                  │ - Multi-attribute Boolean Mask Filtering             │
                  │ - Cross-tabulation & Departmental Aggregations       │
                  └──────────────────────────────────────────────────────┘
```

### OOP Modular Components:
1. **`AcademicDataEngine`**:
   - Manages state for raw, processed, and filtered DataFrames.
   - Enforces the 11-column mandatory schema.
   - Calculates the composite early-warning score:
     $$\text{Risk Score} = \text{Penalty}_{\text{Attendance}} + \text{Penalty}_{\text{Exams}} + \text{Penalty}_{\text{GPA}} + \text{Penalty}_{\text{Assignments}}$$
   - Categorizes students into **High Risk**, **Moderate Risk**, and **Low Risk (Thriving)**.
   - Determines academic standing: **Dean's List**, **Good Standing**, **Probation**.
2. **`StatCard`**:
   - Reusable executive metric card with custom left accent stripe, bold metric typography, and subtitle status.
3. **`FilterPanel`**:
   - Encapsulates multi-dimensional controls (Department `ttk.Combobox`, Risk Tier `ttk.Combobox`, Attendance `ttk.Scale`, and Search `ttk.Entry`).
4. **`DataTable`**:
   - Encapsulates `ttk.Treeview` with dual scrollbars, zebra striping, conditional color badges (red for High Risk, amber for Moderate Risk), and bidirectional click-to-sort column headers.
5. **`AnalyticsChartsView`**:
   - Manages a 3-subplot Matplotlib `Figure` embedded inside a `FigureCanvasTkAgg` canvas:
     1. **Attendance vs. GPA Risk Matrix**: Scatter plot with color-coded risk markers, institutional thresholds (75% min attendance line, 2.0 probation line), and an empirical OLS regression trendline.
     2. **Cohort GPA Distribution & Density**: Histogram with vertical dashed markers for Mean and Median GPAs and shaded Academic Probation zones.
     3. **Departmental Risk Breakdown**: Stacked horizontal bar chart illustrating the distribution of risk tiers per department.

---

## 📊 3. CSV Dataset Schema

The application accepts any standard CSV matching this schema (column headers are case-insensitive and whitespace-tolerant):

| Column Name | Data Type | Description | Valid Domain / Example |
| :--- | :--- | :--- | :--- |
| `student_id` | String | Unique student identification code | `STU-1001` |
| `student_name` | String | Full name of the student | `Elena Vance` |
| `department` | String | Academic department / program | `Computer Science` |
| `semester` | String | Current academic semester | `Sem 1` to `Sem 8` |
| `attendance_rate` | Float | Percentage attendance recorded | `0.0` - `100.0` (Benchmark: `75.0%`) |
| `study_hours_weekly` | Float | Self-study hours per week | `0.0` - `80.0` |
| `assignment_completion_rate`| Float | Assignment submission rate (%) | `0.0` - `100.0` |
| `midterm_score` | Float | Midterm examination score | `0.0` - `100.0` |
| `quiz_average` | Float | Continuous assessment / quiz score | `0.0` - `100.0` |
| `previous_gpa` | Float | Cumulative previous semester GPA | `0.00` - `4.00` |
| `current_gpa` | Float | Current semester GPA | `0.00` - `4.00` (Probation: `< 2.00`) |

---

## 🚀 4. How to Run the Application

Your project is ready with dependencies installed in your virtual environment:

### Step 1: Open Terminal and Navigate to Project Directory
```bash
cd "/home/abhimanyu/Python Project For MCA SEM 1"
```

### Step 2: Run Using Your Virtual Environment
```bash
/home/abhimanyu/myenv/bin/python main.py
```

*Or, if you activate your virtual environment first:*
```bash
source /home/abhimanyu/myenv/bin/activate
python main.py
```

---

## 💡 5. Keyboard Shortcuts
- `Ctrl + O`: Open CSV file dialog to load an external dataset
- `Ctrl + L`: Instant-load the bundled sample dataset (`sample_students.csv`)
- `Ctrl + S`: Export the current filtered data view to a new CSV file
- `Ctrl + R`: Reset all filters back to the full cohort
- `Ctrl + Q`: Quit the application

---

## 🎓 6. MCA Sem 1 Viva / Defense Q&A

**Q1: Why did you use `FigureCanvasTkAgg` instead of `plt.show()`?**
> *Answer:* `plt.show()` opens a blocking, standalone pop-up window managed by Matplotlib's default GUI event loop, which breaks the application flow. `FigureCanvasTkAgg` embeds the Matplotlib drawing canvas directly into Tkinter's widget hierarchy (`pack`/`grid`), allowing the GUI to maintain a single responsive window with dynamic, in-place re-rendering.

**Q2: How does NumPy vectorization improve this application over traditional loops?**
> *Answer:* In Python, standard `for` loops and `iterrows()` execute at interpreted Python speeds and incur significant overhead. By using vectorized operations (`np.where`, `np.select`, `np.clip`, and boolean indexing), computations are pushed down to compiled C-level loops, calculating risk scores and filtering hundreds of records instantaneously.

**Q3: How is missing or dirty data handled?**
> *Answer:* The `AcademicDataEngine` validates presence of all mandatory columns. Numeric columns are coerced via `pd.to_numeric(errors="coerce")`, converting invalid text to `NaN`. Missing numeric values are imputed with the column median, and boundaries are enforced via `np.clip` (e.g., GPA clamped to 0.0 - 4.0, attendance to 0 - 100%).
# EduPulse-Analytics
