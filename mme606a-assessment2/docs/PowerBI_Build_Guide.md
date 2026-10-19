# Power BI Build Guide — MME606A Assessment 2 Paper A

This guide builds the mandatory `.pbix` dashboard from the Python pipeline output.

## 1. Prepare data

1. Run the full pipeline (see `README.md`).
2. In Power BI Desktop: **Get data → Text/CSV** → select `data/manufacturing_operations_featured.csv`.
3. In Power Query:
   - Set `Record_Month_DT` and `Last_Maintenance_Date` to **Date**.
   - Ensure `Predicted_Failure` is text; `Predicted_Failure_Binary` is whole number.
   - **Close & Apply** (no row removal — cleaning already done in Python).

## 2. Create measures

Open **Modeling → New measure** and paste each measure from `powerbi/DAX_Measures.txt`. Rename the table to `ManufacturingOps` if your import name differs, and update table references in DAX.

## 3. Build pages

Follow `powerbi/Layout_Specification.md` for three pages:

- **Executive Overview** — KPIs, slicers, efficiency trend, failure donut, downtime by machine type.
- **Reliability & Maintenance** — vibration–fault scatter, maintenance interval line, machine matrix.
- **Energy & Efficiency** — power vs hours scatter, energy intensity histogram.

Apply the dark theme or custom colours to match the Python `dashboard_preview.png` for consistent appendix figures.

## 4. Slicers and storytelling

Add synchronized slicers for `Location`, `Machine_Type`, and `Record_Month` (between slider or dropdown). Set default to **all data**; save a bookmark **Plant B focus** for the video walkthrough.

## 5. Validate against Python metrics

Check that **Failure Rate %** is approximately **22%** overall and **Avg Efficiency** ≈ **78.9%** with no filters — these should match `outputs/eda_summary.json`.

## 6. Publish and record

1. Save as `MME606A_Manufacturing_Dashboard.pbix`.
2. Record Loom (webcam + screen): overview → filter to high-risk machines → recommendations.
3. Paste link in report: `[INSERT LOOM VIDEO LINK]`.

## 7. Submission packaging

Upload `.pbix`, Python scripts, and CSV via **OneDrive** link on the Moodle cover sheet, per assessment instructions.
