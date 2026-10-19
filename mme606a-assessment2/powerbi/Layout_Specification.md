# Power BI Dashboard Layout Specification

**Report name:** MME606A_Manufacturing_Performance  
**Theme:** Dark blue executive (`#1a1a2e` background, accent `#e94560`, KPI `#00d9ff`)  
**Data source:** `manufacturing_operations_featured.csv` (post Python pipeline)

## Page 1 — Executive Overview

| Zone | Visual | Fields | Purpose |
|------|--------|--------|---------|
| Top | 4× KPI cards | Avg Efficiency, Failure Rate %, Total Downtime Hours, Avg Energy Intensity | At-a-glance health |
| Left | Slicer panel | Location, Machine_Type, Record_Month | Interactive filtering |
| Centre | Clustered column | Record_Month × Avg Efficiency | Trend monitoring |
| Right | Donut | Predicted_Failure | Risk mix |
| Bottom left | Bar chart | Machine_Type × Total Downtime Hours | Focus maintenance by asset class |
| Bottom right | Map (shape) or table | Location × Failure Rate % | Plant comparison |

## Page 2 — Reliability & Maintenance

| Visual | Fields |
|--------|--------|
| Scatter | Vibration_Level (mm/s) vs Faults_Reported, size = Downtime |
| Line | Days_Since_Maintenance (avg) by month |
| Matrix | Machine_ID rows; columns: Downtime, Faults, Failure flag |
| Gauge | Avg Availability vs target 92% |

## Page 3 — Energy & Efficiency

| Visual | Fields |
|--------|--------|
| Scatter | Operating_Hours vs Power_Consumption (kWh) |
| Histogram | Energy_Intensity_kWh_per_hr |
| Waterfall | Maintenance_Cost (USD) by Machine_Type |

## Interactions

- Cross-filter all visuals from slicers.
- Drill-through from matrix row to machine detail page (optional Page 4).

## Accessibility

- Minimum font 10pt; WCAG contrast on KPI cards.
- Tooltips: include engineering definitions (e.g. Availability = operating / (operating + downtime)).

## Export for report appendix

Capture full-page screenshots of Pages 1–3 at 1920×1080 after applying representative slicer states (all plants; worst plant only; CNC-only filter).
