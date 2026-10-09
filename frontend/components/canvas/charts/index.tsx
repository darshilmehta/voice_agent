"use client";

/** Which drawing a visual's `kind` gets, and which table is its "View as table" twin. */

import type { Model } from "@/lib/canvas/model";
import type { Visual } from "@/lib/canvas/types";

import { usePanel } from "../panel-context";
import { BarChart } from "./BarChart";
import { DataTable, EventsTable, TilesTable } from "./DataTable";
import { Donut } from "./Donut";
import { LineChart } from "./LineChart";
import { Comparison, Kpi } from "./Tiles";
import { Timeline } from "./Timeline";
import { Waterfall, waterfallTotals } from "./Waterfall";

export function CanvasChart({ visual, model, summaryId }: { visual: Visual; model: Model; summaryId: string }) {
  switch (visual.kind) {
    case "kpi":
      return <Kpi visual={visual} />;
    case "comparison":
      return <Comparison visual={visual} />;
    case "timeline":
      return <Timeline visual={visual} />;
    case "table":
      return <DataTable model={model} />;
    case "line":
      return <LineChart model={model} summaryId={summaryId} />;
    case "bar":
      return <BarChart model={model} variant="bar" summaryId={summaryId} />;
    case "grouped_bar":
      return <BarChart model={model} variant="grouped" summaryId={summaryId} />;
    case "stacked_bar":
      return <BarChart model={model} variant="stacked" summaryId={summaryId} />;
    case "waterfall":
      return <Waterfall model={model} summaryId={summaryId} />;
    case "donut":
      return <Donut model={model} summaryId={summaryId} />;
  }
}

/** The table that says exactly what a chart draws: every number, its unit, whether it is calculated, its source. */
export function TableView({ visual, model }: { visual: Visual; model: Model }) {
  const { labels } = usePanel();
  switch (visual.kind) {
    case "kpi":
      return <TilesTable visual={visual} />;
    case "timeline":
      return <EventsTable visual={visual} />;
    case "comparison":
      return (
        <>
          {visual.rows.length > 0 && visual.series.length >= 2 && <DataTable model={model} />}
          {visual.tiles.length > 0 && <TilesTable visual={visual} />}
        </>
      );
    case "waterfall": {
      const totals = waterfallTotals(visual);
      const values = visual.rows.map((r, i) => {
        const v = r.values[model.series[0]?.key ?? ""];
        if (totals[i]) return labels.total;
        return v === null || v === undefined ? "" : v >= 0 ? `▲ ${labels.increase}` : `▼ ${labels.decrease}`;
      });
      return <DataTable model={model} extra={{ header: "", values }} />;
    }
    default:
      return <DataTable model={model} />;
  }
}
