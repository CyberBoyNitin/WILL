"use client";

import { fmtCompact } from "@/lib/api";

export interface Series {
  label: string;
  values: Array<number | null>;
  color?: string;
}

const PALETTE = ["#7cf5c8", "#64d8ff", "#f6c177", "#b8a1ff", "#ff7b72"];

function pathFor(values: Array<number | null>, x: (i: number) => number, y: (v: number) => number) {
  let path = "";
  let open = false;
  values.forEach((value, index) => {
    if (value === null || value === undefined || !Number.isFinite(value)) {
      open = false;
      return;
    }
    path += `${open ? "L" : "M"}${x(index).toFixed(2)},${y(value).toFixed(2)} `;
    open = true;
  });
  return path.trim();
}

export function LineChart({
  series,
  height = 210,
  yFormat = (value) => fmtCompact(value),
  xLabels,
  zeroBase = false,
}: {
  series: Series[];
  height?: number;
  yFormat?: (value: number) => string;
  xLabels?: string[];
  zeroBase?: boolean;
}) {
  const width = 680;
  const margin = { left: 44, right: 12, top: 12, bottom: 24 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;
  const values = series.flatMap((s) =>
    s.values.filter((v): v is number => v !== null && v !== undefined && Number.isFinite(v)),
  );
  if (!values.length) {
    return <div className="empty">No measured points yet.</div>;
  }
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (zeroBase) min = Math.min(0, min);
  if (max === min) {
    max = min + 1;
    min = min - 1;
  }
  const pad = (max - min) * 0.08;
  max += pad;
  min -= pad;
  const n = Math.max(...series.map((s) => s.values.length), 2);
  const x = (index: number) =>
    margin.left + (n <= 1 ? 0 : (index / (n - 1)) * plotW);
  const y = (value: number) =>
    margin.top + plotH - ((value - min) / (max - min)) * plotH;
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((fraction) => min + (max - min) * fraction);
  const labelIndexes = [0, Math.floor((n - 1) / 2), n - 1];

  return (
    <div>
      <svg
        className="chart"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        role="img"
        aria-label="line chart"
      >
        {ticks.map((tick) => (
          <g key={tick}>
            <line
              className="grid-line"
              x1={margin.left}
              x2={width - margin.right}
              y1={y(tick)}
              y2={y(tick)}
            />
            <text x={margin.left - 7} y={y(tick) + 3} textAnchor="end">
              {yFormat(tick)}
            </text>
          </g>
        ))}
        <line
          className="axis-line"
          x1={margin.left}
          x2={margin.left}
          y1={margin.top}
          y2={margin.top + plotH}
        />
        <line
          className="axis-line"
          x1={margin.left}
          x2={width - margin.right}
          y1={margin.top + plotH}
          y2={margin.top + plotH}
        />
        {series.map((item, index) => {
          const color = item.color || PALETTE[index % PALETTE.length];
          return (
            <path
              key={item.label}
              d={pathFor(item.values, x, y)}
              fill="none"
              stroke={color}
              strokeWidth={1.8}
              vectorEffect="non-scaling-stroke"
            />
          );
        })}
        {labelIndexes.map((index) => (
          <text
            key={index}
            x={x(index)}
            y={height - 6}
            textAnchor={index === 0 ? "start" : index === n - 1 ? "end" : "middle"}
          >
            {xLabels?.[index] ?? String(index + 1)}
          </text>
        ))}
      </svg>
      <div className="legend">
        {series.map((item, index) => (
          <span className="legend-item" key={item.label}>
            <span
              className="legend-swatch"
              style={{ background: item.color || PALETTE[index % PALETTE.length] }}
            />
            {item.label}
          </span>
        ))}
      </div>
    </div>
  );
}

export function BarRows({
  rows,
  color = "#7cf5c8",
  format = (value: number) => fmtCompact(value),
}: {
  rows: Array<{ label: string; value: number }>;
  color?: string;
  format?: (value: number) => string;
}) {
  const max = Math.max(1, ...rows.map((row) => row.value));
  return (
    <div>
      {rows.map((row) => (
        <div className="bar-row" key={row.label}>
          <span className="mono" style={{ color: "var(--muted)" }}>
            {row.label}
          </span>
          <span className="bar-track">
            <span
              className="bar-fill"
              style={{ width: `${(row.value / max) * 100}%`, background: color }}
            />
          </span>
          <span className="bar-value">{format(row.value)}</span>
        </div>
      ))}
    </div>
  );
}

export function DepthStrip({
  depths,
  tokens,
}: {
  depths: number[];
  tokens?: string[];
}) {
  const max = Math.max(1, ...depths);
  if (!depths.length) return <div className="empty">No token depth data.</div>;
  return (
    <div
      style={{
        display: "flex",
        alignItems: "flex-end",
        gap: 2,
        minHeight: 64,
        padding: "8px 0",
        overflowX: "auto",
      }}
    >
      {depths.map((depth, index) => {
        const ratio = depth / max;
        const color = depth <= 1 ? "#7cf5c8" : depth <= 2 ? "#64d8ff" : depth <= 3 ? "#f6c177" : "#ff7b72";
        return (
          <span
            key={`${index}-${depth}`}
            title={`${tokens?.[index] ?? index}: depth ${depth.toFixed(2)}`}
            style={{
              flex: "0 0 auto",
              width: 5,
              height: `${Math.max(6, ratio * 54)}px`,
              background: color,
              opacity: 0.82,
              borderRadius: 1,
            }}
          />
        );
      })}
    </div>
  );
}
