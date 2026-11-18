"use client";

import type { ReactNode } from "react";
import { useState } from "react";

import { classNames, statusTone } from "@/lib/api";

export function PageHeader({
  kicker,
  title,
  description,
  actions,
}: {
  kicker: string;
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
}) {
  return (
    <header className="page-head">
      <div>
        <div className="kicker">{kicker}</div>
        <h1>{title}</h1>
        {description ? <p>{description}</p> : null}
      </div>
      {actions ? <div className="page-actions">{actions}</div> : null}
    </header>
  );
}

export function Panel({
  title,
  subtitle,
  actions,
  children,
  className,
  flush,
  tight,
}: {
  title?: ReactNode;
  subtitle?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  flush?: boolean;
  tight?: boolean;
}) {
  return (
    <section
      className={classNames("panel", flush && "flush", tight && "tight", className)}
    >
      {title || actions ? (
        <div className="panel-head">
          <div>
            {title ? <h2 className="panel-title">{title}</h2> : null}
            {subtitle ? <p className="panel-sub">{subtitle}</p> : null}
          </div>
          {actions ? <div className="row">{actions}</div> : null}
        </div>
      ) : null}
      {children}
    </section>
  );
}

export function Stat({
  label,
  value,
  foot,
  tone,
  small,
}: {
  label: string;
  value: ReactNode;
  foot?: ReactNode;
  tone?: "mint" | "amber" | "cyan";
  small?: ReactNode;
}) {
  return (
    <div className={classNames("stat", tone)}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">
        {value}
        {small ? <small>{small}</small> : null}
      </div>
      {foot ? <div className="stat-foot">{foot}</div> : null}
    </div>
  );
}

export function Badge({
  children,
  tone,
}: {
  children: ReactNode;
  tone?: "mint" | "amber" | "red" | "cyan" | "";
}) {
  return <span className={classNames("badge", tone)}>{children}</span>;
}

export function StatusBadge({ status }: { status: string }) {
  return <Badge tone={statusTone(status)}>{status}</Badge>;
}

export function Btn({
  children,
  onClick,
  variant = "default",
  size,
  disabled,
  type = "button",
  title,
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "default" | "primary" | "ghost" | "danger";
  size?: "sm";
  disabled?: boolean;
  type?: "button" | "submit";
  title?: string;
}) {
  return (
    <button
      type={type}
      className={classNames("btn", variant !== "default" && variant, size)}
      onClick={onClick}
      disabled={disabled}
      title={title}
    >
      {children}
    </button>
  );
}

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: ReactNode;
  children: ReactNode;
}) {
  return (
    <div className="field">
      <label>{label}</label>
      {children}
      {hint ? <div className="hint">{hint}</div> : null}
    </div>
  );
}

export function KV({
  rows,
}: {
  rows: Array<[string, ReactNode]>;
}) {
  return (
    <div className="kv">
      {rows.map(([key, value]) => (
        <div key={key} style={{ display: "contents" }}>
          <div className="k">{key}</div>
          <div className="v">{value}</div>
        </div>
      ))}
    </div>
  );
}

export function Empty({ children }: { children?: ReactNode }) {
  return <div className="empty">{children || "No data yet."}</div>;
}

export function ErrorBox({
  error,
  onRetry,
}: {
  error: string;
  onRetry?: () => void;
}) {
  return (
    <div className="error-box">
      <div>{error}</div>
      <div className="mono" style={{ marginTop: 6 }}>
        Check that the FastAPI server is running and NEXT_PUBLIC_API_URL is correct.
      </div>
      {onRetry ? (
        <div style={{ marginTop: 10 }}>
          <Btn size="sm" onClick={onRetry}>
            Retry
          </Btn>
        </div>
      ) : null}
    </div>
  );
}

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="loading-inline">
      <span className="spinner" />
      {label}
    </div>
  );
}

export function CopyButton({ value, label = "Copy" }: { value: string; label?: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <Btn
      size="sm"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(value);
          setCopied(true);
          setTimeout(() => setCopied(false), 1200);
        } catch {
          setCopied(false);
        }
      }}
    >
      {copied ? "Copied" : label}
    </Btn>
  );
}

export function ProgressBar({ value }: { value: number }) {
  return (
    <div className="progress">
      <span style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
    </div>
  );
}

export function Notice({ children }: { children: ReactNode }) {
  return <div className="notice">{children}</div>;
}
