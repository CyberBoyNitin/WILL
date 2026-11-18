"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";

import { API_URL, api, classNames } from "@/lib/api";

type Health = {
  status: string;
  device: string;
  default_checkpoint_id: string | null;
};

type Overview = {
  counts: {
    checkpoints: number;
    memory_sessions: number;
    jobs_running: number;
    benchmark_reports: number;
  };
};

const NAV = [
  { href: "/", label: "Overview", icon: "grid" },
  { href: "/model", label: "Model", icon: "layers" },
  { href: "/training", label: "Training", icon: "activity" },
  { href: "/benchmarks", label: "Benchmarks", icon: "bars" },
  { href: "/memory", label: "Memory", icon: "database" },
  { href: "/playground", label: "Playground", icon: "terminal" },
  { href: "/experiments", label: "Experiments", icon: "flask" },
] as const;

function NavIcon({ name }: { name: (typeof NAV)[number]["icon"] }) {
  const common = {
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.5,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
  };
  if (name === "grid") {
    return (
      <svg viewBox="0 0 24 24" {...common}>
        <rect x="3" y="3" width="7" height="7" />
        <rect x="14" y="3" width="7" height="7" />
        <rect x="3" y="14" width="7" height="7" />
        <rect x="14" y="14" width="7" height="7" />
      </svg>
    );
  }
  if (name === "layers") {
    return (
      <svg viewBox="0 0 24 24" {...common}>
        <path d="M12 3 21 8l-9 5-9-5 9-5Z" />
        <path d="m3 12 9 5 9-5" />
        <path d="m3 16 9 5 9-5" />
      </svg>
    );
  }
  if (name === "activity") {
    return (
      <svg viewBox="0 0 24 24" {...common}>
        <path d="M3 12h4l2-7 4 14 2-7h6" />
      </svg>
    );
  }
  if (name === "bars") {
    return (
      <svg viewBox="0 0 24 24" {...common}>
        <path d="M4 20V10M10 20V4M16 20v-7M22 20H2" />
      </svg>
    );
  }
  if (name === "database") {
    return (
      <svg viewBox="0 0 24 24" {...common}>
        <ellipse cx="12" cy="5" rx="8" ry="3" />
        <path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5" />
        <path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6" />
      </svg>
    );
  }
  if (name === "terminal") {
    return (
      <svg viewBox="0 0 24 24" {...common}>
        <rect x="3" y="4" width="18" height="16" rx="1" />
        <path d="m7 9 3 3-3 3M13 15h4" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 24 24" {...common}>
      <path d="M9 3h6M10 3v6L4 20a1 1 0 0 0 .9 1h14.2a1 1 0 0 0 .9-1L14 9V3" />
      <path d="M7 15h10" />
    </svg>
  );
}

export function Sidebar() {
  const pathname = usePathname();
  const [health, setHealth] = useState<Health | null>(null);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const [h, o] = await Promise.all([
          api<Health>("/api/health"),
          api<Overview>("/api/overview"),
        ]);
        if (!alive) return;
        setHealth(h);
        setOverview(o);
        setOffline(false);
      } catch {
        if (alive) setOffline(true);
      }
    };
    load();
    const timer = setInterval(load, 20000);
    return () => {
      alive = false;
      clearInterval(timer);
    };
  }, []);

  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark">
          <strong>WILL</strong>
          <span />
        </div>
        <p>adaptive recurrent lab</p>
      </div>
      <nav className="nav">
        <div className="nav-label">Research console</div>
        {NAV.map((item) => {
          const active =
            item.href === "/"
              ? pathname === "/"
              : pathname === item.href || pathname.startsWith(`${item.href}/`);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={classNames("nav-link", active && "active")}
            >
              <NavIcon name={item.icon} />
              {item.label}
            </Link>
          );
        })}
      </nav>
      <div className="sidebar-status">
        <div className="status-row">
          <span className={classNames("dot", offline ? "err" : health ? "ok" : "warn")} />
          {offline ? "API offline" : health ? "API online" : "connecting"}
        </div>
        <div className="status-row">
          <span className="dot" />
          {health?.device ? `device ${health.device}` : "device —"}
        </div>
        <div className="status-row">
          <span className="dot" />
          <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
            {health?.default_checkpoint_id || "no checkpoint"}
          </span>
        </div>
        <div className="status-row">
          <span className="dot" />
          {overview
            ? `${overview.counts.checkpoints} ckpt · ${overview.counts.memory_sessions} sessions`
            : "counts —"}
        </div>
        <div className="sidebar-foot">
          REAL EXECUTION ONLY
          <br />
          No simulated metrics. No energy claims.
          <br />
          <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer" style={{ color: "var(--mint)" }}>
            API reference ↗
          </a>
        </div>
      </div>
    </aside>
  );
}
