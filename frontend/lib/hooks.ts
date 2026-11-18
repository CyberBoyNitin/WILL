"use client";

import { useCallback, useEffect, useState } from "react";

import { api } from "@/lib/api";
import type { Job, JobEvent } from "@/lib/types";

export function useApi<T = any>(path: string | null, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(Boolean(path));
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (!path) {
      setData(null);
      setLoading(false);
      setError(null);
      return;
    }
    let alive = true;
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    api<T>(path, { signal: controller.signal })
      .then((value) => {
        if (alive) setData(value);
      })
      .catch((err: Error) => {
        if (alive && err.name !== "AbortError") setError(err.message);
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
      controller.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, tick, ...deps]);

  const reload = useCallback(() => setTick((value) => value + 1), []);
  return { data, loading, error, reload };
}

const TERMINAL = new Set([
  "completed",
  "failed",
  "cancelled",
  "interrupted",
]);

export function useJob(kind: "training" | "benchmarks", jobId: string | null) {
  const [job, setJob] = useState<Job | null>(null);
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) {
      setJob(null);
      setEvents([]);
      setError(null);
      return;
    }
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const poll = async () => {
      try {
        const payload = await api<{ job: Job; events: JobEvent[] }>(
          `/api/${kind}/jobs/${jobId}`,
        );
        if (!alive) return;
        setJob(payload.job);
        setEvents(payload.events || []);
        setError(null);
        if (!TERMINAL.has(payload.job.status)) {
          timer = setTimeout(poll, 1200);
        }
      } catch (err) {
        if (!alive) return;
        setError((err as Error).message);
        timer = setTimeout(poll, 2500);
      }
    };
    poll();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [kind, jobId]);

  return { job, events, error };
}

export function useNow(intervalMs = 30000) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), intervalMs);
    return () => clearInterval(timer);
  }, [intervalMs]);
  return now;
}
