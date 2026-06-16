import type { ReactNode } from 'react';
import { AlertTriangle, CheckCircle2, Info, XCircle } from 'lucide-react';
import type { ReviewAnalysisMode } from '../../../api/reviews';

export type ReviewTab = 'overview' | 'retrieval' | 'debate' | 'findings';

export type DetailItem = {
  label: string;
  value: ReactNode;
};

export const severityColors: Record<string, string> = {
  critical: 'bg-red-500/15 text-red-400 border-red-500/30',
  high: 'bg-flame/15 text-flame border-flame/30',
  medium: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
  low: 'bg-sky-500/15 text-sky-400 border-sky-500/30',
  info: 'bg-midnight-lighter text-text-secondary border-surface-border',
};

export const metStatusIcons: Record<string, ReactNode> = {
  met: <CheckCircle2 size={16} className="text-emerald-400" />,
  not_met: <XCircle size={16} className="text-crimson" />,
  partially_met: <AlertTriangle size={16} className="text-amber-400" />,
  na: <Info size={16} className="text-text-muted" />,
  not_applicable: <Info size={16} className="text-text-muted" />,
};

export const ANALYSIS_MODE_OPTIONS: Array<{
  value: ReviewAnalysisMode;
  label: string;
  description: string;
}> = [
  {
    value: 'default',
    label: 'Default',
    description: 'Text and diagram analysis',
  },
  {
    value: 'text_only',
    label: 'Text Only',
    description: 'Requirement analysis only',
  },
  {
    value: 'diagram_only',
    label: 'Diagram Only',
    description: 'Diagram analysis only',
  },
];

export const REVIEW_TABS: Array<{ id: ReviewTab; label: string }> = [
  { id: 'overview', label: 'Overview' },
  { id: 'retrieval', label: 'Retrieval' },
  { id: 'debate', label: 'Multi-Agent Debate' },
  { id: 'findings', label: 'Findings' },
];

export const normalizeAnalysisMode = (value: unknown): ReviewAnalysisMode => {
  const mode = String(value || 'default').trim().toLowerCase();
  if (mode === 'text_only' || mode === 'diagram_only' || mode === 'default') {
    return mode;
  }
  return 'default';
};

export const formatAnalysisModeLabel = (mode: ReviewAnalysisMode) =>
  ANALYSIS_MODE_OPTIONS.find((option) => option.value === mode)?.label || 'Default';

export const isPresent = (value: unknown) => value !== null && value !== undefined && value !== '';

export const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

export const formatLabel = (value: string) =>
  value.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());

export const formatValue = (value: unknown): ReactNode => {
  if (typeof value === 'boolean') {
    return value ? 'Yes' : 'No';
  }
  if (typeof value === 'number') {
    return Number.isInteger(value) ? value : value.toFixed(2);
  }
  if (typeof value === 'string') {
    return value || '—';
  }
  if (Array.isArray(value)) {
    return value.length ? value.join(', ') : '—';
  }
  if (isRecord(value)) {
    return JSON.stringify(value);
  }
  return '—';
};

export const stringList = (value: unknown) =>
  Array.isArray(value)
    ? value.map((item) => String(item).trim()).filter(Boolean)
    : [];
