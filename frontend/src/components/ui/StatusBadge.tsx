import React from 'react';

const statusConfig: Record<string, { bg: string; text: string; dot: string }> = {
  pending:   { bg: 'bg-midnight-lighter', text: 'text-text-secondary', dot: 'bg-text-muted' },
  running:   { bg: 'bg-flame/15', text: 'text-flame', dot: 'bg-flame animate-pulse' },
  completed: { bg: 'bg-emerald-500/15', text: 'text-emerald-400', dot: 'bg-emerald-400' },
  failed:    { bg: 'bg-crimson/15', text: 'text-crimson', dot: 'bg-crimson' },
  ready:     { bg: 'bg-emerald-500/15', text: 'text-emerald-400', dot: 'bg-emerald-400' },
  approved:  { bg: 'bg-sky-500/15', text: 'text-sky-400', dot: 'bg-sky-400' },
  completed_with_findings: { bg: 'bg-flame/15', text: 'text-flame', dot: 'bg-flame' },
  completed_clean: { bg: 'bg-emerald-500/15', text: 'text-emerald-400', dot: 'bg-emerald-400' },
  cancelled: { bg: 'bg-amber-500/15', text: 'text-amber-500', dot: 'bg-amber-500' },
};

interface Props {
  status: string;
  className?: string;
}

export default function StatusBadge({ status, className = '' }: Props) {
  const s = status?.toLowerCase().replace(/ /g, '_');
  const config = statusConfig[s] || statusConfig.pending;

  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${config.bg} ${config.text} ${className}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${config.dot}`} />
      {status.replace(/_/g, ' ')}
    </span>
  );
}
