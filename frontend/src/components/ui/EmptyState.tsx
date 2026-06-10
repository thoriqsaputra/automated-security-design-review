import React from 'react';
import { Inbox } from 'lucide-react';

interface Props {
  icon?: React.ReactNode;
  title: string;
  description?: string;
  action?: React.ReactNode;
}

export default function EmptyState({ icon, title, description, action }: Props) {
  return (
    <div className="flex flex-col items-center justify-center py-16 animate-fade-in">
      <div className="p-4 rounded-2xl bg-surface-card border border-surface-border mb-4">
        {icon || <Inbox size={32} className="text-text-muted" />}
      </div>
      <h3 className="text-lg font-semibold text-text-primary mb-1">{title}</h3>
      {description && <p className="text-sm text-text-muted mb-4 max-w-sm text-center">{description}</p>}
      {action}
    </div>
  );
}
