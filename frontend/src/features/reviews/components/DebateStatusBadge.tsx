import StatusBadge from '../../../components/ui/StatusBadge';

interface DebateStatusBadgeProps {
  status: string;
}

export default function DebateStatusBadge({ status }: DebateStatusBadgeProps) {
  return <StatusBadge status={status} className="capitalize" />;
}
