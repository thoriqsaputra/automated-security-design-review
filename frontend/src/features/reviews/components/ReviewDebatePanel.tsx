import Card from '../../../components/ui/Card';
import { isPresent } from '../utils/reviewPresentation';

interface ReviewDebatePanelProps {
  debatedTotal: number | null;
  debatedProcessed: number | null;
  debatedRemaining: number | null;
  persistenceTotal: number | null;
  persistenceProcessed: number | null;
  persistenceRemaining: number | null;
  skippedByParentApplicability: number | null;
}

export default function ReviewDebatePanel(props: ReviewDebatePanelProps) {
  const {
    debatedTotal,
    debatedProcessed,
    debatedRemaining,
    persistenceTotal,
    persistenceProcessed,
    persistenceRemaining,
    skippedByParentApplicability,
  } = props;

  if (
    !(
      isPresent(debatedTotal)
      || isPresent(debatedProcessed)
      || isPresent(debatedRemaining)
      || isPresent(persistenceTotal)
      || isPresent(persistenceProcessed)
      || isPresent(persistenceRemaining)
      || isPresent(skippedByParentApplicability)
    )
  ) {
    return <div className="space-y-6 animate-fade-in" />;
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <Card>
        <div className="flex items-start justify-between gap-4">
          <div>
            <h3 className="text-sm font-semibold text-text-primary">Analysis Progress</h3>
            <p className="mt-1 text-xs text-text-muted">
              Debate counts reflect post-ASVS and post-parent-applicability children. Persistence tracks final write-out after debate completes.
            </p>
          </div>
        </div>
        <div className="mt-4 grid grid-cols-2 gap-4 lg:grid-cols-4">
          {[
            { label: 'Debated Total', value: debatedTotal ?? '—' },
            { label: 'Debated Completed', value: debatedProcessed ?? '—' },
            { label: 'Debated Remaining', value: debatedRemaining ?? '—' },
            { label: 'Skipped By Parent Applicability', value: skippedByParentApplicability ?? '—' },
          ].map((item) => (
            <div key={item.label} className="rounded-lg border border-surface-border bg-midnight/30 p-3">
              <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider">{item.label}</p>
              <p className="mt-1 text-lg font-semibold text-text-primary">{item.value}</p>
            </div>
          ))}
        </div>
        <div className="mt-4 grid grid-cols-2 gap-4 lg:grid-cols-3">
          {[
            { label: 'Persistence Total', value: persistenceTotal ?? '—' },
            { label: 'Persistence Completed', value: persistenceProcessed ?? '—' },
            { label: 'Persistence Remaining', value: persistenceRemaining ?? '—' },
          ].map((item) => (
            <div key={item.label} className="rounded-lg border border-surface-border bg-midnight/30 p-3">
              <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider">{item.label}</p>
              <p className="mt-1 text-lg font-semibold text-text-primary">{item.value}</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}
