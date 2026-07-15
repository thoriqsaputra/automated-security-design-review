import { useMemo } from 'react';
import Card from '../../../components/ui/Card';
import type { DiagramExtractionSummary } from '../../../api/reviews';

interface DiagramStructurePanelProps {
  extraction: DiagramExtractionSummary | null | undefined;
}

export default function DiagramStructurePanel({ extraction }: DiagramStructurePanelProps) {
  const componentNameById = useMemo(() => {
    const map = new Map<string, string>();
    for (const component of extraction?.components || []) {
      map.set(component.id, component.name);
    }
    return map;
  }, [extraction]);

  if (
    !extraction ||
    (!extraction.components?.length && !extraction.trust_boundaries?.length && !extraction.flows?.length)
  ) {
    return null;
  }

  const components = extraction.components || [];
  const boundaries = extraction.trust_boundaries || [];
  const flows = extraction.flows || [];

  return (
    <Card className="mt-5">
      <div className="text-xs font-semibold uppercase tracking-[0.2em] text-text-muted">Diagram Structure</div>
      <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-3">
        <div>
          <h4 className="text-sm font-semibold text-text-primary">Components ({components.length})</h4>
          <ul className="mt-2 space-y-1">
            {components.length ? (
              components.map((component) => (
                <li key={component.id} className="text-sm text-text-secondary leading-relaxed">
                  {component.name} <span className="text-text-muted">({component.type})</span>
                </li>
              ))
            ) : (
              <li className="text-sm text-text-muted">None extracted.</li>
            )}
          </ul>
        </div>
        <div>
          <h4 className="text-sm font-semibold text-text-primary">Trust Boundaries ({boundaries.length})</h4>
          <ul className="mt-2 space-y-1">
            {boundaries.length ? (
              boundaries.map((boundary) => (
                <li key={boundary.id} className="text-sm text-text-secondary leading-relaxed">
                  {boundary.label}
                  {boundary.encloses_component_ids?.length ? (
                    <span className="text-text-muted">
                      {' '}
                      — encloses:{' '}
                      {boundary.encloses_component_ids
                        .map((id) => componentNameById.get(id) || id)
                        .join(', ')}
                    </span>
                  ) : null}
                </li>
              ))
            ) : (
              <li className="text-sm text-text-muted">None extracted.</li>
            )}
          </ul>
        </div>
        <div>
          <h4 className="text-sm font-semibold text-text-primary">Flows ({flows.length})</h4>
          <ul className="mt-2 space-y-1">
            {flows.length ? (
              flows.map((flow) => (
                <li key={flow.id} className="text-sm text-text-secondary leading-relaxed">
                  {componentNameById.get(flow.source_component_id) || flow.source_component_id}
                  {' → '}
                  {componentNameById.get(flow.target_component_id) || flow.target_component_id}
                  <span className="text-text-muted"> ({flow.protocol || flow.label || 'unlabeled'})</span>
                </li>
              ))
            ) : (
              <li className="text-sm text-text-muted">None extracted.</li>
            )}
          </ul>
        </div>
      </div>
    </Card>
  );
}
