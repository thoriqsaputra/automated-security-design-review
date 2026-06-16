import type {
  DiagramRequirement,
  IngestionJob,
  ParameterParent,
} from '../../../api/standards';

export const parameterPageSizeOptions = [5, 10, 20, 50, 100];

export const selectArrowStyle = {
  backgroundImage:
    'url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns=\'http://www.w3.org/2000/svg\' viewBox=\'0 0 24 24\' fill=\'none\' stroke=\'currentColor\' stroke-width=\'2\' stroke-linecap=\'round\' stroke-linejoin=\'round\'%3e%3cpolyline points=\'6 9 12 15 18 9\'%3e%3c/polyline%3e%3c/svg%3e")',
  backgroundRepeat: 'no-repeat',
  backgroundPosition: 'right 0.25rem center',
  backgroundSize: '1em',
} as const;

export function asvsLevelClass(level: number | null) {
  if (level === 1) {
    return 'bg-emerald-500/15 text-emerald-400';
  }
  if (level === 2) {
    return 'bg-sky-500/15 text-sky-400';
  }
  if (level === 3) {
    return 'bg-fuchsia-500/15 text-fuchsia-400';
  }
  return 'bg-surface-hover text-text-muted';
}

export function asvsLevelLabel(level: number | null) {
  return level ? `L${level}` : 'Unknown';
}

export function countByAsvsLevel<T extends { asvs_level: number | null }>(items: T[]) {
  const byLevel: Record<string, number> = { L1: 0, L2: 0, L3: 0, Unknown: 0 };
  for (const item of items) {
    if (item.asvs_level === 1) {
      byLevel.L1 += 1;
    } else if (item.asvs_level === 2) {
      byLevel.L2 += 1;
    } else if (item.asvs_level === 3) {
      byLevel.L3 += 1;
    } else {
      byLevel.Unknown += 1;
    }
  }
  return { total: items.length, byLevel };
}

export function parentLevelSummary(parent: ParameterParent) {
  const counts = parent.children.reduce<Record<string, number>>((accumulator, child) => {
    const key = asvsLevelLabel(child.asvs_level);
    accumulator[key] = (accumulator[key] || 0) + 1;
    return accumulator;
  }, {});
  return ['L1', 'L2', 'L3', 'Unknown']
    .filter((key) => counts[key])
    .map((key) => `${key}: ${counts[key]}`)
    .join(' · ');
}

export function getAsvsSummaryForJob(job: IngestionJob) {
  const summary = (job.summary_json as { asvs_level_definitions?: { status?: string; count?: number } }).asvs_level_definitions;
  const status = summary?.status || 'pending';
  const count = Number(summary?.count || 0);
  const extracted = status === 'extracted' && count > 0;
  const statusClass = extracted
    ? 'bg-emerald-500/15 text-emerald-400'
    : status === 'fallback'
      ? 'bg-amber-500/15 text-amber-400'
      : 'bg-surface-hover text-text-muted';
  return { status, count, extracted, statusClass };
}

export function flattenParameterChildren(parameters: ParameterParent[]) {
  return parameters.flatMap((parent) => parent.children);
}

export function countDiagramRequirements(diagramRequirements: DiagramRequirement[]) {
  return countByAsvsLevel(diagramRequirements);
}
