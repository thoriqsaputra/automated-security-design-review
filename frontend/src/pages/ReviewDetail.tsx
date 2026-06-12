import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Play, AlertTriangle, CheckCircle2, XCircle, Info, Network, Waypoints } from 'lucide-react';
import Card from '../components/ui/Card';
import StatusBadge from '../components/ui/StatusBadge';
import ReviewPipeline from '../components/flow/ReviewPipeline';
import RaptorTreeView from '../components/flow/RaptorTreeView';
import GraphRagView from '../components/flow/GraphRagView';
import {
  getReview,
  getFindings,
  getRetrievalVisualization,
  triggerReview,
  cancelReview,
  type Review,
  type Finding,
  type RetrievalVisualization,
} from '../api/reviews';

const severityColors: Record<string, string> = {
  critical: 'bg-red-500/15 text-red-400 border-red-500/30',
  high: 'bg-flame/15 text-flame border-flame/30',
  medium: 'bg-amber-500/15 text-amber-400 border-amber-500/30',
  low: 'bg-sky-500/15 text-sky-400 border-sky-500/30',
  info: 'bg-midnight-lighter text-text-secondary border-surface-border',
};

const metStatusIcons: Record<string, React.ReactNode> = {
  met: <CheckCircle2 size={16} className="text-emerald-400" />,
  not_met: <XCircle size={16} className="text-crimson" />,
  partially_met: <AlertTriangle size={16} className="text-amber-400" />,
  na: <Info size={16} className="text-text-muted" />,
  not_applicable: <Info size={16} className="text-text-muted" />,
};

type DetailItem = {
  label: string;
  value: React.ReactNode;
};

const isPresent = (value: unknown) => value !== null && value !== undefined && value !== '';

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

const formatLabel = (value: string) =>
  value.replace(/_/g, ' ').replace(/\b\w/g, char => char.toUpperCase());

const formatValue = (value: unknown): React.ReactNode => {
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'number') return Number.isInteger(value) ? value : value.toFixed(2);
  if (typeof value === 'string') return value || '—';
  if (Array.isArray(value)) return value.length ? value.join(', ') : '—';
  if (isRecord(value)) return JSON.stringify(value);
  return '—';
};

const stringList = (value: unknown) =>
  Array.isArray(value)
    ? value.map(item => String(item).trim()).filter(Boolean)
    : [];

function TextBlock({ title, children }: { title: string; children: React.ReactNode }) {
  if (!isPresent(children)) return null;
  return (
    <section>
      <p className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1">{title}</p>
      <div className="text-sm text-text-secondary leading-relaxed">{children}</div>
    </section>
  );
}

function FieldGrid({ items }: { items: DetailItem[] }) {
  const visibleItems = items.filter(item => isPresent(item.value));
  if (visibleItems.length === 0) return null;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
      {visibleItems.map(item => (
        <div key={item.label} className="bg-midnight/30 p-3 rounded-lg border border-surface-border">
          <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider">{item.label}</p>
          <div className="mt-1 text-sm text-text-primary break-words">{item.value}</div>
        </div>
      ))}
    </div>
  );
}

function ChipList({ items }: { items: string[] }) {
  if (items.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map(item => (
        <span key={item} className="rounded-full border border-surface-border bg-midnight px-2 py-1 text-xs text-text-secondary">
          {formatLabel(item)}
        </span>
      ))}
    </div>
  );
}

function BulletList({ items }: { items: string[] }) {
  if (items.length === 0) return null;
  return (
    <ul className="space-y-1 text-sm text-text-secondary">
      {items.map(item => (
        <li key={item} className="rounded-lg border border-surface-border bg-midnight/30 px-3 py-2">
          {item}
        </li>
      ))}
    </ul>
  );
}

function JsonPanel({ title, value }: { title: string; value: unknown }) {
  if (!isPresent(value)) return null;
  return (
    <details className="rounded-lg border border-surface-border bg-midnight/50">
      <summary className="cursor-pointer px-3 py-2 text-xs font-semibold text-text-muted uppercase tracking-wider">
        {title}
      </summary>
      <pre className="max-h-72 overflow-auto border-t border-surface-border px-3 py-2 text-xs text-text-secondary whitespace-pre-wrap">
        {JSON.stringify(value, null, 2)}
      </pre>
    </details>
  );
}

function MetadataSection({ metadata }: { metadata: Record<string, unknown> | null }) {
  if (!metadata || Object.keys(metadata).length === 0) return null;

  const required = stringList(metadata.required_capabilities);
  const optional = stringList(metadata.optional_capabilities);
  const negative = stringList(metadata.negative_scope_signals);
  const keywords = stringList(metadata.evidence_keywords);
  const applicable = stringList(metadata.when_applicable);
  const notApplicable = stringList(metadata.when_not_applicable);

  return (
    <section className="space-y-3 rounded-xl border border-surface-border bg-surface-base/50 p-4">
      <div>
        <p className="text-xs font-semibold text-text-muted uppercase tracking-wider">Applicability Metadata</p>
        {isPresent(metadata.scope_summary) && (
          <p className="mt-1 text-sm text-text-secondary">{String(metadata.scope_summary)}</p>
        )}
      </div>

      <FieldGrid
        items={[
          { label: 'Control Domain', value: formatValue(metadata.control_domain) },
          { label: 'Scope Type', value: formatValue(metadata.scope_type) },
          { label: 'Generation Source', value: formatValue(metadata.generation_source) },
          { label: 'Generation Version', value: formatValue(metadata.generation_version) },
        ]}
      />

      {(required.length > 0 || optional.length > 0 || negative.length > 0 || keywords.length > 0) && (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          <div>
            <p className="mb-1 text-[10px] font-semibold text-text-muted uppercase tracking-wider">Required Capabilities</p>
            <ChipList items={required} />
          </div>
          <div>
            <p className="mb-1 text-[10px] font-semibold text-text-muted uppercase tracking-wider">Optional Capabilities</p>
            <ChipList items={optional} />
          </div>
          <div>
            <p className="mb-1 text-[10px] font-semibold text-text-muted uppercase tracking-wider">Negative Scope Signals</p>
            <ChipList items={negative} />
          </div>
          <div>
            <p className="mb-1 text-[10px] font-semibold text-text-muted uppercase tracking-wider">Evidence Keywords</p>
            <ChipList items={keywords} />
          </div>
        </div>
      )}

      {(applicable.length > 0 || notApplicable.length > 0) && (
        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
          <div>
            <p className="mb-1 text-[10px] font-semibold text-text-muted uppercase tracking-wider">When Applicable</p>
            <BulletList items={applicable} />
          </div>
          <div>
            <p className="mb-1 text-[10px] font-semibold text-text-muted uppercase tracking-wider">When Not Applicable</p>
            <BulletList items={notApplicable} />
          </div>
        </div>
      )}

      <JsonPanel title="Raw Requirement Metadata" value={metadata} />
    </section>
  );
}

function SeverityAnalysisSection({ analysis }: { analysis: Record<string, unknown> | null }) {
  if (!analysis || Object.keys(analysis).length === 0) return null;

  const isV2 = analysis.version === 'v2' || analysis.source === 'deterministic_risk_rubric';
  if (isV2) {
    const dimensions = isRecord(analysis.dimensions) ? analysis.dimensions : {};
    const domainBase = isRecord(dimensions.domain_base) ? dimensions.domain_base : {};
    const impact = isRecord(dimensions.impact_capabilities) ? dimensions.impact_capabilities : {};
    const exposure = isRecord(dimensions.exposure_capabilities) ? dimensions.exposure_capabilities : {};
    const keywords = isRecord(dimensions.keyword_signals) ? dimensions.keyword_signals : {};
    const confidence = isRecord(dimensions.confidence_adjustment) ? dimensions.confidence_adjustment : {};
    const caps = Array.isArray(analysis.caps)
      ? analysis.caps.filter(isRecord)
      : [];
    const matchedImpact = stringList(impact.matched);
    const matchedExposure = stringList(exposure.matched);
    const matchedKeywords = stringList(keywords.matched);
    const appliedCaps = caps.filter(cap => cap.applied);

    return (
      <section className="space-y-3 rounded-xl border border-surface-border bg-midnight/30 p-4">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <p className="text-xs font-semibold text-text-muted uppercase tracking-wider">Severity Analysis</p>
            {isPresent(analysis.explanation) && (
              <p className="mt-1 text-sm text-text-secondary leading-relaxed">{String(analysis.explanation)}</p>
            )}
          </div>
          <div className="rounded-xl border border-flame/30 bg-flame/10 px-3 py-2 text-right">
            <p className="text-[10px] font-semibold text-flame uppercase tracking-wider">{formatValue(analysis.final_severity)}</p>
            <p className="text-lg font-semibold text-text-primary">{formatValue(analysis.final_score)}</p>
          </div>
        </div>

        <FieldGrid
          items={[
            { label: 'Rubric', value: formatValue(analysis.version) },
            { label: 'Domain Base', value: `${formatValue(domainBase.score)} (${formatValue(domainBase.domain)})` },
            { label: 'Impact Delta', value: formatValue(impact.delta) },
            { label: 'Exposure Delta', value: formatValue(exposure.delta) },
            { label: 'Keyword Delta', value: formatValue(keywords.delta) },
            { label: 'Confidence Delta', value: formatValue(confidence.delta) },
            { label: 'Confidence', value: isPresent(confidence.confidence_score) ? `${(Number(confidence.confidence_score) * 100).toFixed(0)}%` : null },
          ]}
        />

        {(matchedImpact.length > 0 || matchedExposure.length > 0 || matchedKeywords.length > 0) && (
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
            <div>
              <p className="mb-1 text-[10px] font-semibold text-text-muted uppercase tracking-wider">Impact Signals</p>
              <ChipList items={matchedImpact} />
            </div>
            <div>
              <p className="mb-1 text-[10px] font-semibold text-text-muted uppercase tracking-wider">Exposure Signals</p>
              <ChipList items={matchedExposure} />
            </div>
            <div>
              <p className="mb-1 text-[10px] font-semibold text-text-muted uppercase tracking-wider">Risk Keywords</p>
              <ChipList items={matchedKeywords} />
            </div>
          </div>
        )}

        {caps.length > 0 && (
          <div>
            <p className="mb-1 text-[10px] font-semibold text-text-muted uppercase tracking-wider">Caps</p>
            <div className="flex flex-wrap gap-1.5">
              {caps.map(cap => (
                <span
                  key={`${String(cap.kind)}-${String(cap.cap)}`}
                  className={`rounded-full border px-2 py-1 text-xs ${cap.applied
                    ? 'border-amber-500/30 bg-amber-500/10 text-amber-300'
                    : 'border-surface-border bg-midnight text-text-muted'
                    }`}
                >
                  {formatLabel(String(cap.kind || 'cap'))}: {formatValue(cap.cap)}
                </span>
              ))}
            </div>
            {appliedCaps.length > 0 && (
              <p className="mt-1 text-xs text-text-muted">
                Applied {appliedCaps.length} cap(s) to avoid overstating uncertain findings.
              </p>
            )}
          </div>
        )}

        <JsonPanel title="Raw Severity Analysis" value={analysis} />
      </section>
    );
  }

  const impact = isRecord(analysis.impact) ? analysis.impact : {};
  const affectedComponents = stringList(analysis.affected_components);

  return (
    <section className="space-y-3 rounded-xl border border-surface-border bg-midnight/30 p-4">
      <p className="text-xs font-semibold text-text-muted uppercase tracking-wider">Severity Analysis</p>
      <FieldGrid
        items={[
          { label: 'Severity', value: formatValue(analysis.severity) },
          { label: 'Severity Score', value: formatValue(analysis.severity_score) },
          { label: 'Likelihood', value: formatValue(analysis.likelihood) },
          { label: 'Attack Vector', value: formatValue(analysis.attack_vector) },
          { label: 'Confidentiality', value: formatValue(impact.confidentiality) },
          { label: 'Integrity', value: formatValue(impact.integrity) },
          { label: 'Availability', value: formatValue(impact.availability) },
        ]}
      />
      {affectedComponents.length > 0 && (
        <div>
          <p className="mb-1 text-[10px] font-semibold text-text-muted uppercase tracking-wider">Affected Components</p>
          <ChipList items={affectedComponents} />
        </div>
      )}
      {isPresent(analysis.justification) && (
        <p className="text-sm text-text-secondary leading-relaxed">{String(analysis.justification)}</p>
      )}
      <JsonPanel title="Raw Severity Analysis" value={analysis} />
    </section>
  );
}

function FindingDetails({ finding: f }: { finding: Finding }) {
  const auditItems = [
    { label: 'Finding ID', value: f.id },
    { label: 'Review ID', value: f.review_id },
    { label: 'Category ID', value: f.category_id },
    { label: 'Parent Parameter ID', value: f.parent_parameter_id },
    { label: 'Child Parameter ID', value: f.child_parameter_id },
    { label: 'Has Citations', value: f.has_citations ? 'Yes' : 'No' },
    { label: 'Citation Count', value: f.citation_count },
    { label: 'Created At', value: new Date(f.created_at).toLocaleString() },
    { label: 'Updated At', value: new Date(f.updated_at).toLocaleString() },
  ];

  return (
    <div className="mt-4 space-y-4 border-t border-surface-border pt-4 md:pl-9">
      <TextBlock title="Description">{f.description}</TextBlock>
      <FieldGrid
        items={[
          { label: 'Met Status', value: f.met_status ? formatLabel(f.met_status) : null },
          { label: 'Severity', value: f.severity ? formatLabel(f.severity) : null },
          { label: 'Confidence', value: f.confidence_score !== null ? `${(f.confidence_score * 100).toFixed(0)}%` : null },
          { label: 'Severity Score', value: f.severity_score !== null ? f.severity_score.toFixed(1) : null },
          { label: 'Finding Type', value: formatLabel(f.finding_type) },
          { label: 'Actionable', value: f.is_actionable ? 'Yes' : 'No' },
        ]}
      />

      <TextBlock title="Reason">{f.reason}</TextBlock>
      <TextBlock title="Recommendation">{f.recommendation}</TextBlock>

      <section className="space-y-3 rounded-xl border border-surface-border bg-surface-base/50 p-4">
        <p className="text-xs font-semibold text-text-muted uppercase tracking-wider">Requirement Traceability</p>
        <FieldGrid
          items={[
            { label: 'Reference', value: f.requirement_reference },
            { label: 'Category', value: [f.category_code, f.category_name].filter(Boolean).join(' - ') },
            { label: 'Parent Parameter', value: f.parent_parameter_title },
            { label: 'Child Stable Key', value: f.child_parameter_stable_key },
            { label: 'Child Ordinal', value: f.child_parameter_ordinal },
          ]}
        />
        <TextBlock title="Requirement Text">
          {f.requirement_text && (
            <span className="italic">
              {f.requirement_reference && <span className="font-semibold text-text-primary mr-2">[{f.requirement_reference}]</span>}
              {f.requirement_text}
            </span>
          )}
        </TextBlock>
      </section>

      <MetadataSection metadata={f.requirement_metadata} />
      <SeverityAnalysisSection analysis={f.severity_analysis} />

      {(f.diagram_id || f.diagram_caption || f.diagram_image_url || f.vision_reasoning || f.vision_thought_process) && (
        <section className="space-y-3 rounded-xl border border-surface-border bg-midnight/30 p-4">
          <p className="text-xs font-semibold text-text-muted uppercase tracking-wider">Diagram And Vision Analysis</p>
          {f.diagram_image_url && (
            <div className="overflow-hidden rounded-xl border border-surface-border bg-surface-base">
              <div className="flex items-center justify-between gap-3 border-b border-surface-border px-3 py-2">
                <div>
                  <p className="text-sm font-semibold text-text-primary">Related Diagram</p>
                  {f.diagram_caption && <p className="text-xs text-text-muted">{f.diagram_caption}</p>}
                </div>
                <a
                  href={f.diagram_image_url}
                  target="_blank"
                  rel="noreferrer"
                  className="shrink-0 rounded-lg border border-surface-border px-2 py-1 text-xs font-semibold text-text-secondary hover:text-text-primary"
                >
                  Open image
                </a>
              </div>
              <div className="flex max-h-[520px] items-center justify-center overflow-auto bg-midnight/40 p-3">
                <img
                  src={f.diagram_image_url}
                  alt={f.diagram_caption || f.diagram_id || 'Related diagram'}
                  className="max-h-[480px] max-w-full rounded-lg object-contain"
                  loading="lazy"
                />
              </div>
            </div>
          )}
          <FieldGrid
            items={[
              { label: 'Diagram ID', value: f.diagram_id },
              { label: 'Diagram Caption', value: f.diagram_caption },
            ]}
          />
          <TextBlock title="Vision Reasoning">{f.vision_reasoning}</TextBlock>
          <TextBlock title="Vision Thought Process">{f.vision_thought_process}</TextBlock>
        </section>
      )}

      {(f.hunter_reasoning || f.critic_reasoning || f.mediator_reasoning || f.hunter_thought_process || f.critic_thought_process || f.mediator_thought_process) && (
        <section className="space-y-3 rounded-xl border border-surface-border bg-surface-base/50 p-4">
          <p className="text-xs font-semibold text-text-muted uppercase tracking-wider">Agent Audit</p>
          <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
            {[
              { role: 'Hunter', color: 'text-flame', reasoning: f.hunter_reasoning, thought: f.hunter_thought_process },
              { role: 'Critic', color: 'text-burgundy-light', reasoning: f.critic_reasoning, thought: f.critic_thought_process },
              { role: 'Mediator', color: 'text-emerald-400', reasoning: f.mediator_reasoning, thought: f.mediator_thought_process },
            ].map(agent => (
              <div key={agent.role} className="space-y-2 rounded-lg border border-surface-border bg-midnight p-3">
                <p className={`text-[10px] font-bold uppercase tracking-wider ${agent.color}`}>{agent.role}</p>
                <TextBlock title="Reasoning">{agent.reasoning}</TextBlock>
                <TextBlock title="Thought Process">{agent.thought}</TextBlock>
              </div>
            ))}
          </div>
        </section>
      )}

      {f.citations && f.citations.length > 0 && (
        <section className="space-y-2 rounded-xl border border-surface-border bg-surface-base/50 p-4">
          <p className="text-xs font-semibold text-text-muted uppercase tracking-wider">Evidence Citations</p>
          <div className="space-y-2">
            {f.citations.map(citation => (
              <div key={citation.id} className="bg-surface-base border border-surface-border rounded p-3">
                <p className="text-[10px] font-mono text-text-muted mb-1 flex flex-wrap items-center gap-2">
                  <span className="bg-midnight px-1.5 py-0.5 rounded">Page {citation.page_number}</span>
                  <span className="bg-midnight px-1.5 py-0.5 rounded">{citation.anchor_type}</span>
                  <span className="opacity-70 break-all">{citation.block_id}</span>
                </p>
                {citation.quoted_text && (
                  <p className="text-sm text-text-secondary italic border-l-2 border-flame/50 pl-3 py-0.5">
                    "{citation.quoted_text}"
                  </p>
                )}
                <FieldGrid
                  items={[
                    { label: 'X0', value: citation.bbox_x0 },
                    { label: 'Y0', value: citation.bbox_y0 },
                    { label: 'X1', value: citation.bbox_x1 },
                    { label: 'Y1', value: citation.bbox_y1 },
                    { label: 'Created At', value: new Date(citation.created_at).toLocaleString() },
                  ]}
                />
              </div>
            ))}
          </div>
        </section>
      )}

      <section className="space-y-3 rounded-xl border border-surface-border bg-midnight/30 p-4">
        <p className="text-xs font-semibold text-text-muted uppercase tracking-wider">Raw Audit</p>
        <FieldGrid items={auditItems} />
        <JsonPanel title="Raw Finding Payload" value={f} />
      </section>
    </div>
  );
}

export default function ReviewDetail() {
  const { designId, id } = useParams<{ designId: string; id: string }>();
  const navigate = useNavigate();
  const [review, setReview] = useState<Review | null>(null);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [retrievalVisualization, setRetrievalVisualization] = useState<RetrievalVisualization | null>(null);
  const [loading, setLoading] = useState(true);
  const [triggering, setTriggering] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [expandedFinding, setExpandedFinding] = useState<number | null>(null);

  const load = () => {
    if (!id) return;
    return Promise.all([
      getReview(Number(id)).then(r => setReview(r.data)),
      getFindings(Number(id)).then(r => setFindings(r.data)).catch(() => { }),
      getRetrievalVisualization(Number(id)).then(r => setRetrievalVisualization(r.data)).catch(() => { }),
    ]).finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, [id]);

  // Auto-refresh while running
  useEffect(() => {
    if (review?.status !== 'running') return;
    const timer = setInterval(load, 8000);
    return () => clearInterval(timer);
  }, [review?.status]);

  const handleTrigger = async () => {
    if (!id || triggering) return;
    setTriggering(true);
    try {
      await triggerReview(Number(id));
    } catch (err) {
      console.error("Failed to trigger review:", err);
    } finally {
      await load();
      setTriggering(false);
    }
  };

  const handleCancel = async () => {
    if (!id || cancelling) return;
    setCancelling(true);
    try {
      await cancelReview(Number(id));
    } catch (err) {
      console.error("Failed to cancel review:", err);
    } finally {
      await load();
      setCancelling(false);
    }
  };

  if (loading || !review) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-flame border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-slide-up">
      <button onClick={() => navigate(`/designs/${designId}`)} className="flex items-center gap-1.5 text-sm text-text-muted hover:text-text-primary transition-colors">
        <ArrowLeft size={16} /> Back to Design
      </button>

      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">
            Review #{review.id}
          </h1>
          <p className="text-sm text-text-muted mt-1">
            {review.design_name || `Design #${review.design_id}`}
            {review.category && ` · ${String((review.category as Record<string, unknown>).name || '')}`}
          </p>
        </div>
        <div className="flex gap-2">
          {review.status === 'running' && (
            <button
              onClick={handleCancel}
              disabled={cancelling}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-surface-base border border-crimson/50 text-crimson text-sm font-semibold hover:bg-crimson/10 transition-all disabled:opacity-40"
            >
              <XCircle size={16} /> {cancelling ? 'Cancelling...' : 'Cancel Review'}
            </button>
          )}
          {['pending', 'cancelled', 'failed'].includes(review.status) && (
            <button
              onClick={handleTrigger}
              disabled={triggering}
              className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-crimson to-flame text-white text-sm font-semibold hover:shadow-lg hover:shadow-crimson/30 transition-all disabled:opacity-40"
            >
              <Play size={16} /> {triggering ? 'Starting...' : (review.status === 'pending' ? 'Trigger Review' : 'Re-trigger Review')}
            </button>
          )}
        </div>
      </div>

      {/* Pipeline */}
      <ReviewPipeline reviewStatus={review.status} />

      {/* Info */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { label: 'Status', value: <StatusBadge status={review.status} /> },
          { label: 'Findings', value: findings.length },
          { label: 'Started', value: review.started_at ? new Date(review.started_at).toLocaleString() : '—' },
          { label: 'Completed', value: review.completed_at ? new Date(review.completed_at).toLocaleString() : '—' },
        ].map(item => (
          <Card key={item.label}>
            <p className="text-xs text-text-muted mb-1">{item.label}</p>
            <div className="text-sm font-medium text-text-primary">{item.value}</div>
          </Card>
        ))}
      </div>

      {/* Overview */}
      {review.overview && (
        <Card>
          <h3 className="text-sm font-semibold text-text-primary mb-2">Overview</h3>
          <p className="text-sm text-text-secondary leading-relaxed">{review.overview}</p>
        </Card>
      )}

      {/* Error */}
      {review.error_message && (
        <Card className="border-crimson/30">
          <div className="flex items-start gap-2">
            <XCircle size={16} className="text-crimson mt-0.5 shrink-0" />
            <div>
              <p className="text-sm font-medium text-crimson">Error</p>
              <p className="text-sm text-text-secondary mt-1">{review.error_message}</p>
            </div>
          </div>
        </Card>
      )}

      <div className="space-y-4">
        <div className="flex items-center justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold text-text-primary">Retrieval Structures</h2>
            <p className="text-sm text-text-muted mt-1">
              Inspect the RAPTOR summary tree and the GraphRAG entity network used during analysis.
            </p>
          </div>
          {retrievalVisualization?.generated_at && (
            <p className="text-xs text-text-muted">
              Generated {new Date(retrievalVisualization.generated_at).toLocaleString()}
            </p>
          )}
        </div>

        {!retrievalVisualization || retrievalVisualization.status === 'pending' ? (
          <Card>
            <p className="text-sm text-text-muted text-center py-4">
              {review.status === 'running'
                ? 'Retrieval indexes are still being prepared. This section will populate once RAPTOR and GraphRAG finish building.'
                : 'No retrieval visualization snapshot is available for this review yet.'}
            </p>
          </Card>
        ) : (
          <div className="space-y-4">
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <Card className="flex items-center gap-3">
                <Network size={18} className="text-flame shrink-0" />
                <div>
                  <p className="text-sm font-semibold text-text-primary">RAPTOR status: {retrievalVisualization.raptor?.status || 'unknown'}</p>
                  <p className="text-xs text-text-muted">
                    {retrievalVisualization.raptor?.total_nodes || 0} node(s) ready for visualization
                  </p>
                </div>
              </Card>
              <Card className="flex items-center gap-3">
                <Waypoints size={18} className="text-flame shrink-0" />
                <div>
                  <p className="text-sm font-semibold text-text-primary">GraphRAG status: {retrievalVisualization.graph?.status || 'unknown'}</p>
                  <p className="text-xs text-text-muted">
                    {retrievalVisualization.graph?.total_entities || 0} entity(ies), {retrievalVisualization.graph?.total_relations || 0} relation(s)
                  </p>
                </div>
              </Card>
            </div>

            {retrievalVisualization.raptor?.status === 'ready' ? (
              <RaptorTreeView snapshot={retrievalVisualization.raptor} />
            ) : (
              <Card>
                <p className="text-sm text-text-muted text-center py-4">
                  RAPTOR tree was not available for this review.
                </p>
              </Card>
            )}

            {retrievalVisualization.graph?.status === 'ready' ? (
              <GraphRagView snapshot={retrievalVisualization.graph} />
            ) : (
              <Card>
                <p className="text-sm text-text-muted text-center py-4">
                  GraphRAG network was not available for this review.
                </p>
              </Card>
            )}
          </div>
        )}
      </div>

      {/* Findings */}
      <div>
        <h2 className="text-lg font-semibold text-text-primary mb-3">
          Findings ({findings.length})
        </h2>
        {findings.length === 0 ? (
          <Card>
            <p className="text-sm text-text-muted text-center py-4">
              {review.status === 'running' ? 'Analysis in progress...' : 'No findings.'}
            </p>
          </Card>
        ) : (
          <div className="space-y-2">
            {findings.map(f => (
              <Card key={f.id}>
                <button
                  onClick={() => setExpandedFinding(expandedFinding === f.id ? null : f.id)}
                  className="w-full text-left"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {metStatusIcons[f.met_status || ''] || metStatusIcons.not_applicable}
                      <div>
                        <p className="text-sm font-medium text-text-primary">{f.title}</p>
                        <p className="text-xs text-text-muted">
                          {f.parent_parameter_title || f.finding_type}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {f.severity && (
                        <span className={`text-xs font-medium px-2 py-0.5 rounded-full border ${severityColors[f.severity] || severityColors.info}`}>
                          {f.severity}
                        </span>
                      )}
                      {f.met_status && (
                        <span className="text-xs text-text-muted">{f.met_status.replace(/_/g, ' ')}</span>
                      )}
                    </div>
                  </div>
                </button>

                {expandedFinding === f.id && <FindingDetails finding={f} />}
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
