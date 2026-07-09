import { useState, type ReactNode } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';
import type { CitationAnchor, DebateStreamState, Finding, FindingEvidenceSource, JsonRecord } from '../../../api/reviews';
import {
  formatLabel,
  formatValue,
  isPresent,
  isRecord,
  metStatusIcons,
  stringList,
} from '../utils/reviewPresentation';
import type { DetailItem } from '../utils/reviewPresentation';
import AgentMessageBubble from './AgentMessageBubble';

// Mirrors backend build_debate_id (debate_events.py) so a persisted Finding
// can be matched against its live/replayed debate transcript client-side.
function computeDebateId(finding: Finding): string {
  if (finding.finding_type === 'diagram') {
    const diagramId = (finding.diagram_id || '').trim();
    return `diagram:${diagramId || 'unknown'}`;
  }
  if (finding.child_parameter_id != null) {
    return `text:${finding.child_parameter_id}`;
  }
  const reference = (finding.requirement_reference || 'unknown').trim();
  return `text:${reference || 'unknown'}`;
}

export function TextBlock({ title, children }: { title: string; children: ReactNode }) {
  if (!isPresent(children)) {
    return null;
  }

  return (
    <section>
      <p className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1">{title}</p>
      <div className="text-sm text-text-secondary leading-relaxed">{children}</div>
    </section>
  );
}

export function FieldGrid({ items }: { items: DetailItem[] }) {
  const visibleItems = items.filter((item) => isPresent(item.value));
  if (visibleItems.length === 0) {
    return null;
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
      {visibleItems.map((item) => (
        <div key={item.label} className="bg-midnight/30 p-3 rounded-lg border border-surface-border">
          <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider">{item.label}</p>
          <div className="mt-1 text-sm text-text-primary break-words">{item.value}</div>
        </div>
      ))}
    </div>
  );
}

export function ChipList({ items }: { items: string[] }) {
  if (items.length === 0) {
    return null;
  }

  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((item) => (
        <span key={item} className="rounded-full border border-surface-border bg-midnight px-2 py-1 text-xs text-text-secondary">
          {formatLabel(item)}
        </span>
      ))}
    </div>
  );
}

export function JsonPanel({ title, value }: { title: string; value: unknown }) {
  if (!isPresent(value)) {
    return null;
  }

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

export function SeverityAnalysisSection({ analysis }: { analysis: Record<string, unknown> | null }) {
  if (!analysis || Object.keys(analysis).length === 0) {
    return null;
  }

  const isV2 = analysis.version === 'v2' || analysis.source === 'deterministic_risk_rubric';
  if (isV2) {
    const dimensions = isRecord(analysis.dimensions) ? analysis.dimensions : {};
    const domainBase = isRecord(dimensions.domain_base) ? dimensions.domain_base : {};
    const impact = isRecord(dimensions.impact_capabilities) ? dimensions.impact_capabilities : {};
    const exposure = isRecord(dimensions.exposure_capabilities) ? dimensions.exposure_capabilities : {};
    const keywords = isRecord(dimensions.keyword_signals) ? dimensions.keyword_signals : {};
    const confidence = isRecord(dimensions.confidence_adjustment) ? dimensions.confidence_adjustment : {};
    const caps = Array.isArray(analysis.caps) ? analysis.caps.filter(isRecord) : [];
    const matchedImpact = stringList(impact.matched);
    const matchedExposure = stringList(exposure.matched);
    const matchedKeywords = stringList(keywords.matched);
    const appliedCaps = caps.filter((cap) => cap.applied);

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
            {
              label: 'Confidence',
              value: isPresent(confidence.confidence_score)
                ? `${(Number(confidence.confidence_score) * 100).toFixed(0)}%`
                : null,
            },
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
              {caps.map((cap) => (
                <span
                  key={`${String(cap.kind)}-${String(cap.cap)}`}
                  className={`rounded-full border px-2 py-1 text-xs ${
                    cap.applied
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

interface ParsedAssessedRequirement {
  ordinal: string | null;
  stableKey: string | null;
  primaryText: string;
  secondaryText: string;
  verdict: string;
}

// A diagram finding is evaluated against a whole checklist of requirements in
// one pass. Depending on which code path produced it, an item's
// requirement_id comes back either as a bare stable key (with the actual
// observation in "summary"/"reasoning"), or as a compound "N. [stable_key]
// full requirement text" line the vision model echoed back verbatim. Handle
// both so each requirement renders as its own clean row instead of one giant
// concatenated paragraph.
function parseAssessedRequirements(metadata: JsonRecord | null): ParsedAssessedRequirement[] {
  const raw = metadata?.assessed_requirements;
  if (!Array.isArray(raw)) {
    return [];
  }
  return raw
    .filter(isRecord)
    .map((item) => {
      const rawId = String(item.requirement_id || '').trim();
      const match = rawId.match(/^(\d+)\.\s*\[([^\]]+)\]\s*(.*)$/s);
      const ordinal = match ? match[1] : null;
      const stableKey = match ? match[2] : rawId || null;
      const embeddedText = match ? match[3].trim() : '';
      const summary = String(item.summary || item.reasoning || '').trim();
      const primaryText = embeddedText || summary;
      const secondaryText = embeddedText && summary && summary !== embeddedText ? summary : '';
      const verdict = String(item.verdict || '').trim().toLowerCase();
      return { ordinal, stableKey, primaryText, secondaryText, verdict };
    })
    .filter((item) => item.primaryText);
}

export function DiagramAssessedRequirementsList({ metadata }: { metadata: JsonRecord | null }) {
  const items = parseAssessedRequirements(metadata);
  if (items.length === 0) {
    return null;
  }

  return (
    <section>
      <p className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">
        Assessed Diagram Requirements ({items.length})
      </p>
      <div className="space-y-2">
        {items.map((item, index) => (
          <div
            key={item.stableKey || index}
            className="rounded-lg border border-surface-border bg-midnight/30 p-3"
          >
            <div className="flex items-start gap-2">
              <span className="mt-0.5 shrink-0" title={item.verdict || 'unknown'}>
                {metStatusIcons[item.verdict] || metStatusIcons.na}
              </span>
              <div className="min-w-0 flex-1">
                <p className="text-sm text-text-primary">
                  {item.ordinal && <span className="text-text-muted mr-1">{item.ordinal}.</span>}
                  {item.primaryText}
                </p>
                {item.secondaryText && (
                  <p className="mt-1 text-xs text-text-secondary leading-relaxed">{item.secondaryText}</p>
                )}
              </div>
              <span className="shrink-0 rounded-full border border-surface-border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-text-muted">
                {formatLabel(item.verdict || 'unknown')}
              </span>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

interface FindingDetailsProps {
  finding: Finding;
  debatesById?: Record<string, DebateStreamState>;
  activeCitationId?: number | null;
  onCitationSelect?: (citation: CitationAnchor) => void;
}

export function FindingDetails({ finding, debatesById, activeCitationId = null, onCitationSelect }: FindingDetailsProps) {
  const [activeAgentAudit, setActiveAgentAudit] = useState<string | null>('Hunter');
  const debateTranscript = debatesById?.[computeDebateId(finding)]?.transcript ?? null;
  const hasDebateTranscript = Boolean(debateTranscript && debateTranscript.length > 0);
  const evidenceSources = (finding.evidence_sources || []).filter(
    (source): source is FindingEvidenceSource => Boolean(source?.key && source?.label),
  );
  // Match ReviewFindingsPanel's "Jump to citation" filter so both agree on
  // which citation is first — otherwise the two can point to different
  // citations when an invalid (page_number < 1) citation leads the raw list.
  const visibleCitations = (finding.citations || []).filter((citation) => citation.page_number >= 1);

  const auditItems = [
    { label: 'Finding ID', value: finding.id },
    { label: 'Review ID', value: finding.review_id },
    { label: 'Parent Parameter ID', value: finding.parent_parameter_id },
    { label: 'Child Parameter ID', value: finding.child_parameter_id },
  ];

  return (
    <div className="mt-4 space-y-4 border-t border-surface-border pt-4 md:pl-9">
      <TextBlock title="Description">{finding.description}</TextBlock>

      <TextBlock title="Reason">{finding.reason}</TextBlock>
      <TextBlock title="Recommendation">{finding.recommendation}</TextBlock>

      {evidenceSources.length > 0 && (
        <section className="space-y-3 rounded-xl border border-surface-border bg-surface-base/50 p-4">
          <p className="text-xs font-semibold text-text-muted uppercase tracking-wider">Evidence Sources</p>
          <div className="flex flex-wrap gap-2">
            {evidenceSources.map((source) => (
              <span
                key={source.key}
                className="inline-flex items-center gap-2 rounded-full border border-surface-border bg-midnight px-3 py-1 text-xs text-text-secondary"
              >
                <span className="font-semibold text-text-primary">{source.label}</span>
                {source.count > 0 && <span className="opacity-70">{source.count}</span>}
              </span>
            ))}
          </div>
        </section>
      )}

      <section className="space-y-3 rounded-xl border border-surface-border bg-surface-base/50 p-4">
        <FieldGrid
          items={[
            { label: 'Category', value: [finding.category_code, finding.category_name].filter(Boolean).join(' - ') },
            { label: 'Parent Parameter', value: finding.parent_parameter_title },
            { label: 'Child Stable Key', value: finding.child_parameter_stable_key },
            { label: 'Child Ordinal', value: finding.child_parameter_ordinal },
          ]}
        />
        {finding.finding_type === 'diagram' ? (
          <DiagramAssessedRequirementsList metadata={finding.requirement_metadata} />
        ) : (
          <TextBlock title="Requirement Text">
            {finding.requirement_text && (
              <span className="italic">
                {finding.requirement_reference && (
                  <span className="font-semibold text-text-primary mr-2">[{finding.requirement_reference}]</span>
                )}
                {finding.requirement_text}
              </span>
            )}
          </TextBlock>
        )}
      </section>

      <SeverityAnalysisSection analysis={finding.severity_analysis} />

      {(finding.diagram_id || finding.diagram_caption || finding.diagram_image_url || finding.vision_reasoning || finding.vision_thought_process) && (
        <section className="space-y-3 rounded-xl border border-surface-border bg-midnight/30 p-4">
          <p className="text-xs font-semibold text-text-muted uppercase tracking-wider">Diagram And Vision Analysis</p>
          {finding.diagram_image_url && (
            <div className="overflow-hidden rounded-xl border border-surface-border bg-surface-base">
              <div className="flex items-center justify-between gap-3 border-b border-surface-border px-3 py-2">
                <div>
                  <p className="text-sm font-semibold text-text-primary">Related Diagram</p>
                  {finding.diagram_caption && <p className="text-xs text-text-muted">{finding.diagram_caption}</p>}
                </div>
                <a
                  href={finding.diagram_image_url}
                  target="_blank"
                  rel="noreferrer"
                  className="shrink-0 rounded-lg border border-surface-border px-2 py-1 text-xs font-semibold text-text-secondary hover:text-text-primary"
                >
                  Open image
                </a>
              </div>
              <div className="flex max-h-[520px] items-center justify-center overflow-auto bg-midnight/40 p-3">
                <img
                  src={finding.diagram_image_url}
                  alt={finding.diagram_caption || finding.diagram_id || 'Related diagram'}
                  className="max-h-[480px] max-w-full rounded-lg object-contain"
                  loading="lazy"
                />
              </div>
            </div>
          )}
          <FieldGrid
            items={[
              { label: 'Diagram ID', value: finding.diagram_id },
              { label: 'Diagram Caption', value: finding.diagram_caption },
            ]}
          />
          <TextBlock title="Vision Reasoning">{finding.vision_reasoning}</TextBlock>
          <TextBlock title="Vision Thought Process">{finding.vision_thought_process}</TextBlock>
        </section>
      )}

      {hasDebateTranscript ? (
        <section className="space-y-3 rounded-xl border border-surface-border bg-surface-base/50 p-4">
          <p className="text-xs font-semibold text-text-muted uppercase tracking-wider">Agent Audit</p>
          <div className="flex flex-col gap-3">
            {debateTranscript!.map((message) => (
              <AgentMessageBubble key={message.message_id} message={message} />
            ))}
          </div>
        </section>
      ) : (
        (finding.hunter_reasoning || finding.critic_reasoning || finding.mediator_reasoning || finding.hunter_thought_process || finding.critic_thought_process || finding.mediator_thought_process) && (
          <section className="space-y-3 rounded-xl border border-surface-border bg-surface-base/50 p-4">
            <p className="text-xs font-semibold text-text-muted uppercase tracking-wider">Agent Audit</p>
            <div className="flex flex-col gap-3">
              {[
                { role: 'Hunter', color: 'text-flame', reasoning: finding.hunter_reasoning, thought: finding.hunter_thought_process },
                { role: 'Critic', color: 'text-burgundy-light', reasoning: finding.critic_reasoning, thought: finding.critic_thought_process },
                { role: 'Mediator', color: 'text-emerald-400', reasoning: finding.mediator_reasoning, thought: finding.mediator_thought_process },
              ].map((agent) => {
                if (!agent.reasoning && !agent.thought) return null;
                const isActive = activeAgentAudit === agent.role;
                return (
                  <div key={agent.role} className="rounded-lg border border-surface-border bg-midnight">
                    <button
                      type="button"
                      onClick={() => setActiveAgentAudit(isActive ? null : agent.role)}
                      className="flex w-full items-center justify-between p-3 text-left transition-colors hover:bg-surface-border/30"
                    >
                      <p className={`text-[10px] font-bold uppercase tracking-wider ${agent.color}`}>{agent.role}</p>
                      {isActive ? (
                        <ChevronDown size={14} className="text-text-muted shrink-0" />
                      ) : (
                        <ChevronRight size={14} className="text-text-muted shrink-0" />
                      )}
                    </button>
                    {isActive && (
                      <div className="space-y-4 border-t border-surface-border p-3 animate-fade-in">
                        <TextBlock title="Reasoning">{agent.reasoning}</TextBlock>
                        <TextBlock title="Thought Process">{agent.thought}</TextBlock>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </section>
        )
      )}

      {visibleCitations.length > 0 && (
        <section className="space-y-2 rounded-xl border border-surface-border bg-surface-base/50 p-4">
          <div className="flex items-center justify-between gap-3">
            <p className="text-xs font-semibold text-text-muted uppercase tracking-wider">Evidence Citations</p>
          </div>
          <div className="space-y-2">
            {visibleCitations.map((citation) => {
              const isActive = activeCitationId === citation.id;
              return (
                <button
                  key={citation.id}
                  type="button"
                  onClick={() => onCitationSelect?.(citation)}
                  className={`w-full rounded p-3 text-left transition-colors ${
                    isActive
                      ? 'border border-flame bg-flame/10 shadow-[0_0_0_1px_rgba(240,89,65,0.25)]'
                      : 'border border-surface-border bg-surface-base hover:border-flame/40 hover:bg-flame/5'
                  }`}
                >
                  <p className="text-[10px] font-mono text-text-muted mb-1 flex flex-wrap items-center gap-2">
                    <span className="bg-midnight px-1.5 py-0.5 rounded">Page {citation.page_number}</span>
                    <span className="bg-midnight px-1.5 py-0.5 rounded">{citation.anchor_type}</span>
                    {citation.retrieval_origin_label && (
                      <span className="bg-midnight px-1.5 py-0.5 rounded">{citation.retrieval_origin_label}</span>
                    )}
                    <span className="opacity-70 break-all">{citation.block_id}</span>
                  </p>
                  {citation.quoted_text && (
                    <p className="text-sm text-text-secondary italic border-l-2 border-flame/50 pl-3 py-0.5">
                      "{citation.quoted_text}"
                    </p>
                  )}
                  {/* <FieldGrid
                    items={[
                      { label: 'X0', value: typeof citation.bbox_x0 === 'number' ? citation.bbox_x0.toFixed(2) : citation.bbox_x0 },
                      { label: 'Y0', value: typeof citation.bbox_y0 === 'number' ? citation.bbox_y0.toFixed(2) : citation.bbox_y0 },
                      { label: 'X1', value: typeof citation.bbox_x1 === 'number' ? citation.bbox_x1.toFixed(2) : citation.bbox_x1 },
                      { label: 'Y1', value: typeof citation.bbox_y1 === 'number' ? citation.bbox_y1.toFixed(2) : citation.bbox_y1 },
                      { label: 'Created At', value: new Date(citation.created_at).toLocaleString() },
                    ]}
                  /> */}
                </button>
              );
            })}
          </div>
        </section>
      )}

      {Boolean(
        (finding.requirement_metadata?.analysis_trace as JsonRecord | undefined)
          ?.hunter_diagnostics as JsonRecord | undefined,
      ) && (() => {
        const diagnostics = (finding.requirement_metadata?.analysis_trace as JsonRecord)
          .hunter_diagnostics as JsonRecord;
        if (!diagnostics.zero_citation_not_met) return null;
        const availableCount = Array.isArray(diagnostics.available_block_ids)
          ? diagnostics.available_block_ids.length
          : 0;
        return (
          <section className="space-y-2 rounded-xl border border-amber-500/30 bg-amber-500/10 p-4">
            <p className="text-xs font-semibold text-amber-300 uppercase tracking-wider">Retrieval Note</p>
            <p className="text-sm text-amber-200/90">
              Hunter searched {availableCount} retrieved section{availableCount === 1 ? '' : 's'} and found no
              relevant evidence to cite, so this finding was recorded as "not met" without citations. If the TSD
              does describe this control elsewhere, this may indicate a retrieval coverage gap rather than a true
              absence.
            </p>
          </section>
        );
      })()}

      <section className="space-y-3 rounded-xl border border-surface-border bg-midnight/30 p-4">
        <p className="text-xs font-semibold text-text-muted uppercase tracking-wider">Raw Audit</p>
        <FieldGrid items={auditItems} />
        <JsonPanel title="Raw Finding Payload" value={finding} />
      </section>
    </div>
  );
}
