import React, { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Play, AlertTriangle, CheckCircle2, XCircle, Info } from 'lucide-react';
import Card from '../components/ui/Card';
import StatusBadge from '../components/ui/StatusBadge';
import ReviewPipeline from '../components/flow/ReviewPipeline';
import { getReview, getFindings, triggerReview, cancelReview, type Review, type Finding } from '../api/reviews';

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
  not_applicable: <Info size={16} className="text-text-muted" />,
};

export default function ReviewDetail() {
  const { designId, id } = useParams<{ designId: string; id: string }>();
  const navigate = useNavigate();
  const [review, setReview] = useState<Review | null>(null);
  const [findings, setFindings] = useState<Finding[]>([]);
  const [loading, setLoading] = useState(true);
  const [triggering, setTriggering] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [expandedFinding, setExpandedFinding] = useState<number | null>(null);

  const load = () => {
    if (!id) return;
    return Promise.all([
      getReview(Number(id)).then(r => setReview(r.data)),
      getFindings(Number(id)).then(r => setFindings(r.data)).catch(() => { }),
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

                {expandedFinding === f.id && (
                  <div className="mt-4 pl-9 space-y-3 border-t border-surface-border pt-4">
                    <div>
                      <p className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1">Description</p>
                      <p className="text-sm text-text-secondary">{f.description}</p>
                    </div>
                    {f.reason && (
                      <div>
                        <p className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1">Reason</p>
                        <p className="text-sm text-text-secondary">{f.reason}</p>
                      </div>
                    )}
                    {f.recommendation && (
                      <div>
                        <p className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1">Recommendation</p>
                        <p className="text-sm text-text-secondary">{f.recommendation}</p>
                      </div>
                    )}
                    {f.requirement_text && (
                      <div>
                        <p className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1">Requirement</p>
                        <p className="text-sm text-text-secondary italic">{f.requirement_text}</p>
                      </div>
                    )}
                    {/* Agent reasoning */}
                    {(f.hunter_reasoning || f.critic_reasoning || f.mediator_reasoning) && (
                      <div className="space-y-2">
                        <p className="text-xs font-semibold text-text-muted uppercase tracking-wider">Agent Debate</p>
                        {f.hunter_reasoning && (
                          <div className="bg-midnight rounded-lg p-3">
                            <p className="text-[10px] font-bold text-flame uppercase tracking-wider mb-1">Hunter</p>
                            <p className="text-xs text-text-secondary">{f.hunter_reasoning}</p>
                          </div>
                        )}
                        {f.critic_reasoning && (
                          <div className="bg-midnight rounded-lg p-3">
                            <p className="text-[10px] font-bold text-burgundy-light uppercase tracking-wider mb-1">Critic</p>
                            <p className="text-xs text-text-secondary">{f.critic_reasoning}</p>
                          </div>
                        )}
                        {f.mediator_reasoning && (
                          <div className="bg-midnight rounded-lg p-3">
                            <p className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider mb-1">Mediator</p>
                            <p className="text-xs text-text-secondary">{f.mediator_reasoning}</p>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
