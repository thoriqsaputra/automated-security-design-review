import { Network, Waypoints } from 'lucide-react';
import type { RetrievalVisualization, Review } from '../../../api/reviews';
import Card from '../../../components/ui/Card';
import GraphRagView from '../../../components/flow/GraphRagView';
import RaptorTreeView from '../../../components/flow/RaptorTreeView';

interface ReviewRetrievalPanelProps {
  review: Review;
  retrievalVisualization: RetrievalVisualization | null;
}

export default function ReviewRetrievalPanel({
  review,
  retrievalVisualization,
}: ReviewRetrievalPanelProps) {
  return (
    <div className="space-y-4 animate-fade-in">
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
              <p className="text-sm text-text-muted text-center py-4">RAPTOR tree was not available for this review.</p>
            </Card>
          )}

          {retrievalVisualization.graph?.status === 'ready' ? (
            <GraphRagView snapshot={retrievalVisualization.graph} />
          ) : (
            <Card>
              <p className="text-sm text-text-muted text-center py-4">GraphRAG network was not available for this review.</p>
            </Card>
          )}
        </div>
      )}
    </div>
  );
}
