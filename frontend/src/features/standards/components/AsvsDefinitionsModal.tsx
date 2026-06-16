import type { ASVSLevelDefinition, IngestionJob } from '../../../api/standards';
import LoadingSpinner from '../../../components/ui/LoadingSpinner';
import Modal from '../../../components/ui/Modal';

interface AsvsDefinitionsModalProps {
  definitionsJob: IngestionJob | null;
  definitions: ASVSLevelDefinition[];
  loading: boolean;
  onClose: () => void;
}

export default function AsvsDefinitionsModal({
  definitionsJob,
  definitions,
  loading,
  onClose,
}: AsvsDefinitionsModalProps) {
  return (
    <Modal
      open={Boolean(definitionsJob)}
      onClose={onClose}
      title={`ASVS Level Definitions${definitionsJob ? ` · Version ${definitionsJob.version_no}` : ''}`}
    >
      {loading ? (
        <LoadingSpinner className="flex items-center justify-center py-8" sizeClassName="w-7 h-7" />
      ) : definitions.length === 0 ? (
        <div className="space-y-3">
          <p className="text-sm text-text-secondary">No version-specific ASVS level definitions were extracted for this ingestion job.</p>
          {definitionsJob && (
            ((definitionsJob.summary_json as { asvs_level_definitions?: { reason?: string } }).asvs_level_definitions?.reason)
          ) && (
            <p className="text-xs text-text-muted">
              {String((definitionsJob.summary_json as { asvs_level_definitions?: { reason?: string } }).asvs_level_definitions?.reason)}
            </p>
          )}
        </div>
      ) : (
        <div className="space-y-3 max-h-[60vh] overflow-y-auto pr-1">
          {definitions.map((definition) => (
            <div key={definition.id} className="rounded-lg border border-surface-border bg-surface/60 p-3">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-sm font-semibold text-text-primary">
                  {definition.code} · {definition.name}
                </h3>
                {definition.context_marker && <span className="text-[11px] text-text-muted">{definition.context_marker}</span>}
              </div>
              <p className="mt-2 text-xs text-text-secondary">{definition.classification_guidance}</p>
              {definition.source_quote && (
                <p className="mt-2 border-l-2 border-surface-border pl-3 text-xs text-text-muted">{definition.source_quote}</p>
              )}
            </div>
          ))}
        </div>
      )}
    </Modal>
  );
}
