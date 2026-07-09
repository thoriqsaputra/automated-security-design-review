import Modal from '../../../components/ui/Modal';

interface DesignReingestModalProps {
  open: boolean;
  file: File | null;
  reingesting: boolean;
  error: string | null;
  onClose: () => void;
  onFileChange: (file: File | null) => void;
  onSubmit: () => void;
}

export default function DesignReingestModal({
  open,
  file,
  reingesting,
  error,
  onClose,
  onFileChange,
  onSubmit,
}: DesignReingestModalProps) {
  return (
    <Modal open={open} onClose={onClose} title="Reingest Design Document">
      <div className="space-y-4">
        <p className="text-sm text-text-muted">
          This replaces the source PDF and re-runs preparation. Existing reviews keep their
          original document and citations.
        </p>
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1.5">New Document (PDF)</label>
          <input
            type="file"
            accept=".pdf"
            onChange={(event) => onFileChange(event.target.files?.[0] || null)}
            className="w-full text-sm text-text-muted file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-midnight-lighter file:text-text-secondary file:font-medium file:cursor-pointer hover:file:bg-surface-hover file:transition-colors"
          />
        </div>
        {error && <p className="text-sm text-crimson">{error}</p>}
        <button
          onClick={onSubmit}
          disabled={!file || reingesting}
          className="w-full py-2.5 rounded-xl bg-gradient-to-r from-crimson to-flame text-white text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-crimson/30 transition-all"
        >
          {reingesting ? 'Reingesting...' : 'Reingest'}
        </button>
      </div>
    </Modal>
  );
}
