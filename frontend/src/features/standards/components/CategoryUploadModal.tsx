import Modal from '../../../components/ui/Modal';

interface CategoryUploadModalProps {
  open: boolean;
  title: string;
  uploading: boolean;
  file: File | null;
  startPage: string;
  endPage: string;
  levelDefinitionStartPage: string;
  levelDefinitionEndPage: string;
  onClose: () => void;
  onFileChange: (file: File | null) => void;
  onStartPageChange: (value: string) => void;
  onEndPageChange: (value: string) => void;
  onLevelDefinitionStartPageChange: (value: string) => void;
  onLevelDefinitionEndPageChange: (value: string) => void;
  onSubmit: () => void;
}

export default function CategoryUploadModal(props: CategoryUploadModalProps) {
  const {
    open,
    title,
    uploading,
    file,
    startPage,
    endPage,
    levelDefinitionStartPage,
    levelDefinitionEndPage,
    onClose,
    onFileChange,
    onStartPageChange,
    onEndPageChange,
    onLevelDefinitionStartPageChange,
    onLevelDefinitionEndPageChange,
    onSubmit,
  } = props;

  return (
    <Modal open={open} onClose={onClose} title={title}>
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1.5">Standard Document (PDF)</label>
          <input
            type="file"
            accept=".pdf"
            onChange={(event) => onFileChange(event.target.files?.[0] || null)}
            className="w-full text-sm text-text-muted file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-midnight-lighter file:text-text-secondary file:font-medium file:cursor-pointer hover:file:bg-surface-hover file:transition-colors"
          />
        </div>
        {/* <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium text-text-secondary mb-1">Parameter Start Page (Optional)</label>
            <input
              type="number"
              min="1"
              placeholder="e.g. 12"
              value={startPage}
              onChange={(event) => onStartPageChange(event.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-surface border border-surface-border text-text-primary text-sm focus:outline-none focus:border-crimson transition-colors"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-text-secondary mb-1">Parameter End Page (Optional)</label>
            <input
              type="number"
              min="1"
              placeholder="e.g. 55"
              value={endPage}
              onChange={(event) => onEndPageChange(event.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-surface border border-surface-border text-text-primary text-sm focus:outline-none focus:border-crimson transition-colors"
            />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium text-text-secondary mb-1">Level Definition Start Page (Optional)</label>
            <input
              type="number"
              min="1"
              placeholder="e.g. 8"
              value={levelDefinitionStartPage}
              onChange={(event) => onLevelDefinitionStartPageChange(event.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-surface border border-surface-border text-text-primary text-sm focus:outline-none focus:border-crimson transition-colors"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-text-secondary mb-1">Level Definition End Page (Optional)</label>
            <input
              type="number"
              min="1"
              placeholder="e.g. 10"
              value={levelDefinitionEndPage}
              onChange={(event) => onLevelDefinitionEndPageChange(event.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-surface border border-surface-border text-text-primary text-sm focus:outline-none focus:border-crimson transition-colors"
            />
          </div>
        </div> */}
        <button
          onClick={onSubmit}
          disabled={!file || uploading}
          className="w-full py-2.5 rounded-xl bg-gradient-to-r from-crimson to-flame text-white text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-crimson/30 transition-all"
        >
          {uploading ? 'Ingesting...' : 'Start Ingestion'}
        </button>
      </div>
    </Modal>
  );
}
