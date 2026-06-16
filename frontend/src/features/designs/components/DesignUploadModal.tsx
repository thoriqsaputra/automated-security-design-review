import Modal from '../../../components/ui/Modal';

interface DesignUploadModalProps {
  open: boolean;
  name: string;
  file: File | null;
  uploading: boolean;
  onClose: () => void;
  onNameChange: (value: string) => void;
  onFileChange: (file: File | null) => void;
  onSubmit: () => void;
}

export default function DesignUploadModal({
  open,
  name,
  file,
  uploading,
  onClose,
  onNameChange,
  onFileChange,
  onSubmit,
}: DesignUploadModalProps) {
  return (
    <Modal open={open} onClose={onClose} title="Upload Design Document">
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1.5">Design Name (Optional)</label>
          <input
            type="text"
            value={name}
            onChange={(event) => onNameChange(event.target.value)}
            placeholder="e.g. Payment Gateway TSD"
            className="w-full px-3 py-2.5 rounded-lg bg-surface border border-surface-border text-text-primary text-sm placeholder:text-text-muted focus:outline-none focus:border-crimson transition-colors"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1.5">Document (PDF)</label>
          <input
            type="file"
            accept=".pdf"
            onChange={(event) => onFileChange(event.target.files?.[0] || null)}
            className="w-full text-sm text-text-muted file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-midnight-lighter file:text-text-secondary file:font-medium file:cursor-pointer hover:file:bg-surface-hover file:transition-colors"
          />
        </div>
        <button
          onClick={onSubmit}
          disabled={!file || uploading}
          className="w-full py-2.5 rounded-xl bg-gradient-to-r from-crimson to-flame text-white text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-crimson/30 transition-all"
        >
          {uploading ? 'Uploading...' : 'Upload'}
        </button>
      </div>
    </Modal>
  );
}
