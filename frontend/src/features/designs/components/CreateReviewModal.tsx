import type { ReviewAnalysisMode } from '../../../api/reviews';
import type { StandardCategory } from '../../../api/standards';
import Modal from '../../../components/ui/Modal';

interface CreateReviewModalProps {
  open: boolean;
  categories: StandardCategory[];
  selectedCategory: number | '';
  analysisMode: ReviewAnalysisMode;
  creating: boolean;
  onClose: () => void;
  onCategoryChange: (value: number | '') => void;
  onAnalysisModeChange: (value: ReviewAnalysisMode) => void;
  onSubmit: () => void;
}

const analysisModeOptions: Array<{
  value: ReviewAnalysisMode;
  label: string;
  description: string;
}> = [
  {
    value: 'default',
    label: 'Default',
    description: 'Run both text requirement analysis and diagram analysis.',
  },
  {
    value: 'text_only',
    label: 'Text Only',
    description: 'Run requirement analysis only and skip diagram analysis.',
  },
  {
    value: 'diagram_only',
    label: 'Diagram Only',
    description: 'Run diagram analysis only and skip text requirement analysis.',
  },
];

export default function CreateReviewModal({
  open,
  categories,
  selectedCategory,
  analysisMode,
  creating,
  onClose,
  onCategoryChange,
  onAnalysisModeChange,
  onSubmit,
}: CreateReviewModalProps) {
  return (
    <Modal open={open} onClose={onClose} title="Create Security Review">
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1.5">Standard Category</label>
          <select
            value={selectedCategory}
            onChange={(event) => onCategoryChange(event.target.value ? Number(event.target.value) : '')}
            className="w-full px-3 py-2.5 rounded-lg bg-surface border border-surface-border text-text-primary text-sm focus:outline-none focus:border-crimson transition-colors"
          >
            <option value="">Select a category...</option>
            {categories.map((category) => (
              <option key={category.id} value={category.id}>
                {category.name} ({category.code})
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-text-secondary mb-1.5">Analysis Mode</label>
          <select
            value={analysisMode}
            onChange={(event) => onAnalysisModeChange(event.target.value as ReviewAnalysisMode)}
            className="w-full px-3 py-2.5 rounded-lg bg-surface border border-surface-border text-text-primary text-sm focus:outline-none focus:border-crimson transition-colors"
          >
            {analysisModeOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
          <p className="mt-2 text-xs text-text-muted">
            {analysisModeOptions.find((option) => option.value === analysisMode)?.description}
          </p>
        </div>
        <button
          onClick={onSubmit}
          disabled={!selectedCategory || creating}
          className="w-full py-2.5 rounded-xl bg-gradient-to-r from-crimson to-flame text-white text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-crimson/30 transition-all"
        >
          {creating ? 'Creating...' : 'Create Review'}
        </button>
      </div>
    </Modal>
  );
}
