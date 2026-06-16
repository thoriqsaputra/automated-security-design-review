import { useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import { Globe, Upload } from 'lucide-react';
import Card from '../components/ui/Card';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import Modal from '../components/ui/Modal';
import { listCategories, createIngestionJob, type StandardCategory } from '../api/standards';

const categoryIcons: Record<string, ReactNode> = {
  web_application: <Globe size={28} className="text-white" />,
};

const categoryGradients: Record<string, string> = {
  web_application: 'from-crimson to-flame',
};

export default function StandardsHub() {
  const navigate = useNavigate();
  const [categories, setCategories] = useState<StandardCategory[]>([]);
  const [loading, setLoading] = useState(true);
  const [showUpload, setShowUpload] = useState(false);
  const [uploadCat, setUploadCat] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [startPage, setStartPage] = useState('');
  const [endPage, setEndPage] = useState('');
  const [levelDefinitionStartPage, setLevelDefinitionStartPage] = useState('');
  const [levelDefinitionEndPage, setLevelDefinitionEndPage] = useState('');
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    listCategories().then(r => setCategories(r.data)).finally(() => setLoading(false));
  }, []);

  const handleUpload = async () => {
    if (!uploadCat || !file) return;
    setUploading(true);
    try {
      await createIngestionJob(uploadCat, file, startPage, endPage, levelDefinitionStartPage, levelDefinitionEndPage);
      setShowUpload(false);
      setFile(null);
      setUploadCat('');
      setStartPage('');
      setEndPage('');
      setLevelDefinitionStartPage('');
      setLevelDefinitionEndPage('');
      // Refresh categories
      listCategories().then(r => setCategories(r.data));
    } finally {
      setUploading(false);
    }
  };

  const openUploadFor = (code: string) => {
    setUploadCat(code);
    setShowUpload(true);
  };

  if (loading) {
    return <LoadingSpinner />;
  }

  return (
    <div className="space-y-6 animate-slide-up">
      <div>
        <h1 className="text-2xl font-bold text-text-primary">Standards</h1>
        <p className="text-sm text-text-muted mt-1">Security standard categories and ingested parameters</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {categories.map((cat) => (
          <div key={cat.id} className="relative group">
            <Card className="relative overflow-hidden" hover onClick={() => navigate(`/standards/${cat.code}`)}>
              {/* Gradient accent bar */}
              <div className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${categoryGradients[cat.code] || 'from-crimson to-flame'}`} />

              <div className="flex items-center gap-4 mb-4 mt-1">
                <div className={`p-3 rounded-xl bg-gradient-to-br ${categoryGradients[cat.code] || 'from-crimson to-flame'}`}>
                  {categoryIcons[cat.code] || <Globe size={28} className="text-white" />}
                </div>
                <div>
                  <h3 className="text-base font-semibold text-text-primary">{cat.name}</h3>
                  <p className="text-xs text-text-muted">{cat.code}</p>
                </div>
              </div>

              {cat.description && (
                <p className="text-xs text-text-muted mb-4 line-clamp-2">{cat.description}</p>
              )}

              <div className="grid grid-cols-2 gap-3">
                <div className="bg-midnight rounded-lg p-3">
                  <p className="text-lg font-bold text-text-primary">{cat.active_parameters_count}</p>
                  <p className="text-[10px] text-text-muted uppercase tracking-wider">Parameters</p>
                </div>
                <div className="bg-midnight rounded-lg p-3">
                  <p className="text-lg font-bold text-text-primary">{cat.active_job_version ?? '—'}</p>
                  <p className="text-[10px] text-text-muted uppercase tracking-wider">Version</p>
                </div>
              </div>
            </Card>

            {/* Floating upload button */}
            <button
              onClick={(e) => { e.stopPropagation(); openUploadFor(cat.code); }}
              className="absolute top-4 right-4 p-2 rounded-lg bg-surface-hover/80 border border-surface-border text-text-muted hover:text-flame hover:border-flame/40 opacity-0 group-hover:opacity-100 transition-all"
              title="Ingest standard"
            >
              <Upload size={14} />
            </button>
          </div>
        ))}
      </div>

      {/* Upload Modal */}
      <Modal open={showUpload} onClose={() => setShowUpload(false)} title="Ingest Security Standard">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1.5">Category</label>
            <select
              value={uploadCat}
              onChange={e => setUploadCat(e.target.value)}
              className="w-full px-3 py-2.5 rounded-lg bg-surface border border-surface-border text-text-primary text-sm focus:outline-none focus:border-crimson transition-colors"
            >
              <option value="">Select a category...</option>
              {categories.map(c => (
                <option key={c.id} value={c.code}>{c.name}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1.5">Standard Document (PDF)</label>
            <input
              type="file"
              accept=".pdf"
              onChange={e => setFile(e.target.files?.[0] || null)}
              className="w-full text-sm text-text-muted file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-midnight-lighter file:text-text-secondary file:font-medium file:cursor-pointer hover:file:bg-surface-hover file:transition-colors"
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-text-secondary mb-1">Parameter Start Page (Optional)</label>
              <input
                type="number"
                min="1"
                placeholder="e.g. 12"
                value={startPage}
                onChange={e => setStartPage(e.target.value)}
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
                onChange={e => setEndPage(e.target.value)}
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
                onChange={e => setLevelDefinitionStartPage(e.target.value)}
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
                onChange={e => setLevelDefinitionEndPage(e.target.value)}
                className="w-full px-3 py-2 rounded-lg bg-surface border border-surface-border text-text-primary text-sm focus:outline-none focus:border-crimson transition-colors"
              />
            </div>
          </div>
          <button
            onClick={handleUpload}
            disabled={!uploadCat || !file || uploading}
            className="w-full py-2.5 rounded-xl bg-gradient-to-r from-crimson to-flame text-white text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-crimson/30 transition-all"
          >
            {uploading ? 'Ingesting...' : 'Start Ingestion'}
          </button>
        </div>
      </Modal>
    </div>
  );
}
