import { useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import { Globe, Upload } from 'lucide-react';
import Card from '../components/ui/Card';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import CategoryUploadModal from '../features/standards/components/CategoryUploadModal';
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
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    listCategories().then(r => setCategories(r.data)).finally(() => setLoading(false));
  }, []);

  const handleUpload = async () => {
    if (!uploadCat || !file) return;
    setUploading(true);
    try {
      await createIngestionJob(uploadCat, file, startPage, endPage);
      setShowUpload(false);
      setFile(null);
      setUploadCat('');
      setStartPage('');
      setEndPage('');
      // Refresh categories
      listCategories().then(r => setCategories(r.data));
      navigate(`/standards/${uploadCat}`);
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
                  {cat.description && (
                    <p className="text-xs text-text-muted line-clamp-3">{cat.description}</p>
                  )}
                </div>
              </div>

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

      <CategoryUploadModal
        open={showUpload}
        title="Ingest Security Standard"
        uploading={uploading}
        file={file}
        startPage={startPage}
        endPage={endPage}
        onClose={() => setShowUpload(false)}
        onFileChange={setFile}
        onStartPageChange={setStartPage}
        onEndPageChange={setEndPage}
        onSubmit={handleUpload}
      />
    </div>
  );
}
