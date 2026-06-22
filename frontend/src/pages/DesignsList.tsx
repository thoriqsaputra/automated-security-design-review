import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, FileText, Trash2 } from 'lucide-react';
import Card from '../components/ui/Card';
import EmptyState from '../components/ui/EmptyState';
import LoadingSpinner from '../components/ui/LoadingSpinner';
import { listDesigns, createDesign, deleteDesign, type Design } from '../api/designs';
import DesignUploadModal from '../features/designs/components/DesignUploadModal';

export default function DesignsList() {
  const navigate = useNavigate();
  const [designs, setDesigns] = useState<Design[]>([]);
  const [loading, setLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);
  const [name, setName] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);

  const load = () => {
    return listDesigns().then(r => setDesigns(r.data)).finally(() => setLoading(false));
  };

  useEffect(() => { load(); }, []);

  useEffect(() => {
    if (!designs.some((design) => ['queued', 'running'].includes(design.preparation_status))) {
      return undefined;
    }
    const timer = window.setInterval(() => {
      void load();
    }, 4000);
    return () => window.clearInterval(timer);
  }, [designs]);

  const handleCreate = async () => {
    if (!file) return;
    const finalName = name.trim() || file.name.replace(/\.[^/.]+$/, "");
    setUploading(true);
    try {
      await createDesign(finalName, file);
      setShowModal(false);
      setName('');
      setFile(null);
      load();
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (e: React.MouseEvent, id: number) => {
    e.stopPropagation();
    if (!confirm('Delete this design?')) return;
    await deleteDesign(id);
    load();
  };

  if (loading) {
    return <LoadingSpinner />;
  }

  const getPreparationTone = (status: string) => {
    if (status === 'ready') return 'text-emerald-300 bg-emerald-500/10 border-emerald-500/20';
    if (status === 'failed' || status === 'stale') return 'text-amber-200 bg-amber-500/10 border-amber-500/20';
    return 'text-sky-200 bg-sky-500/10 border-sky-500/20';
  };

  return (
    <div className="space-y-6 animate-slide-up">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Designs</h1>
          <p className="text-sm text-text-muted mt-1">Technical Specification documents</p>
        </div>
        <button
          onClick={() => setShowModal(true)}
          className="flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-crimson to-flame text-white text-sm font-semibold hover:shadow-lg hover:shadow-crimson/30 transition-all"
        >
          <Plus size={16} /> Upload Design
        </button>
      </div>

      {designs.length === 0 ? (
        <EmptyState
          icon={<FileText size={32} className="text-text-muted" />}
          title="No designs yet"
          description="Upload a Technical Security Design document to get started."
          action={
            <button
              onClick={() => setShowModal(true)}
              className="px-4 py-2 rounded-lg bg-crimson text-white text-sm font-medium hover:bg-crimson-light transition-colors"
            >
              Upload Design
            </button>
          }
        />
      ) : (
        <div className="grid gap-3 grid-cols-2">
          {designs.map((d) => (
            <Card key={d.id} hover onClick={() => navigate(`/designs/${d.id}`)}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2.5 rounded-xl bg-midnight-lighter">
                    <FileText size={18} className="text-flame" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-text-primary">{d.name}</p>
                    <p className="text-xs text-text-muted">{new Date(d.created_at).toLocaleDateString()}</p>
                    <p className={`mt-1 inline-flex rounded-full border px-2 py-0.5 text-[11px] font-medium uppercase tracking-wide ${getPreparationTone(d.preparation_status)}`}>
                      {d.preparation_status === 'ready' ? 'Ready for analysis' : `Preparation ${d.preparation_status}`}
                    </p>
                    {['queued', 'running'].includes(d.preparation_status) &&
                      d.preparation_progress &&
                      typeof d.preparation_progress === 'object' &&
                      !Array.isArray(d.preparation_progress) && (
                        <div className="mt-2 w-48">
                          <div className="mb-1 flex items-center justify-between text-[11px] text-text-muted">
                            <span>
                              {typeof d.preparation_progress.current_step === 'string'
                                ? d.preparation_progress.current_step
                                : 'Preparing'}
                            </span>
                            <span>
                              {typeof d.preparation_progress.percentage === 'number'
                                ? d.preparation_progress.percentage
                                : Number(d.preparation_progress.percentage || 0)}
                              %
                            </span>
                          </div>
                          <div className="h-1.5 rounded-full bg-surface-border overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-crimson to-flame transition-all duration-500 ease-out"
                              style={{
                                width: `${Math.max(
                                  4,
                                  typeof d.preparation_progress.percentage === 'number'
                                    ? d.preparation_progress.percentage
                                    : Number(d.preparation_progress.percentage || 0),
                                )}%`,
                              }}
                            />
                          </div>
                        </div>
                      )}
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <button
                    onClick={(e) => handleDelete(e, d.id)}
                    className="p-2 rounded-lg hover:bg-crimson/15 text-text-muted hover:text-crimson transition-colors"
                  >
                    <Trash2 size={15} />
                  </button>
                </div>
              </div>
            </Card>
          ))}
        </div>
      )}

      <DesignUploadModal
        open={showModal}
        name={name}
        file={file}
        uploading={uploading}
        onClose={() => setShowModal(false)}
        onNameChange={setName}
        onFileChange={setFile}
        onSubmit={() => void handleCreate()}
      />
    </div>
  );
}
