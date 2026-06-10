import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, FileText, Trash2 } from 'lucide-react';
import Card from '../components/ui/Card';
import StatusBadge from '../components/ui/StatusBadge';
import EmptyState from '../components/ui/EmptyState';
import Modal from '../components/ui/Modal';
import { listDesigns, createDesign, deleteDesign, type Design } from '../api/designs';

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

  const handleCreate = async () => {
    if (!name.trim() || !file) return;
    setUploading(true);
    try {
      await createDesign(name, file);
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
    return (
      <div className="flex items-center justify-center h-64">
        <div className="w-8 h-8 border-2 border-flame border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-slide-up">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-text-primary">Designs</h1>
          <p className="text-sm text-text-muted mt-1">Technical Security Design documents</p>
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
                    <p className="text-xs text-text-muted">{d.original_filename} · {new Date(d.created_at).toLocaleDateString()}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <StatusBadge status={d.status} />
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

      {/* Upload Modal */}
      <Modal open={showModal} onClose={() => setShowModal(false)} title="Upload Design Document">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1.5">Design Name</label>
            <input
              type="text"
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="e.g. Payment Gateway TSD"
              className="w-full px-3 py-2.5 rounded-lg bg-surface border border-surface-border text-text-primary text-sm placeholder:text-text-muted focus:outline-none focus:border-crimson transition-colors"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-text-secondary mb-1.5">Document (PDF)</label>
            <input
              type="file"
              accept=".pdf"
              onChange={e => setFile(e.target.files?.[0] || null)}
              className="w-full text-sm text-text-muted file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-midnight-lighter file:text-text-secondary file:font-medium file:cursor-pointer hover:file:bg-surface-hover file:transition-colors"
            />
          </div>
          <button
            onClick={handleCreate}
            disabled={!name.trim() || !file || uploading}
            className="w-full py-2.5 rounded-xl bg-gradient-to-r from-crimson to-flame text-white text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed hover:shadow-lg hover:shadow-crimson/30 transition-all"
          >
            {uploading ? 'Uploading...' : 'Upload'}
          </button>
        </div>
      </Modal>
    </div>
  );
}
