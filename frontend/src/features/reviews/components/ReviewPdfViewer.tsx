import { FileText } from 'lucide-react';
import { type CSSProperties, useEffect, useMemo, useRef, useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import type { PDFDocumentProxy, PDFPageProxy } from 'pdfjs-dist';
import type { CitationAnchor } from '../../../api/reviews';
import Card from '../../../components/ui/Card';
import LoadingSpinner from '../../../components/ui/LoadingSpinner';

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString();

type PageMetric = {
  width: number;
  height: number;
};

interface ReviewPdfViewerProps {
  documentUrl: string | null;
  activeCitation: CitationAnchor | null;
}

export default function ReviewPdfViewer({ documentUrl, activeCitation }: ReviewPdfViewerProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const pageRefs = useRef<Record<number, HTMLDivElement | null>>({});
  const [pageWidth, setPageWidth] = useState(720);
  const [numPages, setNumPages] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [pageMetrics, setPageMetrics] = useState<Record<number, PageMetric>>({});

  useEffect(() => {
    const node = containerRef.current;
    if (!node || typeof ResizeObserver === 'undefined') {
      return;
    }

    const updateWidth = () => {
      const nextWidth = Math.max(320, Math.floor(node.clientWidth - 24));
      setPageWidth(nextWidth);
    };

    updateWidth();
    const observer = new ResizeObserver(updateWidth);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!activeCitation?.page_number) {
      return;
    }
    const pageNode = pageRefs.current[activeCitation.page_number];
    if (pageNode) {
      pageNode.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [activeCitation?.id, activeCitation?.page_number]);

  const handleDocumentLoadSuccess = (document: PDFDocumentProxy) => {
    setNumPages(document.numPages);
    setLoading(false);
    setLoadError(null);
  };

  const handlePageLoadSuccess = (pageNumber: number, page: PDFPageProxy) => {
    const width = page.view[2] - page.view[0];
    const height = page.view[3] - page.view[1];
    setPageMetrics((current) => ({
      ...current,
      [pageNumber]: { width, height },
    }));
  };

  const activeHighlightStyle = useMemo<CSSProperties | null>(() => {
    if (!activeCitation || !activeCitation.page_number) {
      return null;
    }
    if (
      activeCitation.bbox_x0 === null
      || activeCitation.bbox_y0 === null
      || activeCitation.bbox_x1 === null
      || activeCitation.bbox_y1 === null
    ) {
      return null;
    }

    const metric = pageMetrics[activeCitation.page_number];
    if (!metric || metric.width <= 0 || metric.height <= 0) {
      return null;
    }

    const scale = pageWidth / metric.width;
    const left = activeCitation.bbox_x0 * scale;
    const width = Math.max(8, (activeCitation.bbox_x1 - activeCitation.bbox_x0) * scale);
    const height = Math.max(8, (activeCitation.bbox_y1 - activeCitation.bbox_y0) * scale);
    const top = activeCitation.bbox_y0 * scale;

    return {
      left,
      top,
      width,
      height,
    };
  }, [activeCitation, pageMetrics, pageWidth]);

  const renderEmptyState = (title: string, description: string) => (
    <Card className="h-full min-h-[70vh] flex items-center justify-center border-dashed">
      <div className="max-w-sm text-center space-y-3 px-6">
        <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-midnight-lighter text-flame">
          <FileText size={24} />
        </div>
        <div>
          <p className="text-sm font-semibold text-text-primary">{title}</p>
          <p className="mt-1 text-sm text-text-muted">{description}</p>
        </div>
      </div>
    </Card>
  );

  if (!documentUrl) {
    return renderEmptyState(
      'No review document available',
      'This review does not have a retrievable PDF source yet.',
    );
  }

  return (
    <Card className="h-full min-h-[100vh] overflow-hidden p-0">
      <div className="border-b border-surface-border px-4 py-3">
        <p className="text-sm font-semibold text-text-primary">Source PDF</p>
        <p className="mt-1 text-xs text-text-muted">
          {activeCitation
            ? `Focused on page ${activeCitation.page_number}${activeCitation.quoted_text ? ` · ${activeCitation.quoted_text}` : ''}`
            : 'Select a finding citation to jump to the matching page and highlight.'}
        </p>
      </div>
      <div ref={containerRef} className="h-[calc(100vh-57px)] overflow-auto bg-midnight/40 p-3">
        <Document
          file={documentUrl}
          loading={<LoadingSpinner className="flex items-center justify-center py-16" sizeClassName="w-10 h-10" />}
          onLoadSuccess={handleDocumentLoadSuccess}
          onLoadError={(error) => {
            setLoading(false);
            setLoadError(error instanceof Error ? error.message : 'Failed to load the PDF document.');
          }}
          error={renderEmptyState('Unable to load PDF', 'The review document could not be opened in the viewer.')}
          noData={renderEmptyState('No PDF selected', 'Select a review with a source document to inspect citations.')}
          className="space-y-4"
        >
          {Array.from({ length: numPages }, (_, index) => {
            const pageNumber = index + 1;
            const isActivePage = activeCitation?.page_number === pageNumber;
            const pageMetric = pageMetrics[pageNumber];
            const pageHeight = pageMetric ? (pageMetric.height * pageWidth) / pageMetric.width : undefined;

            return (
              <div
                key={pageNumber}
                ref={(node) => {
                  pageRefs.current[pageNumber] = node;
                }}
                className={`relative mx-auto w-fit rounded-2xl border bg-white shadow-lg transition-all ${isActivePage ? 'border-flame shadow-flame/20' : 'border-surface-border/50'}`}
              >
                <Page
                  pageNumber={pageNumber}
                  width={pageWidth}
                  renderAnnotationLayer={false}
                  renderTextLayer={false}
                  loading={<LoadingSpinner className="flex items-center justify-center py-8" sizeClassName="w-6 h-6" />}
                  onLoadSuccess={(page) => handlePageLoadSuccess(pageNumber, page)}
                />
                {isActivePage && activeHighlightStyle && (
                  <div
                    className="pointer-events-none absolute rounded-md border-2 border-flame bg-flame/20 shadow-[0_0_0_1px_rgba(255,255,255,0.35),0_0_24px_rgba(240,89,65,0.4)]"
                    style={activeHighlightStyle}
                  />
                )}
                {isActivePage && !activeHighlightStyle && pageHeight && (
                  <div
                    className="pointer-events-none absolute inset-3 rounded-xl border-2 border-dashed border-flame/70 bg-flame/5"
                    style={{ height: pageHeight - 24 }}
                  />
                )}
                <div className="absolute left-3 top-3 rounded-full bg-midnight/85 px-2 py-1 text-[11px] font-semibold text-white shadow">
                  Page {pageNumber}
                </div>
              </div>
            );
          })}
        </Document>
        {!loading && !loadError && numPages === 0 && renderEmptyState(
          'PDF has no pages',
          'The review document loaded, but no pages were available to render.',
        )}
        {loadError && renderEmptyState('Unable to load PDF', loadError)}
      </div>
    </Card>
  );
}
