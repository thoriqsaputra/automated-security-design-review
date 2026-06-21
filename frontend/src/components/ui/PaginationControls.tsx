interface PaginationControlsProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
  pageSize?: number;
  pageSizeOptions?: number[];
  onPageSizeChange?: (pageSize: number) => void;
}

const MAX_PAGE_BUTTONS = 5;

function buildVisiblePages(currentPage: number, totalPages: number) {
  const visibleCount = Math.min(MAX_PAGE_BUTTONS, totalPages);
  const pages: number[] = [];
  let start = 1;

  if (totalPages > MAX_PAGE_BUTTONS) {
    start = Math.max(1, currentPage - 2);
    start = Math.min(start, totalPages - MAX_PAGE_BUTTONS + 1);
  }

  for (let index = 0; index < visibleCount; index += 1) {
    pages.push(start + index);
  }

  return pages;
}

export default function PaginationControls({
  currentPage,
  totalPages,
  onPageChange,
  pageSize,
  pageSizeOptions,
  onPageSizeChange,
}: PaginationControlsProps) {
  if (totalPages <= 1) {
    return null;
  }

  const visiblePages = buildVisiblePages(currentPage, totalPages);

  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-4 bg-surface-base/50 border border-surface-border rounded-xl p-3">
      {pageSize && onPageSizeChange && pageSizeOptions?.length ? (
        <div className="flex items-center gap-2">
          <span className="text-sm text-text-muted">Items per page:</span>
          <select
            value={pageSize}
            onChange={(event) => onPageSizeChange(Number(event.target.value))}
            className="bg-midnight border border-surface-border text-sm rounded-lg px-2 py-1 text-text-primary focus:outline-none"
          >
            {pageSizeOptions.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </div>
      ) : (
        <div />
      )}

      <div className="flex items-center gap-1.5">
        <button
          onClick={() => onPageChange(Math.max(1, currentPage - 1))}
          disabled={currentPage === 1}
          className="px-3 py-1.5 text-sm font-medium rounded-lg border border-surface-border bg-midnight hover:bg-surface-hover disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-text-primary"
        >
          Previous
        </button>

        <div className="flex items-center gap-1 mx-2">
          {visiblePages.map((pageNumber) => (
            <button
              key={pageNumber}
              onClick={() => onPageChange(pageNumber)}
              className={`w-8 h-8 flex items-center justify-center text-sm font-medium rounded-lg border transition-colors ${
                currentPage === pageNumber
                  ? 'bg-flame/20 border-flame text-flame'
                  : 'border-surface-border bg-midnight hover:bg-surface-hover text-text-primary'
              }`}
            >
              {pageNumber}
            </button>
          ))}
        </div>

        <button
          onClick={() => onPageChange(Math.min(totalPages, currentPage + 1))}
          disabled={currentPage === totalPages}
          className="px-3 py-1.5 text-sm font-medium rounded-lg border border-surface-border bg-midnight hover:bg-surface-hover disabled:opacity-50 disabled:cursor-not-allowed transition-colors text-text-primary"
        >
          Next
        </button>
      </div>
    </div>
  );
}
