import { Filter, Search, X } from 'lucide-react';
import type { Finding, Review } from '../../../api/reviews';
import Card from '../../../components/ui/Card';
import LoadingSpinner from '../../../components/ui/LoadingSpinner';
import PaginationControls from '../../../components/ui/PaginationControls';
import { FindingDetails } from './ReviewDetailPrimitives';
import { metStatusIcons, severityColors } from '../utils/reviewPresentation';

interface ReviewFindingsPanelProps {
  review: Review;
  findings: Finding[];
  totalFindings: number;
  totalPages: number;
  loadingFindings: boolean;
  expandedFinding: number | null;
  onToggleFinding: (findingId: number) => void;
  search: string;
  searchInput: string;
  onSearchInputChange: (value: string) => void;
  onSearchCommit: () => void;
  filterMetStatus: string;
  filterSeverity: string;
  filterFindingType: string;
  onFilterMetStatusChange: (value: string) => void;
  onFilterSeverityChange: (value: string) => void;
  onFilterFindingTypeChange: (value: string) => void;
  onClearFilters: () => void;
  currentPage: number;
  onCurrentPageChange: (page: number) => void;
  pageSize: number;
  onPageSizeChange: (pageSize: number) => void;
}

const findingsPageSizeOptions = [10, 20, 50, 100];

const selectArrowStyle = {
  backgroundImage:
    'url("data:image/svg+xml;charset=UTF-8,%3csvg xmlns=\'http://www.w3.org/2000/svg\' viewBox=\'0 0 24 24\' fill=\'none\' stroke=\'currentColor\' stroke-width=\'2\' stroke-linecap=\'round\' stroke-linejoin=\'round\'%3e%3cpolyline points=\'6 9 12 15 18 9\'%3e%3c/polyline%3e%3c/svg%3e")',
  backgroundRepeat: 'no-repeat',
  backgroundPosition: 'right 0.25rem center',
  backgroundSize: '1em',
} as const;

export default function ReviewFindingsPanel(props: ReviewFindingsPanelProps) {
  const {
    review,
    findings,
    totalFindings,
    totalPages,
    loadingFindings,
    expandedFinding,
    onToggleFinding,
    search,
    searchInput,
    onSearchInputChange,
    onSearchCommit,
    filterMetStatus,
    filterSeverity,
    filterFindingType,
    onFilterMetStatusChange,
    onFilterSeverityChange,
    onFilterFindingTypeChange,
    onClearFilters,
    currentPage,
    onCurrentPageChange,
    pageSize,
    onPageSizeChange,
  } = props;

  return (
    <div className="space-y-4 animate-fade-in">
      <div className="flex flex-col gap-4 mb-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-text-primary">Findings ({totalFindings})</h2>
        </div>

        <div className="flex flex-col sm:flex-row flex-wrap gap-3 bg-surface-base/50 p-3 rounded-xl border border-surface-border items-center">
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" size={16} />
            <input
              type="text"
              placeholder="Search findings..."
              value={searchInput}
              onChange={(event) => onSearchInputChange(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  onSearchCommit();
                }
              }}
              onBlur={onSearchCommit}
              className="w-full bg-midnight border border-surface-border text-sm rounded-lg pl-9 pr-3 py-2 text-text-primary focus:outline-none focus:border-flame transition-colors"
            />
            {search && (
              <button onClick={onClearFilters} className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary">
                <X size={14} />
              </button>
            )}
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <div className="flex items-center gap-1.5 bg-midnight border border-surface-border rounded-lg px-1 py-1">
              <Filter size={14} className="text-text-muted ml-2" />
              <select
                value={filterMetStatus}
                onChange={(event) => onFilterMetStatusChange(event.target.value)}
                className="bg-transparent text-sm text-text-primary focus:outline-none py-1 pr-6 cursor-pointer appearance-none"
                style={selectArrowStyle}
              >
                <option value="all">Status: All</option>
                <option value="met">Status: Met</option>
                <option value="not_met">Status: Not Met</option>
                <option value="partially_met">Status: Partially Met</option>
                <option value="na">Status: N/A</option>
              </select>
            </div>

            <div className="flex items-center gap-1.5 bg-midnight border border-surface-border rounded-lg px-1 py-1">
              <Filter size={14} className="text-text-muted ml-2" />
              <select
                value={filterSeverity}
                onChange={(event) => onFilterSeverityChange(event.target.value)}
                className="bg-transparent text-sm text-text-primary focus:outline-none py-1 pr-6 cursor-pointer appearance-none"
                style={selectArrowStyle}
              >
                <option value="all">Severity: All</option>
                <option value="critical">Severity: Critical</option>
                <option value="high">Severity: High</option>
                <option value="medium">Severity: Medium</option>
                <option value="low">Severity: Low</option>
                <option value="info">Severity: Info</option>
              </select>
            </div>

            <div className="flex items-center gap-1.5 bg-midnight border border-surface-border rounded-lg px-1 py-1">
              <Filter size={14} className="text-text-muted ml-2" />
              <select
                value={filterFindingType}
                onChange={(event) => onFilterFindingTypeChange(event.target.value)}
                className="bg-transparent text-sm text-text-primary focus:outline-none py-1 pr-6 cursor-pointer appearance-none"
                style={selectArrowStyle}
              >
                <option value="all">Type: All</option>
                <option value="requirement">Type: Requirement</option>
                <option value="diagram">Type: Diagram</option>
              </select>
            </div>

            {(search || filterMetStatus !== 'all' || filterSeverity !== 'all' || filterFindingType !== 'all') && (
              <button onClick={onClearFilters} className="text-xs font-semibold text-text-muted hover:text-text-primary transition-colors ml-2">
                Clear All
              </button>
            )}
          </div>
        </div>
      </div>

      {loadingFindings ? (
        <Card>
          <LoadingSpinner className="flex items-center justify-center py-8" sizeClassName="w-6 h-6" />
        </Card>
      ) : findings.length === 0 ? (
        <Card>
          <p className="text-sm text-text-muted text-center py-8">
            {review.status === 'running' && totalFindings === 0
              ? 'Analysis in progress...'
              : 'No findings match the current filters.'}
          </p>
        </Card>
      ) : (
        <div className="space-y-4">
          <div className="space-y-2">
            {findings.map((finding) => (
              <Card key={finding.id}>
                <button onClick={() => onToggleFinding(finding.id)} className="w-full text-left">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {metStatusIcons[finding.met_status || ''] || metStatusIcons.not_applicable}
                      <div>
                        <p className="text-sm font-medium text-text-primary">{finding.title}</p>
                        <p className="text-xs text-text-muted">{finding.parent_parameter_title || finding.finding_type}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      {finding.severity && (
                        <span className={`text-xs font-medium px-2 py-0.5 rounded-full border ${severityColors[finding.severity] || severityColors.info}`}>
                          {finding.severity}
                        </span>
                      )}
                      {finding.met_status && (
                        <span className="text-xs text-text-muted">{finding.met_status.replace(/_/g, ' ')}</span>
                      )}
                    </div>
                  </div>
                </button>

                {expandedFinding === finding.id && <FindingDetails finding={finding} />}
              </Card>
            ))}
          </div>

          <PaginationControls
            currentPage={currentPage}
            totalPages={totalPages}
            onPageChange={onCurrentPageChange}
            pageSize={pageSize}
            pageSizeOptions={findingsPageSizeOptions}
            onPageSizeChange={onPageSizeChange}
          />
        </div>
      )}
    </div>
  );
}
