import { CheckCircle2, ChevronDown, ChevronRight, Filter, Search, Trash2, X } from 'lucide-react';
import type { DiagramRequirement, ParameterParent } from '../../../api/standards';
import Card from '../../../components/ui/Card';
import PaginationControls from '../../../components/ui/PaginationControls';
import {
  asvsLevelClass,
  asvsLevelLabel,
  parameterPageSizeOptions,
  parentLevelSummary,
  selectArrowStyle,
} from '../utils/standardsPresentation';

interface CategoryParametersPanelProps {
  activeTab: 'requirements' | 'diagram';
  onTabChange: (tab: 'requirements' | 'diagram') => void;
  hasAnyContent: boolean;
  search: string;
  searchInput: string;
  onSearchInputChange: (value: string) => void;
  onSearchCommit: () => void;
  filterAsvsLevel: string;
  onFilterAsvsLevelChange: (value: string) => void;
  onClearFilters: () => void;
  parameters: ParameterParent[];
  filteredParameters: ParameterParent[];
  parameterCounts: { total: number; byLevel: Record<string, number> };
  paginatedParams: ParameterParent[];
  paramsPage: number;
  totalParamsPages: number;
  pageSize: number;
  onParamsPageChange: (page: number) => void;
  onPageSizeChange: (pageSize: number) => void;
  expandedParent: number | null;
  onToggleParent: (parentId: number) => void;
  onDeleteParent: (parentId: number) => void;
  onDeleteChild: (childId: number) => void;
  diagramRequirements: DiagramRequirement[];
  filteredDiagramRequirements: DiagramRequirement[];
  diagramCounts: { total: number; byLevel: Record<string, number> };
  paginatedDiagramRequirements: DiagramRequirement[];
  diagramParamsPage: number;
  totalDiagramParamsPages: number;
  onDiagramParamsPageChange: (page: number) => void;
}

function LevelCountChips({ counts }: { counts: Record<string, number> }) {
  return (
    <>
      {counts.L1 > 0 && (
        <span className="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-semibold bg-emerald-500/15 text-emerald-400">
          L1: {counts.L1}
        </span>
      )}
      {counts.L2 > 0 && (
        <span className="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-semibold bg-sky-500/15 text-sky-400">
          L2: {counts.L2}
        </span>
      )}
      {counts.L3 > 0 && (
        <span className="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-semibold bg-fuchsia-500/15 text-fuchsia-400">
          L3: {counts.L3}
        </span>
      )}
      {counts.Unknown > 0 && (
        <span className="inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-semibold bg-surface-hover text-text-muted">
          Unknown: {counts.Unknown}
        </span>
      )}
    </>
  );
}

export default function CategoryParametersPanel(props: CategoryParametersPanelProps) {
  const {
    activeTab,
    onTabChange,
    hasAnyContent,
    search,
    searchInput,
    onSearchInputChange,
    onSearchCommit,
    filterAsvsLevel,
    onFilterAsvsLevelChange,
    onClearFilters,
    parameters,
    filteredParameters,
    parameterCounts,
    paginatedParams,
    paramsPage,
    totalParamsPages,
    pageSize,
    onParamsPageChange,
    onPageSizeChange,
    expandedParent,
    onToggleParent,
    onDeleteParent,
    onDeleteChild,
    diagramRequirements,
    filteredDiagramRequirements,
    diagramCounts,
    paginatedDiagramRequirements,
    diagramParamsPage,
    totalDiagramParamsPages,
    onDiagramParamsPageChange,
  } = props;

  return (
    <>
      <div className="border-b border-surface-border mb-6 flex gap-6 px-2 overflow-x-auto no-scrollbar">
        {[
          { id: 'requirements' as const, label: 'Requirement Text' },
          { id: 'diagram' as const, label: 'Diagram Requirement' },
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`pb-3 text-sm font-semibold transition-colors border-b-2 whitespace-nowrap ${
              activeTab === tab.id
                ? 'border-flame text-flame'
                : 'border-transparent text-text-muted hover:text-text-primary'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {hasAnyContent && (
        <div className="flex flex-col sm:flex-row flex-wrap gap-3 bg-surface-base/50 p-3 rounded-xl border border-surface-border items-center mb-6">
          <div className="relative flex-1 min-w-[200px]">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-text-muted" size={16} />
            <input
              type="text"
              placeholder={activeTab === 'requirements' ? 'Search requirement text...' : 'Search diagram requirements...'}
              value={searchInput}
              onChange={(event) => onSearchInputChange(event.target.value)}
              onKeyDown={(event) => event.key === 'Enter' && onSearchCommit()}
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
                value={filterAsvsLevel}
                onChange={(event) => onFilterAsvsLevelChange(event.target.value)}
                className="bg-transparent text-sm text-text-primary focus:outline-none py-1 pr-6 cursor-pointer appearance-none"
                style={selectArrowStyle}
              >
                <option value="all">ASVS Level: All</option>
                <option value="1">ASVS Level: L1</option>
                <option value="2">ASVS Level: L2</option>
                <option value="3">ASVS Level: L3</option>
                <option value="unknown">ASVS Level: Unknown</option>
              </select>
            </div>

            {(search || filterAsvsLevel !== 'all') && (
              <button onClick={onClearFilters} className="text-xs font-semibold text-text-muted hover:text-text-primary transition-colors ml-2">
                Clear All
              </button>
            )}
          </div>
        </div>
      )}

      {activeTab === 'requirements' ? (
        <div className="space-y-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex flex-col gap-1.5">
              <h2 className="text-lg font-semibold text-text-primary">Extracted Parameters</h2>
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-sm text-text-muted">{filteredParameters.length} sections · {parameterCounts.total} requirements</span>
                <LevelCountChips counts={parameterCounts.byLevel} />
              </div>
            </div>
          </div>

          {parameters.length === 0 ? (
            <Card>
              <p className="text-sm text-text-muted text-center py-4">No parameters extracted yet. Complete an ingestion job first.</p>
            </Card>
          ) : filteredParameters.length === 0 ? (
            <Card>
              <p className="text-sm text-text-muted text-center py-4">No parameters match the current filters.</p>
            </Card>
          ) : (
            <div className="space-y-4">
              <div className="space-y-2">
                {paginatedParams.map((parent) => (
                  <Card key={parent.id}>
                    <div className="flex items-center justify-between">
                      <button onClick={() => onToggleParent(parent.id)} className="flex-1 flex items-center justify-between text-left">
                        <div className="flex items-center gap-2">
                          {expandedParent === parent.id ? (
                            <ChevronDown size={16} className="text-flame" />
                          ) : (
                            <ChevronRight size={16} className="text-text-muted" />
                          )}
                          <div>
                            <p className="text-sm font-medium text-text-primary">{parent.title}</p>
                            <p className="text-xs text-text-muted">
                              {parent.children.length} requirement(s)
                              {parentLevelSummary(parent) && ` · ${parentLevelSummary(parent)}`}
                            </p>
                          </div>
                        </div>
                        <span className="text-xs text-text-muted font-mono">{parent.stable_key}</span>
                      </button>
                      <button
                        onClick={() => onDeleteParent(parent.id)}
                        className="ml-3 flex items-center justify-center p-1.5 rounded-lg text-text-muted hover:text-flame hover:bg-flame/10 transition-colors shrink-0"
                        title="Delete Section"
                      >
                        <Trash2 size={16} />
                      </button>
                    </div>

                    {expandedParent === parent.id && parent.children.length > 0 && (
                      <div className="mt-3 ml-6 space-y-2 border-l-2 border-surface-border pl-4">
                        {parent.children.map((child) => (
                          <div key={child.id} className="flex items-start justify-between gap-4 group">
                            <div className="flex items-start gap-2">
                              <CheckCircle2 size={14} className="text-burgundy mt-0.5 shrink-0" />
                              <div>
                                <div className="flex flex-wrap items-center gap-2">
                                  <p className="text-sm text-text-primary">{child.requirement_text}</p>
                                  <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-semibold ${asvsLevelClass(child.asvs_level)}`}>
                                    {asvsLevelLabel(child.asvs_level)}
                                  </span>
                                </div>
                                {child.details && <p className="text-xs text-text-muted mt-0.5">{child.details}</p>}
                              </div>
                            </div>
                            <button
                              onClick={() => onDeleteChild(child.id)}
                              className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-flame/10 text-text-muted hover:text-flame transition-all shrink-0"
                              title="Delete Requirement"
                            >
                              <Trash2 size={14} />
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </Card>
                ))}
              </div>

              <PaginationControls
                currentPage={paramsPage}
                totalPages={totalParamsPages}
                onPageChange={onParamsPageChange}
                pageSize={pageSize}
                pageSizeOptions={parameterPageSizeOptions}
                onPageSizeChange={onPageSizeChange}
              />
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-4">
          <div className="flex items-center justify-between mb-4">
            <div className="flex flex-col gap-1.5">
              <h2 className="text-lg font-semibold text-text-primary">Extracted Diagram Requirements</h2>
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-sm text-text-muted">{diagramCounts.total} requirements</span>
                <LevelCountChips counts={diagramCounts.byLevel} />
              </div>
            </div>
          </div>

          {diagramRequirements.length === 0 ? (
            <Card>
              <p className="text-sm text-text-muted text-center py-4">No diagram requirements extracted yet. Complete an ingestion job first.</p>
            </Card>
          ) : filteredDiagramRequirements.length === 0 ? (
            <Card>
              <p className="text-sm text-text-muted text-center py-4">No diagram requirements match the current filters.</p>
            </Card>
          ) : (
            <div className="space-y-3">
              {paginatedDiagramRequirements.map((requirement) => (
                <Card key={requirement.id}>
                  <div className="flex items-start gap-4">
                    <div className="mt-1">
                      <span className={`inline-flex items-center px-2 py-0.5 rounded-md text-[11px] font-semibold ${asvsLevelClass(requirement.asvs_level)}`}>
                        {asvsLevelLabel(requirement.asvs_level)}
                      </span>
                    </div>
                    <div className="flex-1">
                      <p className="text-sm font-medium text-text-primary">{requirement.requirement_text}</p>
                      <p className="text-xs text-text-muted mt-1 italic">{requirement.verification_hint}</p>
                      <div className="flex items-center gap-2 mt-2">
                        <span className="text-[10px] font-mono text-text-muted bg-surface px-1.5 py-0.5 rounded">{requirement.stable_key}</span>
                        <span className="text-[10px] font-medium text-text-secondary">Parent: {requirement.parent_section}</span>
                      </div>
                    </div>
                  </div>
                </Card>
              ))}

              <PaginationControls
                currentPage={diagramParamsPage}
                totalPages={totalDiagramParamsPages}
                onPageChange={onDiagramParamsPageChange}
                pageSize={pageSize}
                pageSizeOptions={parameterPageSizeOptions}
                onPageSizeChange={onPageSizeChange}
              />
            </div>
          )}
        </div>
      )}
    </>
  );
}
