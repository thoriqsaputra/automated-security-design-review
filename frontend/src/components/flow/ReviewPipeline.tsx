import { useMemo } from 'react';
import {
  ReactFlow,
  type Node,
  type Edge,
  Position,
  MarkerType,
  Background,
  BackgroundVariant,
} from '@xyflow/react';
import type { DebateStreamState, ReviewAnalysisMode } from '../../api/reviews';
import '@xyflow/react/dist/style.css';

type StageState = 'pending' | 'active' | 'done' | 'error';

const stageColors: Record<StageState, { bg: string; border: string; text: string }> = {
  pending: { bg: '#2a1435', border: '#3e2050', text: '#8a6e9c' },
  active: { bg: '#F05941', border: '#BE3144', text: '#ffffff' },
  done: { bg: '#10b981', border: '#059669', text: '#ffffff' },
  error: { bg: '#BE3144', border: '#872341', text: '#ffffff' },
};

const ANALYSIS_STAGES = new Set(['6_7_concurrent_debate', '6_text_debate', '7_diagram_debate']);
const TERMINAL_DEBATE_STATUSES = new Set(['completed', 'failed', 'cancelled']);
const COMPLETED_REVIEW_STATUSES = new Set(['completed_clean', 'completed_with_findings', 'approved', 'rejected']);

function isTerminalDebate(debate: DebateStreamState): boolean {
  return TERMINAL_DEBATE_STATUSES.has(debate.status);
}

function hasReachedDebate(debate: DebateStreamState): boolean {
  return (
    isTerminalDebate(debate) ||
    debate.work_phase === 'debate' ||
    debate.work_phase === 'persistence' ||
    debate.active_agent !== null
  );
}

function isWaitingForRetrieval(debate: DebateStreamState): boolean {
  if (isTerminalDebate(debate)) return false;
  if (debate.work_phase === 'queued' || debate.work_phase === 'retrieval') return true;
  if (debate.work_phase === 'debate' || debate.work_phase === 'persistence') return false;

  // Older stream snapshots may not contain work_phase. An active agent means
  // retrieval has already handed the parameter to the debate pipeline.
  return debate.active_agent === null;
}

function derivePipelineStates(
  reviewStatus: string,
  currentStage: string | undefined,
  debates: DebateStreamState[],
) {
  const failed = reviewStatus === 'failed';
  const completed = COMPLETED_REVIEW_STATUSES.has(reviewStatus);
  const atAnalysis = currentStage !== undefined && ANALYSIS_STAGES.has(currentStage);
  const atOverview = currentStage === '8_overview';
  const textDebates = debates.filter((debate) => debate.finding_type === 'requirement');
  const diagramDebates = debates.filter((debate) => debate.finding_type === 'diagram');

  if (failed) {
    return {
      parameter: 'error' as StageState,
      retrieval: 'error' as StageState,
      text: 'error' as StageState,
      diagram: 'error' as StageState,
      overview: 'error' as StageState,
    };
  }

  let parameter: StageState = 'pending';
  if (completed || atAnalysis || atOverview) parameter = 'done';
  else if (currentStage === '4_parameter_resolution' || (reviewStatus === 'running' && !currentStage)) parameter = 'active';

  let retrieval: StageState = 'pending';
  let text: StageState = 'pending';
  let diagram: StageState = 'pending';
  let overview: StageState = 'pending';

  if (completed || atOverview) {
    retrieval = 'done';
    text = 'done';
    diagram = 'done';
  } else if (atAnalysis) {
    // The retrieval branch is complete only after every text parameter has
    // either entered debate/persistence or reached a terminal state.
    retrieval = textDebates.length > 0 && !textDebates.some(isWaitingForRetrieval) ? 'done' : 'active';

    if (textDebates.length > 0 && textDebates.every(isTerminalDebate)) {
      text = 'done';
    } else if (textDebates.some(hasReachedDebate)) {
      text = 'active';
    }

    if (diagramDebates.length > 0 && diagramDebates.every(isTerminalDebate)) {
      diagram = 'done';
    } else {
      // Diagram discovery/filtering happens before its first live debate is
      // published, so the branch is active even while its list is still empty.
      diagram = 'active';
    }
  }

  if (completed) overview = 'done';
  else if (atOverview) overview = 'active';

  return { parameter, retrieval, text, diagram, overview };
}

interface Props {
  reviewStatus: string;
  currentStage?: string;
  debates: DebateStreamState[];
  analysisMode: ReviewAnalysisMode;
}

export default function ReviewPipeline({ reviewStatus, currentStage, debates, analysisMode }: Props) {
  const { nodes, edges } = useMemo(() => {
    const states = derivePipelineStates(reviewStatus, currentStage, debates);
    const showText = analysisMode !== 'diagram_only';
    const showDiagram = analysisMode !== 'text_only';

    const createNode = (id: string, label: string, x: number, y: number, state: StageState) => {
      const colors = stageColors[state];
      return {
        id,
        position: { x, y },
        data: { label },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
        style: {
          background: colors.bg,
          border: `2px solid ${colors.border}`,
          color: colors.text,
          borderRadius: '12px',
          padding: '12px 20px',
          fontSize: '12px',
          fontWeight: 600,
          fontFamily: 'Inter, sans-serif',
          boxShadow: state === 'active' ? `0 0 20px ${colors.border}80` : 'none',
          width: 200,
          textAlign: 'center' as const,
        },
      };
    };

    const ns: Node[] = [];
    if (showText && showDiagram) {
      ns.push(
        createNode('parameter', '1. Parameter Resolution', 0, 60, states.parameter),
        createNode('retrieval', '2A. Retrieval', 270, 0, states.retrieval),
        createNode('text', '3A. Text Debate', 540, 0, states.text),
        createNode('diagram', '2B. Diagram Analysis', 405, 120, states.diagram),
        createNode('overview', '4. Generate Overview', 810, 60, states.overview),
      );
    } else if (showText) {
      ns.push(
        createNode('parameter', '1. Parameter Resolution', 0, 40, states.parameter),
        createNode('retrieval', '2. Retrieval', 270, 40, states.retrieval),
        createNode('text', '3. Text Debate', 540, 40, states.text),
        createNode('overview', '4. Generate Overview', 810, 40, states.overview),
      );
    } else {
      ns.push(
        createNode('parameter', '1. Parameter Resolution', 0, 40, states.parameter),
        createNode('diagram', '2. Diagram Analysis', 320, 40, states.diagram),
        createNode('overview', '3. Generate Overview', 640, 40, states.overview),
      );
    }

    const createEdge = (source: string, target: string, targetState: StageState) => ({
      id: `e-${source}-${target}`,
      source,
      target,
      animated: reviewStatus === 'running' && targetState === 'active',
      style: { stroke: '#872341', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#872341', width: 16, height: 16 },
    });

    const es: Edge[] = [];
    if (showText) {
      es.push(
        createEdge('parameter', 'retrieval', states.retrieval),
        createEdge('retrieval', 'text', states.text),
        createEdge('text', 'overview', states.overview),
      );
    }
    if (showDiagram) {
      es.push(
        createEdge('parameter', 'diagram', states.diagram),
        createEdge('diagram', 'overview', states.overview),
      );
    }

    return { nodes: ns, edges: es };
  }, [reviewStatus, currentStage, debates, analysisMode]);

  return (
    <div className="h-64 w-full overflow-hidden rounded-xl border border-surface-border bg-midnight">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        panOnDrag={true}
        zoomOnScroll={true}
        zoomOnPinch={true}
        zoomOnDoubleClick={true}
        preventScrolling={false}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        minZoom={0.1}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#3e2050" />
      </ReactFlow>
    </div>
  );
}
