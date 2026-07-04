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
import '@xyflow/react/dist/style.css';

const stageColors: Record<string, { bg: string; border: string; text: string }> = {
  pending:  { bg: '#2a1435', border: '#3e2050', text: '#8a6e9c' },
  active:   { bg: '#F05941', border: '#BE3144', text: '#ffffff' },
  done:     { bg: '#10b981', border: '#059669', text: '#ffffff' },
  error:    { bg: '#BE3144', border: '#872341', text: '#ffffff' },
};

const STAGE_WEIGHTS: Record<string, number> = {
  '4_parameter_resolution': 0,
  '6_7_concurrent_debate': 1, // backend emits this before per-branch stages
  '6_text_debate': 1,
  '7_diagram_debate': 1,
  '8_overview': 2,
};

function getStageState(reviewStatus: string, currentStage: string | undefined, nodeStageId: string): string {
  if (reviewStatus === 'failed') return 'error';
  if (reviewStatus === 'completed_clean' || reviewStatus === 'completed_with_findings') return 'done';
  if (reviewStatus === 'pending') return 'pending';

  if (!currentStage) return 'active'; // Fallback if missing

  if (currentStage === '6_7_concurrent_debate') {
    if (nodeStageId === '4_parameter_resolution') return 'done';
    if (nodeStageId === '7_diagram_debate') return 'active';
    return 'pending';
  }

  const currentWeight = STAGE_WEIGHTS[currentStage] ?? -1;

  let nodeWeight = -1;
  if (nodeStageId === '4_parameter_resolution') nodeWeight = 0;
  if (nodeStageId === '6_text_debate' || nodeStageId === '7_diagram_debate') nodeWeight = 1; // concurrent
  if (nodeStageId === '8_overview') nodeWeight = 2;

  if (currentWeight === -1 || nodeWeight === -1) return 'pending';
  if (nodeWeight < currentWeight) return 'done';
  if (nodeWeight === currentWeight) return 'active';
  return 'pending';
}

interface Props {
  reviewStatus: string;
  currentStage?: string;
}

export default function ReviewPipeline({ reviewStatus, currentStage }: Props) {
  const { nodes, edges } = useMemo(() => {
    const createNode = (id: string, label: string, x: number, y: number, state: string) => {
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

    const ns: Node[] = [
      createNode('n4', '1. Parameter Resolution', 0, 40, getStageState(reviewStatus, currentStage, '4_parameter_resolution')),
      createNode('n6', '2A. Text Debate', 300, 0, getStageState(reviewStatus, currentStage, '6_text_debate')),
      createNode('n7', '2B. Diagram Debate', 300, 80, getStageState(reviewStatus, currentStage, '7_diagram_debate')),
      createNode('n8', '3. Generate Overview', 600, 40, getStageState(reviewStatus, currentStage, '8_overview')),
    ];

    const isEdgeActive = (targetStageId: string) => {
      if (reviewStatus !== 'running') return false;
      if (!currentStage) return true; // animate all if we don't know
      if (currentStage === '6_7_concurrent_debate') {
        return targetStageId === '7_diagram_debate';
      }
      const currentWeight = STAGE_WEIGHTS[currentStage] ?? -1;
      let targetWeight = -1;
      if (targetStageId === '4_parameter_resolution') targetWeight = 0;
      if (targetStageId === '6_text_debate' || targetStageId === '7_diagram_debate') targetWeight = 1;
      if (targetStageId === '8_overview') targetWeight = 2;
      return currentWeight >= targetWeight;
    };

    const createEdge = (source: string, target: string, animated: boolean) => ({
      id: `e-${source}-${target}`,
      source,
      target,
      animated,
      style: { stroke: '#872341', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#872341', width: 16, height: 16 },
    });

    const es: Edge[] = [
      createEdge('n4', 'n6', isEdgeActive('6_text_debate')),
      createEdge('n4', 'n7', isEdgeActive('7_diagram_debate')),
      createEdge('n6', 'n8', isEdgeActive('8_overview')),
      createEdge('n7', 'n8', isEdgeActive('8_overview')),
    ];

    return { nodes: ns, edges: es };
  }, [reviewStatus, currentStage]);

  return (
    <div className="h-48 w-full rounded-xl border border-surface-border overflow-hidden bg-midnight">
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
