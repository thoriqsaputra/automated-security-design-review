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

const STAGES_ORDER = [
  '1_ingestion',
  '2_retrieval',
  '3_asvs_classification',
  '4_parameter_resolution',
  '5_parent_retrieval',
  '6_text_debate',
  '7_diagram_debate',
  '8_overview'
];

function getStageState(reviewStatus: string, currentStage: string | undefined, nodeStageId: string): string {
  if (reviewStatus === 'failed') return 'error';
  if (reviewStatus === 'completed_clean' || reviewStatus === 'completed_with_findings') return 'done';
  if (reviewStatus === 'pending') return 'pending';
  
  if (!currentStage) return 'active'; // Fallback if missing
  
  const currentIndex = STAGES_ORDER.indexOf(currentStage);
  const nodeIndex = STAGES_ORDER.indexOf(nodeStageId);
  
  if (currentIndex === -1 || nodeIndex === -1) return 'pending';
  if (nodeIndex < currentIndex) return 'done';
  if (nodeIndex === currentIndex) return 'active';
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
      createNode('n1', '1. Ingestion & Screening', 0, 40, getStageState(reviewStatus, currentStage, '1_ingestion')),
      createNode('n2', '2. Retrieval Indexing', 250, 40, getStageState(reviewStatus, currentStage, '2_retrieval')),
      createNode('n3', '3. ASVS Classification', 500, 40, getStageState(reviewStatus, currentStage, '3_asvs_classification')),
      createNode('n4', '4. Parameter Resolution', 750, 40, getStageState(reviewStatus, currentStage, '4_parameter_resolution')),
      createNode('n5', '5. Parent Retrieval', 1000, 40, getStageState(reviewStatus, currentStage, '5_parent_retrieval')),
      createNode('n6', '6. Text Multi-Agent Debate Loop', 1250, 40, getStageState(reviewStatus, currentStage, '6_text_debate')),
      createNode('n7', '7. Diagram Multi-Agent Debate Loop', 1500, 40, getStageState(reviewStatus, currentStage, '7_diagram_debate')),
      createNode('n8', '8. Generate Overview', 1750, 40, getStageState(reviewStatus, currentStage, '8_overview')),
    ];

    const isEdgeActive = (targetStage: string) => {
      if (reviewStatus !== 'running') return false;
      if (!currentStage) return true; // animate all if we don't know
      const currentIndex = STAGES_ORDER.indexOf(currentStage);
      const targetIndex = STAGES_ORDER.indexOf(targetStage);
      // Animate if the current stage is at or beyond the target stage
      return currentIndex >= targetIndex;
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
      createEdge('n1', 'n2', isEdgeActive('2_retrieval')),
      createEdge('n2', 'n3', isEdgeActive('3_asvs_classification')),
      createEdge('n3', 'n4', isEdgeActive('4_parameter_resolution')),
      createEdge('n4', 'n5', isEdgeActive('5_parent_retrieval')),
      createEdge('n5', 'n6', isEdgeActive('6_text_debate')),
      createEdge('n6', 'n7', isEdgeActive('7_diagram_debate')),
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
