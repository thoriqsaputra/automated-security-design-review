import React, { useMemo } from 'react';
import {
  ReactFlow,
  type Node,
  type Edge,
  Position,
  MarkerType,
  Background,
  BackgroundVariant,
} from '@xyflow/react';

const stageColors: Record<string, { bg: string; border: string; text: string }> = {
  pending:  { bg: '#2a1435', border: '#3e2050', text: '#8a6e9c' },
  active:   { bg: '#F05941', border: '#BE3144', text: '#ffffff' },
  done:     { bg: '#10b981', border: '#059669', text: '#ffffff' },
  error:    { bg: '#BE3144', border: '#872341', text: '#ffffff' },
};

const stages = [
  { id: 'prep',     label: 'Preparation' },
  { id: 'hunter',   label: 'Hunter Agent' },
  { id: 'critic',   label: 'Critic Agent' },
  { id: 'mediator', label: 'Mediator Agent' },
  { id: 'complete', label: 'Complete' },
];

function getStageState(reviewStatus: string, stageIndex: number): string {
  const statusMap: Record<string, number> = {
    pending: -1,
    running: 1,
    completed_clean: 5,
    completed_with_findings: 5,
    failed: -2,
  };
  const current = statusMap[reviewStatus] ?? -1;
  if (current === -2) return 'error';
  if (stageIndex < current) return 'done';
  if (stageIndex === current) return 'active';
  return 'pending';
}

interface Props {
  reviewStatus: string;
}

export default function ReviewPipeline({ reviewStatus }: Props) {
  const { nodes, edges } = useMemo(() => {
    const ns: Node[] = stages.map((stage, i) => {
      const state = getStageState(reviewStatus, i);
      const colors = stageColors[state];
      return {
        id: stage.id,
        position: { x: i * 220, y: 40 },
        data: { label: stage.label },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
        style: {
          background: colors.bg,
          border: `2px solid ${colors.border}`,
          color: colors.text,
          borderRadius: '12px',
          padding: '12px 20px',
          fontSize: '13px',
          fontWeight: 600,
          fontFamily: 'Inter, sans-serif',
          boxShadow: state === 'active' ? `0 0 20px ${colors.border}80` : 'none',
        },
      };
    });

    const es: Edge[] = stages.slice(0, -1).map((stage, i) => ({
      id: `e-${stage.id}-${stages[i + 1].id}`,
      source: stage.id,
      target: stages[i + 1].id,
      animated: getStageState(reviewStatus, i) === 'active' || getStageState(reviewStatus, i + 1) === 'active',
      style: { stroke: '#872341', strokeWidth: 2 },
      markerEnd: { type: MarkerType.ArrowClosed, color: '#872341', width: 16, height: 16 },
    }));

    return { nodes: ns, edges: es };
  }, [reviewStatus]);

  return (
    <div className="h-40 w-full rounded-xl border border-surface-border overflow-hidden bg-midnight">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        panOnDrag={false}
        zoomOnScroll={false}
        zoomOnPinch={false}
        zoomOnDoubleClick={false}
        preventScrolling={false}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#3e2050" />
      </ReactFlow>
    </div>
  );
}
