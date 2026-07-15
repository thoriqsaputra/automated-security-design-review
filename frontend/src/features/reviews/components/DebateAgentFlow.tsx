import { useMemo } from 'react';
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  MarkerType,
  Position,
  type Edge,
  type Node,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import { ScanEye, Brain } from 'lucide-react';
import type { DebateAgent, DebatePipelineMode, DebateStreamState, DebateTranscriptMessage } from '../../../api/reviews';

const AGENT_ORDER_BY_MODE: Record<DebatePipelineMode, DebateAgent[]> = {
  debate: ['hunter', 'critic', 'mediator'],
  extract_reason: ['extractor', 'reasoner'],
};

const AGENT_LABELS: Record<DebateAgent, string> = {
  hunter: 'Hunter',
  critic: 'Critic',
  mediator: 'Mediator',
  extractor: 'Extractor',
  reasoner: 'Reasoner',
  system: 'System',
};

const AGENT_ICONS: Partial<Record<DebateAgent, typeof ScanEye>> = {
  extractor: ScanEye,
  reasoner: Brain,
};

const AGENT_COLORS: Record<DebateAgent, { bg: string; border: string; text: string }> = {
  hunter: { bg: '#0c2233', border: '#38bdf8', text: '#e0f2fe' },
  critic: { bg: '#2b1d05', border: '#f59e0b', text: '#fef3c7' },
  mediator: { bg: '#062b22', border: '#10b981', text: '#d1fae5' },
  extractor: { bg: '#1a0f2e', border: '#a78bfa', text: '#ede9fe' },
  reasoner: { bg: '#2a0f14', border: '#872341', text: '#fbd5d5' },
  system: { bg: '#161324', border: '#3e2050', text: '#c9b8d6' },
};

const IDLE_COLORS = { bg: '#161324', border: '#3e2050', text: '#8a6e9c' };
const FAILED_COLORS = { bg: '#2a0f14', border: '#BE3144', text: '#fecaca' };

function latestMessageForAgent(transcript: DebateTranscriptMessage[], agent: DebateAgent) {
  for (let i = transcript.length - 1; i >= 0; i -= 1) {
    if (transcript[i].agent === agent) {
      return transcript[i];
    }
  }
  return null;
}

function agentStatus(debate: DebateStreamState, agent: DebateAgent): 'pending' | 'running' | 'completed' | 'failed' {
  const message = latestMessageForAgent(debate.transcript, agent);
  if (!message) {
    return 'pending';
  }
  if (message.status === 'failed') {
    return 'failed';
  }
  if (message.status === 'running') {
    return 'running';
  }
  return 'completed';
}

interface DebateAgentFlowProps {
  debate: DebateStreamState;
  selectedAgent: DebateAgent | null;
  onSelectAgent: (agent: DebateAgent | null) => void;
}

export default function DebateAgentFlow({ debate, selectedAgent, onSelectAgent }: DebateAgentFlowProps) {
  const agentOrder = AGENT_ORDER_BY_MODE[debate.pipeline_mode ?? 'debate'];
  const { nodes, edges } = useMemo(() => {
    const ns: Node[] = [
      {
        id: 'requirement',
        position: { x: 0, y: 60 },
        data: { label: debate.finding_type === 'diagram' ? 'Diagram' : 'Requirement' },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
        selectable: false,
        draggable: false,
        style: {
          background: '#161324',
          border: '2px solid #3e2050',
          color: '#c9b8d6',
          borderRadius: '12px',
          padding: '10px 16px',
          fontSize: '11px',
          fontWeight: 600,
          fontFamily: 'Inter, sans-serif',
          width: 140,
          textAlign: 'center' as const,
        },
      },
    ];

    agentOrder.forEach((agent, index) => {
      const status = agentStatus(debate, agent);
      const message = latestMessageForAgent(debate.transcript, agent);
      const colors = status === 'failed' ? FAILED_COLORS : status === 'pending' ? IDLE_COLORS : AGENT_COLORS[agent];
      const isSelected = selectedAgent === agent;
      const isActive = debate.active_agent === agent && debate.status === 'running';
      const snippet = (message?.content || '').trim().slice(-140);
      const AgentIcon = AGENT_ICONS[agent];
      ns.push({
        id: agent,
        position: { x: 300 * (index + 1), y: 40 },
        data: {
          label: (
            <div className="text-left">
              <div className="flex items-center justify-between gap-2">
                <span className="flex items-center gap-1.5 text-xs font-semibold uppercase tracking-[0.16em]">
                  {AgentIcon && <AgentIcon className="h-3 w-3" />}
                  {AGENT_LABELS[agent]}
                </span>
                <span
                  className={`h-1.5 w-1.5 rounded-full ${
                    status === 'running' ? 'animate-pulse bg-current' : status === 'failed' ? 'bg-current' : 'bg-current opacity-70'
                  }`}
                />
              </div>
              <div className="mt-1 text-[10px] font-normal uppercase tracking-wider opacity-70">{status}</div>
              {snippet && (
                <p className="mt-2 line-clamp-3 text-[11px] font-normal leading-4 opacity-90">{snippet}</p>
              )}
            </div>
          ),
        },
        sourcePosition: Position.Right,
        targetPosition: Position.Left,
        style: {
          background: colors.bg,
          border: `2px solid ${isSelected ? '#F05941' : colors.border}`,
          color: colors.text,
          borderRadius: '14px',
          padding: '12px 16px',
          fontFamily: 'Inter, sans-serif',
          boxShadow: isActive ? `0 0 22px ${colors.border}90` : isSelected ? '0 0 0 2px #F05941' : 'none',
          width: 260,
          cursor: 'pointer',
        },
      });
    });

    ns.push({
      id: 'verdict',
      position: { x: 300 * (agentOrder.length + 1), y: 60 },
      data: { label: debate.status === 'completed' ? 'Finding Recorded' : 'Verdict' },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
      selectable: false,
      draggable: false,
      style: {
        background: debate.status === 'completed' ? '#062b22' : '#161324',
        border: `2px solid ${debate.status === 'completed' ? '#10b981' : '#3e2050'}`,
        color: debate.status === 'completed' ? '#d1fae5' : '#c9b8d6',
        borderRadius: '12px',
        padding: '10px 16px',
        fontSize: '11px',
        fontWeight: 600,
        fontFamily: 'Inter, sans-serif',
        width: 150,
        textAlign: 'center' as const,
      },
    });

    const chain = ['requirement', ...agentOrder, 'verdict'];
    const terminalAgent = agentOrder[agentOrder.length - 1];
    const es: Edge[] = [];
    for (let i = 0; i < chain.length - 1; i += 1) {
      const source = chain[i];
      const target = chain[i + 1];
      const targetIsActiveAgent = agentOrder.includes(target as DebateAgent) && debate.active_agent === target;
      es.push({
        id: `e-${source}-${target}`,
        source,
        target,
        animated: debate.status === 'running' && (targetIsActiveAgent || (source === terminalAgent && target === 'verdict' && debate.status === 'running')),
        style: { stroke: '#872341', strokeWidth: 2 },
        markerEnd: { type: MarkerType.ArrowClosed, color: '#872341', width: 14, height: 14 },
      });
    }

    return { nodes: ns, edges: es };
  }, [debate, selectedAgent, agentOrder]);

  return (
    <div className="h-80 lg:h-[28rem] w-full rounded-xl border border-surface-border overflow-hidden bg-midnight">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        fitView
        fitViewOptions={{ padding: 0.25 }}
        panOnDrag
        zoomOnScroll
        zoomOnPinch
        zoomOnDoubleClick={false}
        preventScrolling={false}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable
        minZoom={0.3}
        maxZoom={1.5}
        onNodeClick={(_, node) => {
          if (agentOrder.includes(node.id as DebateAgent)) {
            onSelectAgent(selectedAgent === node.id ? null : (node.id as DebateAgent));
          }
        }}
        onPaneClick={() => onSelectAgent(null)}
      >
        <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="#3e2050" />
      </ReactFlow>
    </div>
  );
}
