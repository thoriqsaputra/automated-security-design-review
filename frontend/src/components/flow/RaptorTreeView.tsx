import { useEffect, useMemo, useState } from 'react';
import dagre from '@dagrejs/dagre';
import { Background, Controls, MiniMap, ReactFlow, type Edge, type Node } from '@xyflow/react';
import { Network } from 'lucide-react';
import Card from '../ui/Card';
import type { RaptorNodeSnapshot, RaptorSnapshot } from '../../api/reviews';

interface Props {
  snapshot: RaptorSnapshot;
}

const DEFAULT_LIMIT = 25;

function clampLimit(totalNodes: number, current: number) {
  const max = Math.min(Math.max(totalNodes, 10), 200);
  return Math.max(10, Math.min(current, max));
}

function breadthFirstVisibleNodes(snapshot: RaptorSnapshot, limit: number) {
  const nodesById = new Map(snapshot.nodes.map((node) => [node.id, node]));
  const childrenByParent = new Map<string, RaptorNodeSnapshot[]>();
  let rootId = snapshot.root_node_id;

  for (const node of snapshot.nodes) {
    if (!node.parent_id && !rootId) {
      rootId = node.id;
    }
    if (!node.parent_id) continue;
    const siblings = childrenByParent.get(node.parent_id) || [];
    siblings.push(node);
    childrenByParent.set(node.parent_id, siblings);
  }

  if (!rootId || !nodesById.has(rootId)) {
    return snapshot.nodes.slice(0, limit);
  }

  const queue = [rootId];
  const visible: RaptorNodeSnapshot[] = [];
  const seen = new Set<string>();

  while (queue.length > 0 && visible.length < limit) {
    const currentId = queue.shift();
    if (!currentId || seen.has(currentId)) continue;
    seen.add(currentId);
    const node = nodesById.get(currentId);
    if (!node) continue;
    visible.push(node);
    const children = childrenByParent.get(currentId) || [];
    for (const child of children) {
      queue.push(child.id);
    }
  }

  return visible;
}

export default function RaptorTreeView({ snapshot }: Props) {
  const maxVisible = Math.min(Math.max(snapshot.total_nodes, 10), 200);
  const [visibleLimit, setVisibleLimit] = useState(clampLimit(snapshot.total_nodes, DEFAULT_LIMIT));
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(snapshot.root_node_id);

  useEffect(() => {
    setVisibleLimit(clampLimit(snapshot.total_nodes, DEFAULT_LIMIT));
  }, [snapshot.total_nodes]);

  const visibleSnapshotNodes = useMemo(
    () => breadthFirstVisibleNodes(snapshot, visibleLimit),
    [snapshot, visibleLimit],
  );

  const visibleNodeIds = useMemo(
    () => new Set(visibleSnapshotNodes.map((node) => node.id)),
    [visibleSnapshotNodes],
  );

  const { nodes, edges } = useMemo(() => {
    const graph = new dagre.graphlib.Graph();
    graph.setGraph({ rankdir: 'TB', ranksep: 72, nodesep: 24, marginx: 16, marginy: 16 });
    graph.setDefaultEdgeLabel(() => ({}));

    const flowNodes: Node[] = visibleSnapshotNodes.map((node) => {
      graph.setNode(node.id, { width: 260, height: 92 });
      return {
        id: node.id,
        position: { x: 0, y: 0 },
        data: {
          label: (
            <div className="space-y-1">
              <div className="flex items-center justify-between gap-2 text-[10px] uppercase tracking-[0.24em] text-text-muted">
                <span>Level {node.level}</span>
                <span>{node.child_count} child{node.child_count === 1 ? '' : 'ren'}</span>
              </div>
              <div className="text-sm font-semibold text-text-primary line-clamp-2">
                {node.section_heading || node.id}
              </div>
              <div className="text-xs text-text-secondary line-clamp-2">
                {node.text_preview || 'No preview available.'}
              </div>
            </div>
          ),
        },
        style: {
          width: 260,
          borderRadius: 16,
          border: '1px solid rgba(240, 89, 65, 0.22)',
          background: 'rgba(42, 20, 53, 0.95)',
          color: 'var(--color-text-primary)',
          boxShadow: '0 16px 32px rgba(0, 0, 0, 0.18)',
        },
      };
    });

    const flowEdges: Edge[] = snapshot.nodes
      .filter((node) => node.parent_id && visibleNodeIds.has(node.id) && visibleNodeIds.has(node.parent_id))
      .map((node) => {
        graph.setEdge(node.parent_id as string, node.id);
        return {
          id: `${node.parent_id}->${node.id}`,
          source: node.parent_id as string,
          target: node.id,
          type: 'smoothstep',
          animated: false,
          style: { stroke: 'rgba(190, 49, 68, 0.5)', strokeWidth: 1.4 },
        };
      });

    dagre.layout(graph);

    for (const node of flowNodes) {
      const position = graph.node(node.id);
      node.position = {
        x: position.x - position.width / 2,
        y: position.y - position.height / 2,
      };
    }

    return { nodes: flowNodes, edges: flowEdges };
  }, [snapshot.nodes, visibleNodeIds, visibleSnapshotNodes]);

  useEffect(() => {
    if (!selectedNodeId || visibleNodeIds.has(selectedNodeId)) return;
    setSelectedNodeId(visibleSnapshotNodes[0]?.id || null);
  }, [selectedNodeId, visibleNodeIds, visibleSnapshotNodes]);

  const selectedNode = visibleSnapshotNodes.find((node) => node.id === selectedNodeId) || visibleSnapshotNodes[0] || null;

  return (
    <div className="space-y-4">
      <Card className="space-y-4">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="flex items-start gap-3">
            <Network size={18} className="mt-0.5 shrink-0 text-flame" />
            <div className="space-y-1">
              <p className="text-sm font-semibold text-text-primary">
                RAPTOR status: {snapshot.status || 'unknown'}
              </p>
              <p className="text-xs text-text-muted">
                {snapshot.total_nodes} node(s) across {snapshot.max_level + 1} levels
              </p>
            </div>
          </div>

          <label className="flex w-full max-w-sm flex-col gap-1 text-xs text-text-muted">
            <span>Visible nodes: {visibleLimit}</span>
            <input
              type="range"
              min={10}
              max={maxVisible}
              value={visibleLimit}
              onChange={(event) => setVisibleLimit(Number(event.target.value))}
              className="accent-flame"
            />
          </label>
        </div>
      </Card>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
        <Card className="p-0 overflow-hidden">
          <div className="h-[540px] w-full">
            <ReactFlow
              key={`${visibleLimit}-${nodes.length}`}
              nodes={nodes}
              edges={edges}
              fitView
              onNodeClick={(_, node) => setSelectedNodeId(node.id)}
              nodesDraggable={false}
              nodesConnectable={false}
              elementsSelectable
              proOptions={{ hideAttribution: true }}
            >
              <MiniMap pannable zoomable />
              <Controls showInteractive={false} />
              <Background color="rgba(158, 58, 86, 0.22)" gap={18} />
            </ReactFlow>
          </div>
        </Card>

        <Card className="space-y-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-text-muted">Node Detail</p>
            <p className="mt-1 text-sm font-semibold text-text-primary line-clamp-1">
              {selectedNode?.section_heading || selectedNode?.id || 'No node selected'}
            </p>
          </div>

          {selectedNode ? (
            <>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="rounded-lg border border-surface-border bg-midnight p-3">
                  <p className="text-text-muted">Level</p>
                  <p className="mt-1 text-text-primary">{selectedNode.level}</p>
                </div>
                <div className="rounded-lg border border-surface-border bg-midnight p-3">
                  <p className="text-text-muted">Source blocks</p>
                  <p className="mt-1 text-text-primary">{selectedNode.source_block_count}</p>
                </div>
              </div>

              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-text-muted">Pages</p>
                <p className="mt-1 text-sm text-text-secondary">
                  {selectedNode.page_numbers.length > 0 ? selectedNode.page_numbers.join(', ') : 'No page references'}
                </p>
              </div>

              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-text-muted">Preview</p>
                <p className="mt-1 text-sm leading-relaxed text-text-secondary">
                  {selectedNode.text_preview || 'No preview available.'}
                </p>
              </div>
            </>
          ) : (
            <p className="text-sm text-text-muted">No visible RAPTOR nodes.</p>
          )}
        </Card>
      </div>
    </div>
  );
}
