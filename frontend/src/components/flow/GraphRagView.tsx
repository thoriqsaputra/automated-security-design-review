import { useEffect, useMemo, useRef, useState } from 'react';
import * as d3 from 'd3';
import Card from '../ui/Card';
import type { GraphEdgeSnapshot, GraphNodeSnapshot, GraphSnapshot } from '../../api/reviews';

interface Props {
  snapshot: GraphSnapshot;
}

const DEFAULT_LIMIT = 30;
const ENTITY_COLORS: Record<string, string> = {
  service: '#F05941',
  api: '#BE3144',
  database: '#7cc9ff',
  auth_mechanism: '#f8b84b',
  protocol: '#8bdfc8',
  external_system: '#d8a6ff',
  queue: '#88d8b0',
  cache: '#ff9aa2',
  component: '#f0e6f6',
  data_store: '#9fb6ff',
  user: '#ffd166',
};

function clampLimit(total: number, current: number) {
  const max = Math.min(Math.max(total, 10), 200);
  return Math.max(10, Math.min(current, max));
}

function formatPages(pages: number[]) {
  return pages.length > 0 ? pages.join(', ') : 'No page references';
}

type SimNode = GraphNodeSnapshot & {
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
};

type SimLink = GraphEdgeSnapshot & {
  source: string | SimNode;
  target: string | SimNode;
};

function edgeEndpointId(endpoint: string | GraphNodeSnapshot) {
  return typeof endpoint === 'string' ? endpoint : endpoint.id;
}

function edgeId(edge: { source: string | GraphNodeSnapshot; target: string | GraphNodeSnapshot }) {
  return `${edgeEndpointId(edge.source)}->${edgeEndpointId(edge.target)}`;
}

export default function GraphRagView({ snapshot }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [visibleLimit, setVisibleLimit] = useState(clampLimit(snapshot.total_entities, DEFAULT_LIMIT));
  const [dimensions, setDimensions] = useState({ width: 960, height: 540 });
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);

  useEffect(() => {
    setVisibleLimit(clampLimit(snapshot.total_entities, DEFAULT_LIMIT));
  }, [snapshot.total_entities]);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const nextWidth = Math.max(320, Math.floor(entry.contentRect.width));
        setDimensions({ width: nextWidth, height: 540 });
      }
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  const maxVisible = Math.min(Math.max(snapshot.total_entities, 10), 200);

  const visibleData = useMemo(() => {
    const sortedNodes = [...snapshot.nodes].sort((a, b) => {
      if (b.degree !== a.degree) return b.degree - a.degree;
      if (b.source_block_count !== a.source_block_count) return b.source_block_count - a.source_block_count;
      return a.label.localeCompare(b.label);
    });
    const nodes = sortedNodes.slice(0, visibleLimit);
    const visibleIds = new Set(nodes.map((node) => node.id));
    const edges = snapshot.edges.filter(
      (edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target),
    );
    return { nodes, edges };
  }, [snapshot.edges, snapshot.nodes, visibleLimit]);

  useEffect(() => {
    if (selectedNodeId && visibleData.nodes.some((node) => node.id === selectedNodeId)) return;
    setSelectedNodeId(visibleData.nodes[0]?.id || null);
  }, [selectedNodeId, visibleData.nodes]);

  useEffect(() => {
    if (selectedEdgeId && visibleData.edges.some((edge) => edgeId(edge) === selectedEdgeId)) return;
    setSelectedEdgeId(null);
  }, [selectedEdgeId, visibleData.edges]);

  useEffect(() => {
    const svgElement = svgRef.current;
    if (!svgElement) return;

    const svg = d3.select(svgElement);
    svg.selectAll('*').remove();
    svg.attr('viewBox', `0 0 ${dimensions.width} ${dimensions.height}`);

    const root = svg.append('g');
    svg.call(
      d3.zoom<SVGSVGElement, unknown>()
        .scaleExtent([0.35, 2.5])
        .on('zoom', (event: d3.D3ZoomEvent<SVGSVGElement, unknown>) => root.attr('transform', event.transform.toString())),
    );

    const nodes: SimNode[] = visibleData.nodes.map((node) => ({ ...node }));
    const links: SimLink[] = visibleData.edges.map((edge) => ({ ...edge }));

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink<SimNode, SimLink>(links).id((d: SimNode) => d.id).distance(120).strength(0.35))
      .force('charge', d3.forceManyBody().strength(-320))
      .force('center', d3.forceCenter(dimensions.width / 2, dimensions.height / 2))
      .force('collision', d3.forceCollide<SimNode>().radius((d: SimNode) => 18 + Math.min(d.degree, 10) * 1.8));

    const linkGroup = root.append('g').attr('stroke', 'rgba(190, 49, 68, 0.45)').attr('stroke-width', 1.2);
    const nodeGroup = root.append('g');
    const labelGroup = root.append('g');

    const linkSelection = linkGroup
      .selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('stroke-opacity', (d: SimLink) => Math.max(0.2, d.confidence || 0.25))
      .on('click', (_: MouseEvent, edge: SimLink) => {
        setSelectedEdgeId(edgeId(edge));
        setSelectedNodeId(null);
      });

    linkSelection.append('title').text((d: SimLink) => {
      const relation = d.relation_type || 'relation';
      const protocol = d.protocol ? ` via ${d.protocol}` : '';
      return `${relation}${protocol}`;
    });

    const nodeSelection = nodeGroup
      .selectAll('circle')
      .data(nodes)
      .enter()
      .append('circle')
      .attr('r', (d: SimNode) => 12 + Math.min(d.degree, 10) * 1.6)
      .attr('fill', (d: SimNode) => ENTITY_COLORS[d.entity_type] || '#f0e6f6')
      .attr('stroke', 'rgba(255,255,255,0.86)')
      .attr('stroke-width', 1.1)
      .style('cursor', 'pointer')
      .on('click', (_: MouseEvent, node: SimNode) => {
        setSelectedNodeId(node.id);
        setSelectedEdgeId(null);
      })
      .call(
        d3.drag<SVGCircleElement, SimNode>()
          .on('start', (event: d3.D3DragEvent<SVGCircleElement, SimNode, SimNode>, node: SimNode) => {
            if (!event.active) simulation.alphaTarget(0.24).restart();
            node.fx = node.x;
            node.fy = node.y;
          })
          .on('drag', (event: d3.D3DragEvent<SVGCircleElement, SimNode, SimNode>, node: SimNode) => {
            node.fx = event.x;
            node.fy = event.y;
          })
          .on('end', (event: d3.D3DragEvent<SVGCircleElement, SimNode, SimNode>, node: SimNode) => {
            if (!event.active) simulation.alphaTarget(0);
            node.fx = null;
            node.fy = null;
          }),
      );

    nodeSelection.append('title').text((d: SimNode) => `${d.label} (${d.entity_type})`);

    const labelSelection = labelGroup
      .selectAll('text')
      .data(nodes)
      .enter()
      .append('text')
      .attr('fill', 'rgba(240, 230, 246, 0.92)')
      .attr('font-size', 11)
      .attr('font-weight', 600)
      .attr('text-anchor', 'middle')
      .attr('dy', 28)
      .text((d: SimNode) => d.label.length > 18 ? `${d.label.slice(0, 18)}...` : d.label);

    simulation.on('tick', () => {
      linkSelection
        .attr('x1', (d: SimLink) => (d.source as SimNode).x || 0)
        .attr('y1', (d: SimLink) => (d.source as SimNode).y || 0)
        .attr('x2', (d: SimLink) => (d.target as SimNode).x || 0)
        .attr('y2', (d: SimLink) => (d.target as SimNode).y || 0);

      nodeSelection
        .attr('cx', (d: SimNode) => d.x || 0)
        .attr('cy', (d: SimNode) => d.y || 0);

      labelSelection
        .attr('x', (d: SimNode) => d.x || 0)
        .attr('y', (d: SimNode) => d.y || 0);
    });

    return () => {
      simulation.stop();
    };
  }, [dimensions.height, dimensions.width, visibleData.edges, visibleData.nodes]);

  const selectedNode = visibleData.nodes.find((node) => node.id === selectedNodeId) || null;
  const selectedEdge = visibleData.edges.find((edge) => edgeId(edge) === selectedEdgeId) || null;

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="text-sm font-semibold text-text-primary">GraphRAG Network</p>
          <p className="text-xs text-text-muted">
            {snapshot.total_entities} entities · {snapshot.total_relations} relations
          </p>
        </div>
        <label className="flex min-w-72 flex-col gap-1 text-xs text-text-muted">
          <span>Visible entities: {visibleLimit}</span>
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

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_320px]">
        <Card className="p-0 overflow-hidden">
          <div ref={containerRef} className="h-[540px] w-full bg-[radial-gradient(circle_at_top,rgba(240,89,65,0.10),transparent_40%),linear-gradient(180deg,rgba(26,11,34,0.98),rgba(34,9,44,0.98))]">
            <svg ref={svgRef} className="h-full w-full" />
          </div>
        </Card>

        <Card className="space-y-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.24em] text-text-muted">Selection Detail</p>
            <p className="mt-1 text-sm font-semibold text-text-primary">
              {selectedNode ? selectedNode.label : selectedEdge ? selectedEdge.relation_type || 'Relation' : 'No selection'}
            </p>
          </div>

          {selectedNode && (
            <>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="rounded-lg border border-surface-border bg-midnight p-3">
                  <p className="text-text-muted">Type</p>
                  <p className="mt-1 text-text-primary">{selectedNode.entity_type}</p>
                </div>
                <div className="rounded-lg border border-surface-border bg-midnight p-3">
                  <p className="text-text-muted">Degree</p>
                  <p className="mt-1 text-text-primary">{selectedNode.degree}</p>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-3 text-xs">
                <div className="rounded-lg border border-surface-border bg-midnight p-3">
                  <p className="text-text-muted">In</p>
                  <p className="mt-1 text-text-primary">{selectedNode.in_degree}</p>
                </div>
                <div className="rounded-lg border border-surface-border bg-midnight p-3">
                  <p className="text-text-muted">Out</p>
                  <p className="mt-1 text-text-primary">{selectedNode.out_degree}</p>
                </div>
                <div className="rounded-lg border border-surface-border bg-midnight p-3">
                  <p className="text-text-muted">Blocks</p>
                  <p className="mt-1 text-text-primary">{selectedNode.source_block_count}</p>
                </div>
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-text-muted">Pages</p>
                <p className="mt-1 text-sm text-text-secondary">{formatPages(selectedNode.source_pages)}</p>
              </div>
            </>
          )}

          {!selectedNode && selectedEdge && (
            <>
              <div className="rounded-lg border border-surface-border bg-midnight p-3 text-sm text-text-secondary">
                <p><span className="text-text-muted">Source:</span> {selectedEdge.source}</p>
                <p className="mt-1"><span className="text-text-muted">Target:</span> {selectedEdge.target}</p>
                <p className="mt-1"><span className="text-text-muted">Relation:</span> {selectedEdge.relation_type || 'Unknown'}</p>
                <p className="mt-1"><span className="text-text-muted">Confidence:</span> {selectedEdge.confidence.toFixed(2)}</p>
              </div>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div className="rounded-lg border border-surface-border bg-midnight p-3">
                  <p className="text-text-muted">Protocol</p>
                  <p className="mt-1 text-text-primary">{selectedEdge.protocol || '—'}</p>
                </div>
                <div className="rounded-lg border border-surface-border bg-midnight p-3">
                  <p className="text-text-muted">Encrypted</p>
                  <p className="mt-1 text-text-primary">
                    {selectedEdge.is_encrypted === null ? 'Unknown' : selectedEdge.is_encrypted ? 'Yes' : 'No'}
                  </p>
                </div>
              </div>
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.24em] text-text-muted">Auth requirement</p>
                <p className="mt-1 text-sm text-text-secondary">
                  {selectedEdge.requires_auth === null ? 'Unknown' : selectedEdge.requires_auth ? 'Required' : 'Not required'}
                </p>
              </div>
            </>
          )}

          {!selectedNode && !selectedEdge && (
            <p className="text-sm text-text-muted">Click an entity or relation to inspect it.</p>
          )}
        </Card>
      </div>
    </div>
  );
}
