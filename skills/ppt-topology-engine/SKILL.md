# SKILL: ppt-topology-engine

## Description
Topology engine for PPT HTML slides. Used exclusively for complex flowcharts, system architectures, cross-functional bands, and path-routing diagrams. This skill handles crossing logic, node hierarchy, and custom graphic boundaries mapping.

## 1. Engine Boundaries & Import Standard
- **When NOT to use**: Do not use for statistical charting (use `ppt-chart-engine` instead) or simple linear step cards (use basic HTML/Tailwind flex).
- **Core Engine**: ALWAYS use AntV X6 **v1.34.14 (UMD)** for independent HTML slide rendering. Do NOT use v2.x ES Modules which break under file:// or simple script tag execution.
```html
<script src="https://unpkg.com/@antv/x6@1.34.14/dist/x6.js"></script>
```

## 2. The HTML Node Paradigm (Critical)
Never use standard SVG `<text>` elements for labels inside nodes, as they fail at multiline auto-wrapping, overflow control, and centering.
Always enforce the HTML Node pattern using `inherit: 'html'` so you can use Tailwind CSS inside X6 nodes.

**Registration Example:**
```javascript
// Always prefix with X6. space since UMD exposes global X6.
X6.Graph.registerNode('html-node', {
  inherit: 'html',
  width: 160,
  height: 60,
  ports: {
    groups: {
      left: { position: 'left' },
      right: { position: 'right' },
      top: { position: 'top' },
      bottom: { position: 'bottom' }
    }
  },
  effect: ['data'],
  html: (cell) => {
    const data = cell.getData();
    // Use Tailwind CSS classes for reliable styling
    const wrap = document.createElement('div');
    wrap.style.width = '100%';
    wrap.style.height = '100%';
    wrap.innerHTML = `
      <div class="w-full h-full flex items-center justify-center p-2 rounded-md shadow-sm border border-slate-300 bg-white" 
           style="box-sizing: border-box;">
        <span class="text-xs text-center text-slate-800 leading-tight">${data.label}</span>
      </div>
    `;
    return wrap;
  },
});
```

## 3. Orthogonal Routing Standard
Edges must look industrial and systematic. Avoid diagonal lines across the canvas.
- Always prefer `orth` routing with step options to avoid node overlap and enforce proper obstacles calculation.
- Add rounded corners to make intersections look modern.

**Global Options Setup:**
```javascript
const graph = new X6.Graph({
  container: document.getElementById('container'),
  autoResize: true,
  connecting: {
    router: {
      name: 'orth', // better than orth as it avoids obstacles
      args: { padding: 15, startDirections: ['right', 'bottom', 'top'], endDirections: ['left', 'top', 'bottom'] },
    },
    connector: {
      name: 'rounded',
      args: { radius: 8 },
    },
    defaultEdge: {
      attrs: {
        line: {
          stroke: '#64748b',
          strokeWidth: 2,
          targetMarker: {
            name: 'block',
            width: 8,
            height: 8,
          },
        },
      },
    }
  }
});
```

**Edge Connection Using Ports and Routing Traps (Crucial):**
To ensure lines explicitly enter/exit from correct sides and don't tangle through nodes or overlapping regions:
1. **Semantic Port Assignments**: ALWAYS specify specific ports based on the ACTUAL geometric relationships and business flow (e.g. `top`, `bottom`, `left`, `right`). Do not blindly assign every connection as `right` -> `left`. For example, Data APIs might talk to Merge APIs above/below them (requiring `top`/`bottom`).
2. **Dense Obstacles and Node Penetration**: 
   - Do not let Manhattan routing guess the obstacle boundaries blindly. 
   - **Never let a line cross through a node.** If Manhattan routing fails because nodes are too close, or if it crosses an unrelated vertical node, you MUST use explicit waypoints `vertices: [{x: 100, y: 200}]` to route the edge safely around the obstacle.
   - **Node Penetration Fix**: When using HTML nodes, always give the node a solid background and set `zIndex: 10` (higher than edges). Ensure ports are rigidly defined at borders so links don't pierce into the node's visual center.
```javascript
{ 
  source: { cell: 'node-A', port: 'right' }, // context-dependent! Use top/bottom/left/right logically
  target: { cell: 'node-B', port: 'left' } 
}
```

## 4. Standard Assets (Shape Catalog)
When architectural components require specific visual metaphors:

### 4.1 Database (Cylinder)
Never use pure SVG paths for databases if you need multiline text and port connections. Use `inherit: 'html'` with CSS background shapes, or register a robust HTML node:

```javascript
X6.Graph.registerNode('arch-database', {
  inherit: 'html',
  width: 100,
  height: 90,
  ports: {
    groups: {
      left: { position: 'left' }, right: { position: 'right' },
      top: { position: 'top' }, bottom: { position: 'bottom' }
    }
  },
  effect: ['data'],
  html: (cell) => {
    const data = cell.getData();
    // Use an inline SVG for the cylinder background, and an absolute div for perfect text centering.
    const wrap = document.createElement('div');
    wrap.style.width = '100%';
    wrap.style.height = '100%';
    wrap.innerHTML = `
      <div style="position: relative; width: 100%; height: 100%;">
        <svg viewBox="0 0 100 110" style="width: 100%; height: 100%; position: absolute; z-index: 0;">
          <path d="M 0,20 A 50,20 0 1,1 100,20 L 100,90 A 50,20 0 1,1 0,90 Z" fill="#e0f2fe" stroke="#0ea5e9" stroke-width="2"/>
        </svg>
        <div class="absolute inset-0 z-10 flex items-center justify-center p-2 text-center text-xs font-bold text-sky-900 leading-tight">
          ${data.label}
        </div>
      </div>
    `;
    return wrap;
  }
});
```

### 4.2 Application / Container / Group Boundaries
Use Parent nodes when enclosing sub-components. Set `zIndex` carefully.
```javascript
const groupNode = graph.addNode({
  x: 10, y: 10, width: 400, height: 300,
  zIndex: 1,
  attrs: {
    body: { fill: '#f8fafc', stroke: '#cbd5e1', strokeDasharray: '5,5' },
    label: { text: 'Core System', refY: 15, textAnchor: 'middle' }
  }
});
const childNode = graph.addNode({ shape: 'html-node', x: 50, y: 50 });
groupNode.addChild(childNode);
```

## 6. Execution Pipeline
1. **Understand Logic:** Parse the prompt into a logical graph (Nodes/Groups, Edges).
2. **Setup X6:** Inject Script Tag -> Init Container -> Init Graph.
3. **Register Assets:** Register required `html-node` shapes and custom paths. Be sure to use `X6.Graph.registerNode`.
4. **Coordinate Planning:** Plan the `x` and `y` coordinates meticulously on a grid system (e.g., column 1 at x=100, column 2 at x=350, etc.). Prevent overlapping layout structures.
5. **Add Ports & Edges:** Ensure nodes define port groups. Use specific `{ cell: 'id', port: 'right' }` targets to avoid chaotic routing.
6. **Render:** Add nodes manually or via `graph.addNodes()` and `graph.addEdges()`.

## 7. Edge Labels & Text Formatting (Critical)
- **Multiline Edge Labels**: Never use literal double-escaped strings like `'Tag\\n[T+2]'` inside `labels`. X6 standard text parses true `\n` to `<tspan>`. Pass actual multiline strings using template literals or `\n` explicitly, omitting arbitrary backslashes.
- **Node Text Formatting**: For `html` nodes, always use `<br>` to split text, NOT `\n`.
- **Label Obstacle Avoidance**: Labels default to the visual center of edges, which often causes text to overlap the line or cross nodes. **Always offset edge labels** using `position: { distance: 0.5, offset: -20 }` (adjust positive/negative) to float the text cleanly above or below the line segment.
