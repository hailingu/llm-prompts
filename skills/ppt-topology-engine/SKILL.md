# SKILL: ppt-topology-engine

## Description
Topology engine for PPT HTML slides. Used exclusively for complex flowcharts, system architectures, cross-functional bands, and path-routing diagrams. This skill handles crossing logic, node hierarchy, and custom graphic boundaries mapping.

## 1. Engine & Dependencies
- **Core Engine File**: See `assets/x6-boilerplate.html` for complete initialization and import bindings (`AntV X6 v1.34.14 (UMD)`).
- **Rule**: NEVER use standard SVG `<text>` elements for UI-rich labels. Use the `inherit: 'html'` paradigm. An example of `html-node` and custom `arch-database` is provided in `assets/x6-boilerplate.html`.

## 2. Orthogonal Routing & Edges Standard
Edges must look industrial and systematic. Avoid chaotic spaghetti lines, excessive overlaps, and diagonal lines.

- **Routing Type**: 
  - Prefer `orth` or `manhattan` routing with rounded connectors.
  - **CRITICAL - Router Obstacles & Bounding Boxes**: X6 Auto-routing (especially `orth` and `manhattan`) treats ALL nodes—including large background groups—as physical obstacles. This inevitably causes severe zig-zagging as lines try to navigate "around" them instead of through them.
  - **MANDATORY FIX FOR GROUPS**: For edges that cross into or out of large group boxes, you MUST disable auto-routing on that specific edge using `router: { name: 'normal' }` (or use `er` router) and supply your own exact `vertices` to form the path. Otherwise, paths will look chaotic and broken.
  - **PERFECT RIGHT ANGLES (CRITICAL)**: When using `normal` router and manual `vertices`, the x/y coordinates of the vertices **MUST exactly align** with the target port's absolute center coordinate or the next vertex's coordinate. If the last vertex's `y` is 170 but the target node's port is at `y=150`, the final segment will be an ugly diagonal line. Calculate exact centers to ensure 100% horizontal/vertical lines.
- **Port Semantics (Crucial)**: ALWAYS specify exact entry/exit ports (`top`, `bottom`, `left`, `right`). Do not blindly assign every connection as `right` -> `left`. Route logically based on relative geographic coordinates.
- **Crossings & Overlaps Defense**:
  - Do NOT let auto-routing guess paths if the target is deeply enclosed in subgroups, or crossing multiple large sections. 
  - NEVER let a line cross completely through an unrelated Node.
  - **MANDATORY**: You MUST actively declare `vertices: [{x: 100, y: 200}, {x: 100, y: 300}]` waypoints for ANY long-distance edges or complex bus flows. This prevents messy auto-generated detours.
  - Separate multiple edges routing between the same areas by offsetting their `vertices` so they do not overlap perfectly and form a single thick unreadable line.
- **Labels (Avoid Overlaps)**:
  - Labels by default snap to the exact center of the edge, often obstructing lines or nodes. 
  - **Always offset edge labels** using `position: { distance: 0.5, offset: { x: 0, y: -20 } }` (adjust as needed contextually). If the edge segment is vertical, use an X offset (e.g., `{ x: 25, y: 0 }`); if horizontal, use a Y offset.
  - Do not place nodes and multi-line labels too close together. Leave plenty of breathing room.
  - Use `\n` appropriately for native multiline X6 edge labels instead of raw double escapes. (For `<br>` inside HTML Nodes).

## 3. String Interpolation Safety Mechanism (CRITICAL)
When building HTML Nodes that contain vanilla Javascript inline strings, Agent code generators often accidentally trigger template literal escaping bugs (e.g. producing `\${data.label}` instead of executing it).
- **Rule**: NEVER use ES6 Template Literals (`${var}`) for DOM string assignments inside generic render cycles if there's any ambiguity.
- **Implementation**: STRICTLY use old-school concatenation `"<div class='w-full'> " + (data?.label || '') + " </div>";` as shown in the boilerplate `assets/x6-boilerplate.html` to avert the escaping trap.

## 4. Execution Pipeline
1. **Understand Logic:** Parse the prompt into a Node and Edge logical matrix.
2. **Boilerplate Merge:** Read `assets/x6-boilerplate.html` using the read file tool for reference to get the accurate node registration logic, and inject the boilerplate into your final PPT canvas container.
3. **Coordinate Planning:** Plan `x` and `y` matrix on a defined grid system (`COL_WIDTH`, `ROW_HEIGHT`, `Margins`) manually. Prevent overlapping layouts explicitly via calculation.
4. **Render Check:** Load nodes into `graph.addNodes()` and specific semantic edges into `graph.addEdges()`. Verify no visual collapses occur.
