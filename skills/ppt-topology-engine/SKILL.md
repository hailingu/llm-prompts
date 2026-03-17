# SKILL: ppt-topology-engine

## Description
Topology engine for PPT HTML slides. Used exclusively for complex flowcharts, system architectures, cross-functional bands, and path-routing diagrams. This skill handles crossing logic, node hierarchy, and custom graphic boundaries mapping.

## 1. Engine & Dependencies
- **Core Engine File**: Use the stable CDN link in your HTML: `<script src="https://unpkg.com/@antv/x6/dist/x6.min.js"></script>`.
- **Rule**: NEVER use standard SVG `<text>` elements for UI-rich labels. Use the `inherit: 'html'` paradigm.
- **CRITICAL SHAPE REGISTRATION**: Every `shape` string passed to `graph.addNode({ shape: 'my-shape' })` **MUST** be explicitly registered. 
  - **Syntax Rule (CRITICAL)**: Use the modern `Shape.HTML.register(...)` method. For example: `const { Graph, Shape } = X6; Shape.HTML.register({ shape: 'my-shape', width: 120, height: 60, html(cell) { ... } });`. If you pass an unregistered shape, the entire X6 execution will throw a fatal JavaScript exception and **halt rendering completely**, resulting in a completely blank canvas. Validate every shape name!

## 2. Mathematical Layout & Bounding Box Extractor
Before writing any HTML, you MUST generate a `thinking.md` document that acts as your spatial mapping and measurement step.
- **Unit Grid System**: Do not guess random pixel layouts. First, divide the canvas into an abstract grid (e.g. 1 Unit = 100px).
- **Spanning & Bridging**: Explicitly declare nodes that cross between rows or columns (e.g., a "Storage" block that runs the entire height of the left side).
- **Visual Bounding Box Matrix**: You MUST output a structured table in your `thinking.md` documenting the Width (U), Height (U), Logic-X, and Logic-Y for every macro element. Relying only on text indentation and list summaries is forbidden. Use the `knowledge/templates/ppt-topology-thinking-template.md` contract.

## 3. Diagram Typology: Block Architecture vs. Flowchart
When converting a source diagram, determine its fundamental type before writing logic:
- **Block Architecture Diagrams**: These are structural zoning diagrams (like cloud architectures, platform capabilities, component matrices) built heavily on grouped boxes, sub-modules, and geographic regions. *Characteristics*: Lots of nested boxes, grid alignments, icons, text, but almost **NO explicitly drawn flow lines**. 
  - **ACTION**: DO NOT invent edges/flowlines if the source material mainly displays structural boxes. Use purely spatial grouping and nested coordinates. Remove connection logic.
  - **ZERO EDGES POLICY**: For Block Architecture diagrams, you MUST NOT draw any edges or flowlines. Do not create an `edges` array. Do not call `graph.addEdges`. Agents frequently hallucinate data flows ("Data Source -> Data Ingest") because they assume architectures must have lines. **IF THE SOURCE IMAGE DOES NOT HAVE EXPLICIT ARROWS, DO NOT ADD THEM.** Rely purely on spatial positioning and group boundaries.
  - **INVISIBLE PORTS**: Always configure port groups to be visually invisible unless interactive wires are explicitly requested. Add `attrs: { circle: { r: 0, magnet: true, stroke: 'transparent', fill: 'transparent' } }` to port definitions so empty port circles are hidden.
  - **High-Fidelity Detail Extraction (NO SHORTCUTS)**: Do not lazily summarize or drop sub-nodes. If a zoned group in the original diagram contains 8 micro-service blocks, you must plot all 8 individually using a properly scaled local grid. Include secondary texts, descriptive subtypes (e.g., "(Metadata Driven)"), and background logos as HTML elements inside nodes if they exist in the source.

## 3. Strict Structural and Sub-Node Fidelity (MANDATORY)
When converting a diagram, your primary goal is identical visual replication of the content structure.
- **Deep Nesting Fidelity (NO FLATTENING)**: Analyze the exact depth of the original group boundaries. If a diagram has a 3-level structure (e.g., L1: `Data Platform Core` -> L2: `Data Warehouse` -> L3: `DWD/DWS` nodes), you MUST create nested nodes for ALL 3 levels. Do NOT flatten L3 nodes directly under L1. Create the L2 group boxes with `parent.addChild()` and attach L3 components to L2.
- **Node Quantity & Completeness**: Count the exact number of nodes, sub-nodes, and distinct modules in the original image. Your generated HTML **MUST** contain the exact same number of blocks. Do not flatten 4 distinct service icons into 1 text box. If "Network" has 4 sub-icons under it, you must either create 4 sub-nodes or accurately recreate the 4 icons within a structured HTML layout inside the node.
- **Layout Style & Aspect Ratio**: Respect the visual weight and relative proportions of the original diagram. 
  - If a group node spans across the width of 3 other nodes beneath it, calculate its `width` to visually align with those 3 nodes.
  - Replicate multi-column vs single-row groupings correctly. If a parent box holds a grid of 2x2 children, configure your `x`, `y`, `width`, `height` logic to perfectly match that 2x2 spatial arrangement.
- **Styling Consistency**: Replicate the exact visual hierarchy conventions in the source. If the source uses colored headers for outer groups (e.g., grey for Data Source, light blue for Platform), map those to Tailwind classes. Use dashed lines (`strokeDasharray: '5,5'`) if the source group uses dashed borders.

## 4. Orthogonal Routing & Edges Standard
Edges must look industrial and systematic. Avoid chaotic spaghetti lines, excessive overlaps, and diagonal lines.

- **Strict Edge Fidelity (NO HALLUCINATIONS)**: NEVER invent, assume, or guess connections that are not explicitly drawn in the source image or explicitly requested. If a node or group does not have a visible line connected to it, **do not draw one**. Connecting unrelated modules ("because they logically might communicate") is strictly forbidden. Wait for explicit visual proof or textual instruction before adding an edge.
- **Routing Type**: 
  - Prefer `orth` or `manhattan` routing with rounded connectors. Set `padding: 20` to keep lines away from node borders.
  - **Cross-Group Collapse Preventions**: Standard `orth` or `manhattan` routers often completely fail or produce chaotic "spaghetti" lines when crossing into or out of deeply nested backgrounds. 
  - **MANDATORY FIX FOR GROUPS**: For edges that cross into or out of large group boxes:
    1. You MUST set the edge `zIndex: 20` to ensure it renders above the group background.
    2. You MUST disable auto-obstacle routing on that specific edge. Use `router: { name: 'er', args: { direction: 'H' } }` for an automated orthogonal flow, OR use `router: { name: 'normal' }` and supply your own exact `vertices` to form the path. Otherwise, paths will look chaotic or vanish completely.
  - **PERFECT RIGHT ANGLES (CRITICAL)**: When using `normal` router and manual `vertices`, the x/y coordinates of the vertices **MUST exactly align** with the target port's absolute center coordinate or the next vertex's coordinate. If the last vertex's `y` is 170 but the target node's port is at `y=150`, the final segment will be an ugly diagonal line. Calculate exact centers to ensure 100% horizontal/vertical lines.
- **Bus Corridors (Lanes)**: Reserve empty space (e.g., 20-40px wide) between columns and rows explicitly for routing lines. Do not pack nodes so tightly that edges have nowhere to pass without overlapping.
- **Port Semantics (Crucial)**: ALWAYS specify exact entry/exit ports (`top`, `bottom`, `left`, `right`). Do not blindly assign every connection as `right` -> `left`. Route logically based on relative geographic coordinates (e.g., node directly below should connect `top` to `bottom`).
- **Crossings & Overlaps Defense**:
  - Do NOT let auto-routing guess paths if the target is deeply enclosed in subgroups, or crossing multiple large sections. 
  - NEVER let a line cross completely through an unrelated Node. Use waypoints (`vertices`) to route the edge around obstacles if auto-routing fails.
  - **MANDATORY**: You MUST actively declare `vertices: [{x: 100, y: 200}, {x: 100, y: 300}]` waypoints for ANY long-distance edges or complex bus flows. This prevents messy auto-generated detours.
  - Separate multiple edges routing between the same areas by offsetting their `vertices` so they do not overlap perfectly and form a single thick unreadable line.
- **Labels (Avoid Overlaps)**:
  - Labels by default snap to the exact center of the edge, often obstructing lines or nodes. 
  - **Always offset edge labels** using `position: { distance: 0.5, offset: { x: 0, y: -20 } }` (adjust as needed contextually). If the edge segment is vertical, use an X offset (e.g., `{ x: 25, y: 0 }`); if horizontal, use a Y offset.
  - Do not place nodes and multi-line labels too close together. Leave plenty of breathing room.
  - Use `\n` appropriately for native multiline X6 edge labels instead of raw double escapes. (For `<br>` inside HTML Nodes).

## 5. String Interpolation Safety Mechanism (CRITICAL)
When building HTML Nodes that contain vanilla Javascript inline strings, Agent code generators often accidentally trigger template literal escaping bugs (e.g. producing `\${data.label}` instead of executing it).
- **Rule**: NEVER use ES6 Template Literals (`${var}`) for DOM string assignments inside generic render cycles if there's any ambiguity.
- **Implementation**: STRICTLY use old-school concatenation `"<div class='w-full'> " + (data?.label || '') + " </div>";` as shown in the boilerplate `assets/x6-boilerplate.html` to avert the escaping trap.

## 6. Layout Constraints & Node Positioning
- **Z-Index Layering (CRITICAL for Visibility)**: Elements must be strictly layered so lines and text are not hidden behind group backgrounds:
  - `zIndex: 1`: Large Group Nodes (background boxes).
  - `zIndex: 10`: Standard functional Nodes (children and standalone).
  - `zIndex: 20`: **Edges and Lines** (Crucial! If you don't put edges on a higher z-index than group nodes, lines going into groups will disappear behind the group's background fill).
- **Group Containment (Padding) & Auto-Sizing**: If a Node is inside a Group, carefully calculate its absolute `x` and `y` coordinates to ensure it sits comfortably inside the group boundaries. 
  - **Visual Styling for Groups (Light Theme Protocol)**: NEVER use dark/black fills for macro group boundaries unless specifically requested. Group nodes must use a very light, soft background tint (e.g., `#f0f9ff`, `#f8fafc`, or a `color + '1A'` low-opacity trick) with a solid colored border (`stroke: '#0ea5e9'`, `strokeWidth: 1.5`, `rx: 4`, `ry: 4`). Ensure the group label uses a matching semantic color and sits clearly at the top. Group background nodes must not overwhelm the white child nodes.
  - **Dynamic Group Bounding (Anti-Overlap) Principles**: NEVER handcode `width` and `height` for Group/Background boxes via visual guessing if it risks overlap. Either calculate group boundaries dynamically by following the design principles below or carefully map them to a rigid grid:
    1. **Strict Hierarchy Nesting**: For X6 to treat a group properly, you MUST explicitly assign children. Either pass `parent: parentId` when adding a child, or call `parent.addChild(childNode)`. This ensures panning and dragging affect the whole group.
    2. **Apply Semantic Padding**: Allocate generous header padding at the top specifically for the group's title text (e.g., leaving 30-40px at the top empty), ensuring titles never overlap the nested nodes. Implement standard spatial padding for left, right, and bottom.
    3. **Layer Backgrounds**: Always instantiate the macro group box with a securely lower `zIndex` (e.g., `1`) so child nodes (e.g. `zIndex: 10`) sit cleanly on top and are not obscured.
  - Avoid putting nodes directly on the boundary lines (`x=group.x` or `x=group.x + group.width`).
- **Explicit Node Dimensions (Anti-Stretching Bug)**: **CRITICAL**: Always provide explicit `width` and `height` attributes when invoking `graph.addNode({ width: 140, height: 60... })`, even for `html-node`s. Failure to pass explicit sizes will cause nodes to stretch drastically (vertical/horizontal visual collapse distortion) depending on the flex container context.
- **Strict Row/Col Alignment (CENTERING PROTOCOL)**: Nodes in the same logical row or column MUST share perfectly aligned centers. 
  - Do NOT simply share the same `x` coordinate if nodes have different widths. To center nodes of variable widths in a column, compute `x = colCenter - (node.width / 2)`. To align rows, use `y = rowCenter - (node.height / 2)`. Avoid arbitrary `+/- 5px` tweaks. Align mathematical centers.
- **Node Spacing & Collision**: Never allow nodes to overlap. Maintain strictly uniform spacing (e.g., `gapX: 20px`, `gapY: 30px`) between siblings. 
  - Standardize node sizes within the same group (e.g. all L3 boxes in Data Lake should be `width: 100, height: 90`) rather than creating chaotic mismatched boxes unless necessary.
  - Distribute items evenly. If you have 4 identical nodes in a 500px wide group, calculate their offsets explicitly using a loop to distribute empty space.
- **Micro-Layouts (Flexbox inside HTML Nodes)**: When designing the inner HTML for custom nodes, use flexbox correctly to ensure perfectly centered icons and text: `display: flex; flex-direction: column; align-items: center; justify-content: center;`. This creates tight, professional cards instead of misaligned jumbled text.
- **Canvas Constraints**: Use a robust `COL_WIDTH` and `ROW_HEIGHT` to coordinate absolute positions to eliminate guessing. Map every single node coordinate in a consistent tabular manner before generating `.html`.

## 7. Execution Pipeline
1. **Understand Logic:** Parse the prompt into a Node and Edge logical matrix.
2. **Boilerplate Merge:** Read `assets/x6-boilerplate.html` using the read file tool for reference to get the accurate node registration logic, and inject the boilerplate into your final PPT canvas container.
3. **Coordinate Planning:** Plan `x` and `y` matrix on a defined grid system (`COL_WIDTH`, `ROW_HEIGHT`, `Margins`) manually. Prevent overlapping layouts explicitly via calculation. Enforce strict padding for child nodes.
4. **Render Check:** Load nodes into `graph.addNodes()` and specific semantic edges into `graph.addEdges()`. Verify no visual collapses occur.

## 8. Crash Avoidance & Rendering Failures (CRITICAL)
If your generated HTML only shows groups but no child nodes or edges (a "blank canvas bug"), you have triggered a fatal JS exception that halted execution. Always follow these rules to avoid crashes:
1. **Unregistered Shapes Stop Execution**: If you use `graph.addNode({ shape: 'my-shape' })`, `'my-shape'` MUST be registered earlier with `X6.Graph.registerNode`. If you rename shapes (e.g., `'html-node'` to `'html-base'`), ensure all usages are updated. One unregistered shape will crash the entire render.
2. **Inline HTML Overrides**: If you provide an inline `html: () => "..."` function inside `graph.addNode(...)`, the specified `shape` **MUST** be one that inherited from `'html'`. 
3. **Invalid Ports/Routing**: Do not connect to `port` IDs that do not exist on the target shape's `ports` definition. E.g. If you use `port: 'left'`, ensure the target Node's registered shape explicitly defines `left` in its `ports.groups`. 
4. **Try-Catch Block Pattern**: When iterating over many edges or nodes, write defensively so a single failed node doesn't abort the rest of the diagram.

## 9. Pure HTML/CSS Zoned Area Architectures (Non-X6 Layouts)
When building complex layered block architectures using pure Tailwind CSS Flexbox/Grid (bypassing X6), you **MUST** strictly manage vertical real estate to prevent the document from breaking out of the 1920x1080 fixed boundary.
- **Vertical Height Budgeting (1080px Total)**:
  - Subtract Header (e.g., `100px`) and Footer (e.g., `60px`), leaving exactly `920px` for the Main container.
  - Subtract paddings (e.g., `p-8` is `64px` total), leaving strictly ~`850px` of usable inner vertical space.
  - If your architecture stacks 3 massive sections vertically (e.g., Top Core, Middle Management, Bottom Infra), you MUST assign explicit fractional heights (e.g. `flex-[3]`, `h-[120px]`, `h-[100px]`) that safely sum up to <= `850px`. 
- **The `min-h-0` Savior constraint**: Flex children naturally resist shrinking below their content height. In dense architectures, this causes explosive out-of-bounds stretching. You MUST add `min-h-0` to all major flex-column wrappers (`flex flex-col min-h-0`) to allow internal scrolls or shrinkage, preventing the layout from overflowing past the window baseline.
- **Overflow Prevention via Inner Scroll**: For highly dense lists inside cards, always add `overflow-y-auto` to the immediate child container of the card, combined with `flex-1 min-h-0`, so it scrolls internally rather than pushing the boundary.
- **Micro-Typography & Density**: For dense L3/L4 nodes, drop text to `text-[10px]` or `text-[9px]`, use tight leading (`leading-tight`), and remove excessive vertical paddings. Shrink the gap to `gap-1` or `gap-2`. Do not use huge padding like `p-4` inside deeply nested inner-boxes.
- **Flexible Layout Proportions (No Hardcoded Heights)**: NEVER use static pixel minimums like `min-h-[80px]` or `h-[120px]` on major intermediate rows if they stack vertically inside a space-constrained parent. Instead, use relative flex values (`flex-[2]`, `flex-1`, `flex-[0.5]`) and rely strictly on natural flexbox proportions. Hardcoding heights structurally guarantees that children will snap through the bottom boundary when the math does not align.
- **Viewport Auto-Scaling & HTML Boilerplate (CRITICAL)**: Because the `.slide-container` acts as a rigid `1920x1080` canvas, opening it on smaller computer screens will physically slice off the bottom of the slide. 
  - **Rule:** Do not write the CSS and scaling JavaScript from scratch.
  - **Action:** You **MUST** use the read file tool to read `assets/html-boilerplate.html` to get the robust Flexbox-based centering and auto-scaling logic, then inject your HTML nodes into it.
