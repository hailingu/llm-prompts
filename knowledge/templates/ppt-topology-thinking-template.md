# Slide {N}: Topology Thinking

## 1. Mission and Page Objective
- **Goal**: State the single system architecture, flowchart, or diagram task this slide must accomplish.
- **Topology Class**: Block Architecture (Zoning/Nested) | Flowchart (Sequence/Process) | Hybrid
- **Data Strategy**: Topology-led

## 2. Canvas & Grid Coordination System
- **Global Canvas Size**: W: {Width}, H: {Height} (e.g. 1920x1080 or standard 1800x900 internal bounding)
- **Unit Grid System**: W={W}px, H={H}px per unit
- **Layout Margins**: top, left, right, bottom

## 3. Visual Bounding Box Matrix
*Crucial: For Block Architecture, explicit positioning (X, Y, W, H) is mandatory. Do not rely on flow-line automated layouts. Explicitly state units.*

| Area/Node | Parent | Type | Logic-X | Logic-Y | Width | Height | Alignment / Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MAIN: [Name]** | Canvas | Container | [val] | [val] | [val] | [val] | [Desc] |
| - Sub: [Name] | [MAIN ID] | Item | [val] | [val] | [val] | [val] | [Desc] |
| - Column: [Name] | Canvas | Spanner | [val] | [val] | [val] | [val] | [Desc] |
*Note: Values can be specified in Absolute Pixels or Grid Units. Just ensure internal consistency.*

## 4. Edge & Flow Strategy (Only if Topology Class is Flowchart or Hybrid)
- **Edge Routing**: orthogonal | straight | curve
- **Ports Strategy**: Left-to-Right | Top-to-Bottom | Matrix
- **Key Intersections**: [List major connection lines. Return 'None - Zero Edge Policy' if purely Block Architecture]

## 5. Visual Styling & HTML Injection
- **Group Containers**: Expected stroke, thickness, label position (e.g. top-left absolute).
- **Node HTML Design**: List which nodes require icons, secondary text, or specific HTML styling (e.g. Tailwind classes: `bg-blue-50 text-blue-800`).
- **Icons**: Mention if standard text icons (e.g. ☁️, ⚙️) or CSS background elements will be used.

## 6. Layout Mitigations & Fallback
- **Overflow Risk**: What happens if the nodes exceed container limits?
- **Alignment Lock**: Explicit alignment rules (e.g. "Management container width must equal sum of Core + WS containers").