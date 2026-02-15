
const fs = require('fs');
const path = require('path');

const dir = 'docs/presentations/cpu_industry_20260215_v1';
const files = fs.readdirSync(dir).filter(f => f.startsWith('slide-') && f.endsWith('.html'));

// Generic Replacer for common variables
// We assume we want to map:
// #177C52 -> --brand-primary
// #2D4036 -> --brand-accent
// #000000 -> --brand-secondary
// #333333 -> --brand-text
// #666666 -> --brand-text-light

// Semantic colors
// #D32F2F -> --color-risk (approx red)
// #FFA000 -> --color-warning (approx amber)
// #0288D1 -> --color-info (approx blue)
// #388E3C -> --color-success (approx green)

const colorMap = {
    '#177C52': '--brand-primary',
    '#2D4036': '--brand-accent',
    '#009A44': '--brand-primary', // New definition
    '#1D428A': '--brand-secondary', // New definition
    '#FF671F': '--brand-accent', // New definition
};


files.forEach(file => {
    let content = fs.readFileSync(path.join(dir, file), 'utf8');
    let changed = false;

    // 1. Inject Helper Function if Chart.js is present and helper is missing
    if (content.includes('new Chart(') && !content.includes('function getBrandColor')) {
        const helperScript = `
        function getBrandColor(variable) {
            return getComputedStyle(document.body).getPropertyValue(variable).trim();
        }
        `;
        // Insert before new Chart
        content = content.replace(/(new Chart\()/g, `${helperScript}\n        $1`);
        changed = true;
    }

    // 2. Replace Hardcoded Colors in Chart config
    // We look for patterns like: borderColor: '#177C52'
    // And replace with: borderColor: getBrandColor('--brand-primary')
    
    // Regex for hex colors inside Chart config context is hard.
    // Simpler approach: Iterate over known hardcoded values found in previous steps.
    // The previous generation was consistent.
    
    // Brand Primary
    if (content.includes("'#177C52'")) {
        content = content.replace(/'#177C52'/g, "getBrandColor('--brand-primary')");
        changed = true;
    }
     if (content.includes('"#177C52"')) {
        content = content.replace(/"#177C52"/g, "getBrandColor('--brand-primary')");
        changed = true;
    }

    // Brand Accent
    if (content.includes("'#2D4036'")) {
        content = content.replace(/'#2D4036'/g, "getBrandColor('--brand-accent')");
        changed = true;
    }
    
    // Brand Secondary / Black
    if (content.includes("'#000000'")) {
        content = content.replace(/'#000000'/g, "getBrandColor('--brand-secondary')");
        changed = true;
    }

    // Fix other colors 
    // Example: Slide 11 used #4B5563 (Gray), #10B981 (Emerald), #8B5CF6 (Purple)
    // We should map them to brand palette or semantic colors.
    
    // #10B981 (Emerald) -> brand-primary
    if (content.includes("'#10B981'")) {
        content = content.replace(/'#10B981'/g, "getBrandColor('--brand-primary')");
        changed = true;
    }
    
    // #4B5563 (Gray) -> brand-secondary (or text-light)
    if (content.includes("'#4B5563'")) {
         content = content.replace(/'#4B5563'/g, "getBrandColor('--brand-text-light')");
         changed = true;
    }

    if (changed) {
        fs.writeFileSync(path.join(dir, file), content);
        console.log(`Updated ${file}`);
    }
});
