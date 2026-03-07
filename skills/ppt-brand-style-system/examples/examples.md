# PPT Brand-Style System — Implementation Examples

> Derived from brands.yml single source of truth.
> These examples show how to implement style-profile switching in HTML.

## 1. CSS Style Profile Blocks

Generate one block per style profile from `brands.yml -> brands.{id}`

```css
/* KPMG 风格档案样式 */
.brand-kpmg {
  --brand-primary: #00338D;
  --brand-secondary: #0091DA;
  --brand-accent: #483698;
  font-family: Georgia, Arial, "Noto Sans SC", sans-serif;
}

.brand-kpmg .sidebar {
  background-color: var(--brand-primary);
}

/* McKinsey 风格档案样式 */
.brand-mckinsey {
  --brand-primary: #00A3A0;
  --brand-secondary: #1A1A1A;
  --brand-accent: #FF6B35;
  font-family: "Helvetica Neue", "PingFang SC", sans-serif;
}

.brand-mckinsey .sidebar {
  display: none; /* McKinsey无侧边栏 */
}

/* BCG 风格档案样式 — C1 fix: primary changed to #009A44 */
.brand-bcg {
  --brand-primary: #009A44;
  --brand-secondary: #1D428A;
  --brand-accent: #FF671F;
  font-family: Arial, "Microsoft YaHei", sans-serif;
}

.brand-bcg .navbar {
  background-color: var(--brand-primary);
}

/* Bain 风格档案样式 */
.brand-bain {
  --brand-primary: #DC291E;
  --brand-secondary: #1A1A1A;
  --brand-accent: #F5A623;
  font-family: "Gotham", Arial, "Source Han Sans SC", sans-serif;
}

.brand-bain .insight-sidebar {
  background-color: var(--brand-primary);
}

/* Deloitte 风格档案样式 */
.brand-deloitte {
  --brand-primary: #86BC25;
  --brand-secondary: #0033A0;
  --brand-accent: #FF671F;
  font-family: Arial, "Noto Sans SC", sans-serif;
}

.brand-deloitte .card {
  border-radius: 12px;
  background: linear-gradient(135deg, #f8f9fa, #ffffff);
}
```

## 2. JS Style Profile Color Lookup

Derived from `brands.yml → brands.{id}.colors`. Updated to return full palette including supplementary colors.

```javascript
function getStyleProfileColors(styleProfileId) {
  const colors = {
    kpmg:     { primary: '#00338D', secondary: '#0091DA', accent: '#483698', supplementary: ['#00A3A1', '#E31C79'] },
    mckinsey: { primary: '#00A3A0', secondary: '#1A1A1A', accent: '#FF6B35', supplementary: ['#5D5FEF', '#00B4A6'] },
    bcg:      { primary: '#009A44', secondary: '#1D428A', accent: '#FF671F', supplementary: ['#6CC24A', '#0072CE'] },
    bain:     { primary: '#DC291E', secondary: '#1A1A1A', accent: '#F5A623', supplementary: ['#7ED321', '#4A90E2'] },
    deloitte: { primary: '#86BC25', secondary: '#0033A0', accent: '#FF671F', supplementary: ['#00A3A0', '#E31C79'] },
    strategic: { primary: '#0F172A', secondary: '#64748B', accent: '#E2E8F0', supplementary: ['#334155', '#CBD5E1'] }
  };
  return colors[styleProfileId] || colors.kpmg;
}

const getBrandColors = getStyleProfileColors;
```

## 3. JS Style Profile Switching Function

```javascript
function switchStyleProfile(styleProfileId) {
  document.body.className = 'brand-' + styleProfileId;
  // 风格切换后必须延迟 50ms 重绘图表
  if (window.myChart) {
    setTimeout(() => {
      updateChartColors(styleProfileId);
      window.myChart.resize();
    }, 50);
  }
}

const switchBrand = switchStyleProfile;

function updateChartColors(styleProfileId) {
  const colors = getStyleProfileColors(styleProfileId);
  window.myChart.data.datasets[0].backgroundColor = colors.primary;
  window.myChart.update();
}
```

## 4. HTML Style Switching Template (开发预览用)

> **注意**: 成片 `slide-*.html` 不应包含此控件

```html
<div class="style-profile-selector fixed top-4 right-4 z-50 bg-white p-3 rounded-lg shadow-lg">
  <label class="block text-sm font-medium mb-2">选择风格档案：</label>
  <select class="style-profile-select w-full p-2 border rounded" onchange="switchStyleProfile(this.value)">
    <option value="kpmg">KPMG</option>
    <option value="mckinsey">McKinsey</option>
    <option value="bcg">BCG</option>
    <option value="bain">Bain</option>
    <option value="deloitte">Deloitte</option>
    <option value="strategic">Editorial Briefing</option>
  </select>
</div>
```

## 5. Complete HTML Template (开发预览页)

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>多风格演示文稿</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.4.3/dist/echarts.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    :root {
      --slide-width: 1280px;
      --slide-height: 720px;
    }
    /* 引入上方 CSS Style Profile Blocks 中的风格样式 */
  </style>
</head>
<body class="brand-kpmg">
  <div class="slide-container w-[var(--slide-width)] h-[var(--slide-height)] mx-auto p-8">
    <!-- 风格切换控件（仅开发预览用） -->
    <!-- 引入上方 HTML Style Selector -->
    <!-- 幻灯片内容 -->
  </div>
  <script>
    // 引入上方 JS Style Profile Color Lookup + JS Style Profile Switching Function
    // 如需兼容旧调用，可继续暴露 getBrandColors / switchBrand 别名
  </script>
</body>
</html>
