"""Mermaid renderer using mermaid-cli to convert mermaid code to images."""

import subprocess
import shutil
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from pptx.util import Inches

from ...protocols.visual_data_protocol import VisualDataProtocol
from ...protocols.renderer_interface import IRenderer
from ..base import BaseRenderer


logger = logging.getLogger(__name__)


class MermaidRenderer(BaseRenderer):
    """Mermaid图表渲染器，使用 mermaid-cli (mmdc) 将 mermaid 代码渲染为图片"""
    
    @property
    def name(self) -> str:
        return "mermaid-cli"
    
    @property
    def supported_types(self) -> List[str]:
        """支持所有 mermaid 图表类型（不包括 gantt，优先使用 native renderer）"""
        return [
            "mermaid",  # generic catch-all for any mermaid code
            "flowchart", "flow", "process",
            "sequence", "sequenceDiagram", 
            # "gantt",  # 移除 gantt 支持，避免与 native gantt renderer 冲突
            "class", "classDiagram",
            "state", "stateDiagram",
            "er", "erDiagram",
            "journey", "userJourney",
            "gitGraph",
            "pie", "pieChart",
            "matrix",  # matrix visuals with mermaid_code
        ]
    
    def is_available(self) -> bool:
        """检查 mmdc 命令是否可用"""
        return shutil.which("mmdc") is not None
    
    def estimate_quality(self, visual_data: VisualDataProtocol) -> int:
        """
        评估渲染质量
        
        Mermaid 对所有图表都提供专业级渲染，质量评分高。
        只有在数据非常复杂（可能超时）时才降低评分。
        """
        # 检查是否有 mermaid_code
        if not visual_data.placeholder_data:
            return 0
        
        mermaid_code = visual_data.placeholder_data.get('mermaid_code', '')
        if not mermaid_code:
            return 0
        
        # 统计节点/元素数量（粗略估计）
        line_count = len(mermaid_code.strip().split('\n'))
        
        # ≤20行：优秀 (90分)
        if line_count <= 20:
            return 90
        # 21-50行：良好 (85分)
        elif line_count <= 50:
            return 85
        # >50行：可能超时或渲染慢 (70分)
        else:
            return 70
    
    def _do_render(
        self, 
        slide: Any, 
        visual_data: VisualDataProtocol, 
        spec: Dict[str, Any],
        left: float, 
        top: float, 
        width: float, 
        height: float
    ) -> bool:
        """渲染 Mermaid 图表到 PPT"""
        # 获取 mermaid 代码
        if not visual_data.placeholder_data:
            logger.error("No placeholder_data found in visual_data")
            return False
        
        mermaid_code = visual_data.placeholder_data.get('mermaid_code', '')
        if not mermaid_code:
            logger.error("No mermaid_code found in placeholder_data")
            return False
        
        # 注入主题配置，使颜色与 PPT 设计规范一致
        themed_code = self._inject_theme_config(mermaid_code, spec)
        
        try:
            # 渲染为图片
            image_path = self._render_to_image(
                themed_code, 
                width_inches=width, 
                height_inches=height,
                spec=spec
            )
            
            if not image_path or not Path(image_path).exists():
                logger.error(f"Failed to render mermaid chart to {image_path}")
                return False
            
            # 插入图片到 PPT
            slide.shapes.add_picture(
                str(image_path),
                Inches(left),
                Inches(top),
                width=Inches(width),
                height=Inches(height)
            )
            
            logger.info(f"Successfully rendered mermaid chart: {visual_data.type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to render mermaid chart: {e}", exc_info=True)
            return False
    
    def _render_to_image(
        self, 
        mermaid_code: str, 
        width_inches: float,
        height_inches: float,
        spec: Dict[str, Any]
    ) -> Optional[str]:
        """
        使用 mmdc 将 mermaid 代码渲染为图片
        
        Args:
            mermaid_code: Mermaid 图表代码
            width_inches: 目标宽度（英寸）
            height_inches: 目标高度（英寸）
            spec: 设计规范（用于主题配置）
            
        Returns:
            渲染后的图片路径，失败返回 None
        """
        # 创建临时文件
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.mmd', 
            delete=False,
            encoding='utf-8'
        ) as mmd_file:
            mmd_file.write(mermaid_code)
            mmd_path = mmd_file.name
        
        # 输出 PNG 路径
        output_path = mmd_path.replace('.mmd', '.png')
        
        try:
            # 转换尺寸：1英寸 = 96像素（PPT标准DPI）
            width_px = int(width_inches * 96)
            height_px = int(height_inches * 96)
            
            # 构建 mmdc 命令
            cmd = [
                'mmdc',
                '-i', mmd_path,
                '-o', output_path,
                '-b', 'transparent',  # 透明背景
                '-w', str(width_px),  # 宽度
                '-H', str(height_px), # 高度
            ]
            
            # 如果有主题配置，可以添加 -t 参数
            # 例如：'-t', 'default' / 'forest' / 'dark' / 'neutral'
            theme = self._get_mermaid_theme(spec)
            if theme:
                cmd.extend(['-t', theme])
            
            # 执行渲染
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30秒超时
            )
            
            if result.returncode != 0:
                logger.error(f"mmdc command failed: {result.stderr}")
                return None
            
            # 验证输出文件
            if not Path(output_path).exists():
                logger.error(f"Output file not created: {output_path}")
                return None
            
            return output_path
            
        except subprocess.TimeoutExpired:
            logger.error("mmdc command timed out (>30s)")
            return None
        except Exception as e:
            logger.error(f"Failed to execute mmdc: {e}", exc_info=True)
            return None
        finally:
            # 清理临时 .mmd 文件
            try:
                Path(mmd_path).unlink()
            except:
                pass
    
    def _get_mermaid_theme(self, spec: Dict[str, Any]) -> Optional[str]:
        """
        根据 PPT 设计规范选择 Mermaid 主题
        
        Mermaid 主题选项：
        - default: 默认蓝色主题
        - forest: 绿色主题
        - dark: 深色主题
        - neutral: 中性灰色主题
        - base: 自定义主题（通过 themeVariables 配置）
        """
        # 使用 neutral 主题作为基础，颜色相对柔和
        return 'neutral'
    
    def _inject_theme_config(self, mermaid_code: str, spec: Dict[str, Any]) -> str:
        """
        在 mermaid 代码中注入主题配置，使其与 PPT 设计规范一致
        
        Args:
            mermaid_code: 原始 mermaid 代码
            spec: 设计规范
            
        Returns:
            注入主题配置后的 mermaid 代码
        """
        # 提取颜色配置
        color_system = spec.get('color_system', {})
        primary = color_system.get('primary', '#003A70')
        primary_container = color_system.get('primary_container', '#C8A951')
        surface_variant = color_system.get('surface_variant', '#F3F4F6')
        on_surface = color_system.get('on_surface', '#0F172A')
        tertiary = color_system.get('tertiary', '#6B7280')
        
        # 构建主题变量（增加字体大小、箭头粗细，强制白色文字，箭头标签可见）
        theme_config = (
            f"%%{{init: {{'theme':'base', 'themeVariables': {{"
            f"'primaryColor':'{primary}',"
            f"'primaryTextColor':'#FFFFFF',"  # 强制白色文字
            f"'primaryBorderColor':'{primary}',"
            f"'lineColor':'{primary_container}',"
            f"'secondaryColor':'{surface_variant}',"
            f"'secondaryTextColor':'#0F172A',"  # 浅色背景用深色文字
            f"'secondaryBorderColor':'{tertiary}',"
            f"'tertiaryColor':'#FFFFFF',"
            f"'tertiaryTextColor':'#0F172A',"
            f"'tertiaryBorderColor':'{tertiary}',"
            f"'noteBkgColor':'{surface_variant}',"
            f"'noteTextColor':'{on_surface}',"
            f"'noteBorderColor':'{tertiary}',"
            f"'textColor':'#FFFFFF',"  # 全局文字颜色白色
            f"'labelTextColor':'#FFFFFF',"  # 标签文字白色
            f"'edgeLabelBackground':'{primary}',"  # 箭头标签深蓝背景（#003A70）
            f"'edgeLabelTextColor':'#FFFFFF',"  # 箭头标签白色文字
            f"'fontSize':'16px',"  # 增加字体大小 14px→16px
            f"'fontFamily':'Georgia, SimHei, sans-serif'"
            f"}}, 'flowchart': {{'curve': 'linear', 'defaultRenderer': 'elk'}} }}}}%%\n"
        )
        
        # 如果已有 init 配置，跳过注入（避免重复）
        if '%%{init:' in mermaid_code or 'themeVariables' in mermaid_code:
            return mermaid_code
        
        # 在第一行图表类型声明前插入主题配置
        lines = mermaid_code.strip().split('\n')
        if lines:
            return theme_config + '\n'.join(lines)
        return mermaid_code
        return 'default'
