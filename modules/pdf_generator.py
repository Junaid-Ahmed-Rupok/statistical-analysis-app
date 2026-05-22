from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether, HRFlowable
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.platypus.flowables import Flowable
from reportlab.pdfgen import canvas
import matplotlib.pyplot as plt
import io
import pandas as pd
import numpy as np
import datetime

class PDFReportGenerator:
    """
    Premium PDF report generator with professional styling.
    """
    
    def __init__(self):
        self.styles = self._create_styles()
        self.colors = {
            'navy': colors.HexColor('#0A2342'),
            'medium_navy': colors.HexColor('#1B4F72'),
            'gold': colors.HexColor('#F0A500'),
            'gold_light': colors.HexColor('#FFF3CD'),
            'green': colors.HexColor('#1E8449'),
            'red': colors.HexColor('#C0392B'),
            'bg_light': colors.HexColor('#F4F6F9'),
            'text_primary': colors.HexColor('#1A1A2E'),
            'text_muted': colors.HexColor('#6C757D'),
            'border': colors.HexColor('#E0E6ED'),
            'white': colors.white,
        }
        
    def _create_styles(self):
        styles = getSampleStyleSheet()
        
        styles.add(ParagraphStyle(
            name='CoverTitle',
            parent=styles['Title'],
            fontSize=32,
            textColor=colors.HexColor('#0A2342'),
            spaceAfter=12,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            leading=38
        ))
        
        styles.add(ParagraphStyle(
            name='CoverSubtitle',
            parent=styles['Normal'],
            fontSize=16,
            textColor=colors.HexColor('#F0A500'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica',
            leading=22
        ))
        
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading2'],
            fontSize=20,
            textColor=colors.HexColor('#0A2342'),
            spaceAfter=16,
            spaceBefore=24,
            fontName='Helvetica-Bold',
            leading=26,
            borderPadding=(0, 0, 8, 0),
        ))
        
        styles.add(ParagraphStyle(
            name='SubSectionHeader',
            parent=styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#1B4F72'),
            spaceAfter=10,
            spaceBefore=16,
            fontName='Helvetica-Bold',
            leading=18,
        ))
        
        styles.add(ParagraphStyle(
            name='BodyText',
            parent=styles['Normal'],
            fontSize=10,
            leading=15,
            textColor=colors.HexColor('#1A1A2E'),
            alignment=TA_JUSTIFY,
            fontName='Helvetica',
        ))
        
        styles.add(ParagraphStyle(
            name='MetricValue',
            parent=styles['Normal'],
            fontSize=28,
            textColor=colors.HexColor('#0A2342'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            leading=32,
        ))
        
        styles.add(ParagraphStyle(
            name='MetricLabel',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor('#6C757D'),
            alignment=TA_CENTER,
            fontName='Helvetica',
            leading=12,
        ))
        
        styles.add(ParagraphStyle(
            name='InsightBullet',
            parent=styles['Normal'],
            fontSize=10,
            leading=16,
            textColor=colors.HexColor('#1A1A2E'),
            leftIndent=20,
            bulletIndent=10,
            fontName='Helvetica',
        ))
        
        styles.add(ParagraphStyle(
            name='CaptionText',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.HexColor('#6C757D'),
            alignment=TA_CENTER,
            fontName='Helvetica-Oblique',
            leading=10,
        ))
        
        styles.add(ParagraphStyle(
            name='WarningText',
            parent=styles['Normal'],
            fontSize=10,
            leading=15,
            textColor=colors.HexColor('#C0392B'),
            fontName='Helvetica-Bold',
        ))
        
        styles.add(ParagraphStyle(
            name='SuccessText',
            parent=styles['Normal'],
            fontSize=10,
            leading=15,
            textColor=colors.HexColor('#1E8449'),
            fontName='Helvetica-Bold',
        ))
        
        return styles
    
    # ── Page Template Callbacks ─────────────────────────────────
    
    def _on_first_page(self, canvas_obj, doc):
        """Header/footer for first page."""
        self._draw_footer(canvas_obj, doc)
    
    def _on_later_pages(self, canvas_obj, doc):
        """Header/footer for subsequent pages."""
        self._draw_header(canvas_obj, doc)
        self._draw_footer(canvas_obj, doc)
    
    def _draw_header(self, canvas_obj, doc):
        canvas_obj.saveState()
        canvas_obj.setStrokeColor(self.colors['gold'])
        canvas_obj.setLineWidth(2)
        canvas_obj.line(inch, 10.85*inch, 7.5*inch, 10.85*inch)
        canvas_obj.setFillColor(self.colors['navy'])
        canvas_obj.setFont('Helvetica-Bold', 9)
        canvas_obj.drawString(inch, 10.92*inch, "StatsPro Analysis Report")
        canvas_obj.setFillColor(self.colors['text_muted'])
        canvas_obj.setFont('Helvetica', 7)
        canvas_obj.drawRightString(7.5*inch, 10.92*inch, "Confidential")
        canvas_obj.restoreState()
    
    def _draw_footer(self, canvas_obj, doc):
        canvas_obj.saveState()
        canvas_obj.setStrokeColor(self.colors['border'])
        canvas_obj.setLineWidth(0.5)
        canvas_obj.line(inch, 0.65*inch, 7.5*inch, 0.65*inch)
        canvas_obj.setFillColor(self.colors['text_muted'])
        canvas_obj.setFont('Helvetica', 7)
        canvas_obj.drawString(inch, 0.45*inch, f"StatsPro | Confidential | Generated {datetime.datetime.now().strftime('%d %b %Y')}")
        canvas_obj.drawRightString(7.5*inch, 0.45*inch, f"Page {canvas_obj.getPageNumber()}")
        canvas_obj.restoreState()
    
    # ── Gold Divider Flowable ───────────────────────────────────
    
    class GoldDivider(Flowable):
        def __init__(self, width, thickness=1.5):
            Flowable.__init__(self)
            self.width = width
            self.thickness = thickness
            self.height = thickness + 6
        
        def draw(self):
            self.canv.setStrokeColor(colors.HexColor('#F0A500'))
            self.canv.setLineWidth(self.thickness)
            self.canv.line(0, 3, self.width, 3)
    
    # ── Main Generator ──────────────────────────────────────────
    
    def generate(
        self,
        df: pd.DataFrame,
        overview: dict,
        stats_results: dict,
        insights: dict,
        figures: list = None,
        cleaned_df: pd.DataFrame = None
    ) -> io.BytesIO:
        """
        Generate the complete professional PDF report.
        """
        buffer = io.BytesIO()
        
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            topMargin=0.7*inch,
            bottomMargin=0.9*inch,
            leftMargin=0.75*inch,
            rightMargin=0.75*inch,
            title='StatsPro Analysis Report',
            author='StatsPro',
        )
        
        story = []
        
        # Build report sections
        story.extend(self._cover_page(overview))
        story.append(PageBreak())
        story.extend(self._executive_summary(overview, insights))
        story.append(PageBreak())
        story.extend(self._data_overview(df, overview))
        story.append(PageBreak())
        story.extend(self._statistical_tests_section(stats_results))
        story.append(PageBreak())
        
        if figures:
            story.extend(self._visualizations_section(figures))
            story.append(PageBreak())
        
        story.extend(self._insights_section(insights))
        
        doc.build(story, onFirstPage=self._on_first_page, onLaterPages=self._on_later_pages)
        buffer.seek(0)
        return buffer
    
    # ── COVER PAGE ──────────────────────────────────────────────
    
    def _cover_page(self, overview: dict) -> list:
        elements = []
        
        elements.append(Spacer(1, 1.2*inch))
        
        # Gold top bar
        elements.append(HRFlowable(
            width="60%", thickness=4, color=self.colors['gold'],
            spaceAfter=30, spaceBefore=0
        ))
        
        # Title
        elements.append(Paragraph("STATSPRO", self.styles['CoverTitle']))
        elements.append(Paragraph("Professional Statistical Analysis Report", self.styles['CoverSubtitle']))
        
        # Gold bottom bar
        elements.append(HRFlowable(
            width="60%", thickness=4, color=self.colors['gold'],
            spaceAfter=40, spaceBefore=10
        ))
        
        elements.append(Spacer(1, 0.5*inch))
        
        # Info cards
        card_data = [
            ['Dataset Overview', ''],
            ['Total Rows', f"{overview['rows']:,}"],
            ['Total Columns', str(overview['columns'])],
            ['Data Quality Score', f"{overview['quality_score']:.1f}/100"],
            ['Completeness', f"{100 - overview['missing_percentage']:.1f}%"],
            ['Report Generated', datetime.datetime.now().strftime('%B %d, %Y at %H:%M')],
        ]
        
        table = Table(card_data, colWidths=[2.5*inch, 3.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), self.colors['white']),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 13),
            ('SPAN', (0, 0), (-1, 0)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND', (0, 1), (-1, -1), self.colors['white']),
            ('TEXTCOLOR', (0, 1), (0, -1), self.colors['text_muted']),
            ('TEXTCOLOR', (1, 1), (1, -1), self.colors['navy']),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 1), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, self.colors['border']),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [self.colors['white'], self.colors['bg_light']]),
            ('PADDING', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(table)
        
        elements.append(Spacer(1, 1*inch))
        elements.append(Paragraph("© 2025 StatsPro | Confidential | All Rights Reserved", self.styles['CaptionText']))
        
        return elements
    
    # ── EXECUTIVE SUMMARY ───────────────────────────────────────
    
    def _executive_summary(self, overview: dict, insights: dict) -> list:
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        elements.append(self.GoldDivider(450))
        elements.append(Spacer(1, 16))
        
        # KPI cards in 3-column layout
        quality = overview['quality_score']
        completeness = 100 - overview['missing_percentage']
        readiness = insights.get('readiness_score', 0) if insights else 0
        
        kpi_data = [
            [
                Paragraph(f"{quality:.0f}/100", self.styles['MetricValue']),
                Paragraph(f"{completeness:.1f}%", self.styles['MetricValue']),
                Paragraph(f"{readiness:.0f}/100", self.styles['MetricValue']),
            ],
            [
                Paragraph("Data Quality", self.styles['MetricLabel']),
                Paragraph("Completeness", self.styles['MetricLabel']),
                Paragraph("Readiness", self.styles['MetricLabel']),
            ]
        ]
        
        kpi_table = Table(kpi_data, colWidths=[2*inch, 2*inch, 2*inch])
        kpi_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['gold_light']),
            ('BACKGROUND', (0, 1), (-1, 1), self.colors['white']),
            ('BOX', (0, 0), (-1, -1), 1.5, self.colors['gold']),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, self.colors['border']),
            ('PADDING', (0, 0), (-1, -1), 12),
        ]))
        
        elements.append(kpi_table)
        elements.append(Spacer(1, 24))
        
        # Key Findings
        elements.append(Paragraph("Key Findings", self.styles['SubSectionHeader']))
        
        if insights:
            for h in insights.get('highlights', [])[:5]:
                elements.append(Paragraph(f"✓ {h}", self.styles['InsightBullet']))
        
        elements.append(Spacer(1, 12))
        
        # Warnings
        if insights and insights.get('warnings'):
            elements.append(Paragraph("⚠ Warnings", self.styles['SubSectionHeader']))
            for w in insights['warnings'][:3]:
                elements.append(Paragraph(f"• {w}", self.styles['WarningText']))
        
        return elements
    
    # ── DATA OVERVIEW ───────────────────────────────────────────
    
    def _data_overview(self, df: pd.DataFrame, overview: dict) -> list:
        elements = []
        
        elements.append(Paragraph("Data Overview", self.styles['SectionHeader']))
        elements.append(self.GoldDivider(450))
        elements.append(Spacer(1, 16))
        
        # Column summary
        elements.append(Paragraph("Column Summary", self.styles['SubSectionHeader']))
        
        col_info = [
            ['Column Name', 'Data Type', 'Missing', 'Unique Values'],
        ]
        for col in df.columns[:20]:
            missing = df[col].isnull().sum()
            col_info.append([
                col,
                str(df[col].dtype),
                str(missing),
                str(df[col].nunique()),
            ])
        
        table = Table(col_info, colWidths=[2.2*inch, 1.5*inch, 1.2*inch, 1.3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['medium_navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), self.colors['white']),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, self.colors['border']),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [self.colors['white'], self.colors['bg_light']]),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(table)
        
        # Missing values
        if overview['total_missing'] > 0:
            elements.append(Spacer(1, 16))
            elements.append(Paragraph("Missing Values Analysis", self.styles['SubSectionHeader']))
            
            missing_df = overview['missing_table']
            missing_data = [['Column', 'Missing Count', 'Missing %']]
            for _, row in missing_df.iterrows():
                if row['Missing Count'] > 0:
                    missing_data.append([
                        row['Column'],
                        str(row['Missing Count']),
                        f"{row['Missing Percentage']:.1f}%"
                    ])
            
            if len(missing_data) > 1:
                mtable = Table(missing_data, colWidths=[2.5*inch, 1.8*inch, 1.8*inch])
                mtable.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), self.colors['medium_navy']),
                    ('TEXTCOLOR', (0, 0), (-1, 0), self.colors['white']),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                    ('GRID', (0, 0), (-1, -1), 0.5, self.colors['border']),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [self.colors['white'], self.colors['bg_light']]),
                    ('PADDING', (0, 0), (-1, -1), 6),
                ]))
                elements.append(mtable)
        
        # Descriptive statistics
        elements.append(Spacer(1, 16))
        elements.append(Paragraph("Descriptive Statistics", self.styles['SubSectionHeader']))
        
        desc = df.describe().round(2)
        desc_data = [['Statistic'] + list(desc.columns)]
        for idx in desc.index:
            desc_data.append([idx] + [str(desc.loc[idx, col]) for col in desc.columns])
        
        if len(desc.columns) <= 8:
            col_widths = [1.5*inch] + [0.8*inch] * len(desc.columns)
            dtable = Table(desc_data, colWidths=col_widths)
            dtable.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors['medium_navy']),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.colors['white']),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('GRID', (0, 0), (-1, -1), 0.5, self.colors['border']),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [self.colors['white'], self.colors['bg_light']]),
                ('PADDING', (0, 0), (-1, -1), 4),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ]))
            elements.append(dtable)
        
        return elements
    
    # ── STATISTICAL TESTS ───────────────────────────────────────
    
    def _statistical_tests_section(self, stats_results: dict) -> list:
        elements = []
        
        elements.append(Paragraph("Statistical Analysis Results", self.styles['SectionHeader']))
        elements.append(self.GoldDivider(450))
        elements.append(Spacer(1, 16))
        
        # Normality
        norm = stats_results.get('normality', pd.DataFrame())
        if not norm.empty:
            elements.append(Paragraph("Normality Tests", self.styles['SubSectionHeader']))
            elements.append(self._build_test_table(norm, ['Column', 'Test', 'Statistic', 'P-Value', 'Normal']))
            elements.append(Spacer(1, 10))
        
        # Pearson Correlation
        pearson = stats_results.get('pearson_correlation', pd.DataFrame())
        if not pearson.empty:
            elements.append(Paragraph("Pearson Correlations", self.styles['SubSectionHeader']))
            top_p = pearson.nlargest(8, _corr_col(pearson))
            elements.append(self._build_test_table(top_p, ['Variable 1', 'Variable 2', 'Correlation (r)', 'P-Value', 'Effect Size']))
            elements.append(Spacer(1, 10))
        
        # ANOVA
        anova = stats_results.get('anova', pd.DataFrame())
        if not anova.empty:
            elements.append(Paragraph("ANOVA Results", self.styles['SubSectionHeader']))
            elements.append(self._build_test_table(anova.head(10), ['Numeric Variable', 'Grouping Variable', 'F-Statistic', 'P-Value', 'Eta-Squared (η²)']))
            elements.append(Spacer(1, 10))
        
        # VIF
        vif = stats_results.get('vif', pd.DataFrame())
        if not vif.empty:
            elements.append(Paragraph("Multicollinearity (VIF)", self.styles['SubSectionHeader']))
            elements.append(self._build_test_table(vif, ['Feature', 'VIF', 'VIF Risk']))
            
            if 'Condition Number (κ)' in vif.columns:
                kappa = vif['Condition Number (κ)'].iloc[0]
                k_risk = vif['κ Risk'].iloc[0]
                elements.append(Spacer(1, 6))
                elements.append(Paragraph(f"Condition Number κ = {kappa:.2f} ({k_risk} risk)", self.styles['CaptionText']))
            
            elements.append(Spacer(1, 10))
        
        # OLS
        ols_stats = stats_results.get('ols_model_stats', {})
        if ols_stats:
            elements.append(Paragraph(f"OLS Regression — Target: {ols_stats.get('Target', '')}", self.styles['SubSectionHeader']))
            
            ols_info = [
                ['Metric', 'Value'],
                ['R²', str(ols_stats.get('R²', ''))],
                ['Adj. R²', str(ols_stats.get('Adj. R²', ''))],
                ['F-Statistic', str(ols_stats.get('F-Statistic', ''))],
                ['F P-Value', str(ols_stats.get('F P-Value', ''))],
                ['AIC', str(ols_stats.get('AIC', ''))],
                ['BIC', str(ols_stats.get('BIC', ''))],
                ['N', str(ols_stats.get('N', ''))],
            ]
            
            ols_table = Table(ols_info, colWidths=[2.5*inch, 2.5*inch])
            ols_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), self.colors['medium_navy']),
                ('TEXTCOLOR', (0, 0), (-1, 0), self.colors['white']),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, self.colors['border']),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [self.colors['white'], self.colors['bg_light']]),
                ('PADDING', (0, 0), (-1, -1), 6),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ]))
            elements.append(ols_table)
        
        return elements
    
    def _build_test_table(self, df: pd.DataFrame, columns: list) -> Table:
        """Build a styled table from a DataFrame."""
        valid_cols = [c for c in columns if c in df.columns]
        data = [valid_cols]
        
        for _, row in df[valid_cols].head(12).iterrows():
            data.append([str(row[c]) for c in valid_cols])
        
        col_width = 6.0 / len(valid_cols)
        table = Table(data, colWidths=[col_width*inch] * len(valid_cols))
        
        style_cmds = [
            ('BACKGROUND', (0, 0), (-1, 0), self.colors['medium_navy']),
            ('TEXTCOLOR', (0, 0), (-1, 0), self.colors['white']),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('FONTSIZE', (0, 1), (-1, -1), 7),
            ('GRID', (0, 0), (-1, -1), 0.5, self.colors['border']),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [self.colors['white'], self.colors['bg_light']]),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]
        
        # Color p-values
        if 'P-Value' in valid_cols:
            p_idx = valid_cols.index('P-Value')
            for i, row in enumerate(data[1:], start=1):
                try:
                    p_val = float(row[p_idx])
                    if p_val < 0.05:
                        style_cmds.append(('BACKGROUND', (p_idx, i), (p_idx, i), self.colors['green']))
                        style_cmds.append(('TEXTCOLOR', (p_idx, i), (p_idx, i), self.colors['white']))
                except:
                    pass
        
        table.setStyle(TableStyle(style_cmds))
        return table
    
    # ── VISUALIZATIONS ──────────────────────────────────────────
    
    def _visualizations_section(self, figures: list) -> list:
        elements = []
        
        elements.append(Paragraph("Visualizations", self.styles['SectionHeader']))
        elements.append(self.GoldDivider(450))
        elements.append(Spacer(1, 16))
        
        chart_names = [
            "Distribution Analysis",
            "Correlation Heatmap",
            "Box Plot Analysis",
        ]
        
        for i, fig in enumerate(figures[:6]):
            try:
                name = chart_names[i] if i < len(chart_names) else f"Chart {i+1}"
                
                elements.append(Paragraph(name, self.styles['SubSectionHeader']))
                
                # Convert matplotlib figure to image
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                img_buffer.seek(0)
                
                img = Image(img_buffer, width=5.5*inch, height=3.5*inch)
                elements.append(img)
                elements.append(Paragraph(f"Figure {i+1}: {name}", self.styles['CaptionText']))
                elements.append(Spacer(1, 14))
                
                img_buffer.close()
                
            except Exception as e:
                elements.append(Paragraph(f"Chart {i+1} could not be rendered", self.styles['CaptionText']))
        
        return elements
    
    # ── INSIGHTS ────────────────────────────────────────────────
    
    def _insights_section(self, insights: dict) -> list:
        elements = []
        
        elements.append(Paragraph("AI-Powered Insights & Recommendations", self.styles['SectionHeader']))
        elements.append(self.GoldDivider(450))
        elements.append(Spacer(1, 16))
        
        if not insights:
            elements.append(Paragraph("No insights available. Run statistical tests first.", self.styles['BodyText']))
            return elements
        
        # Highlights
        elements.append(Paragraph("✅ Key Highlights", self.styles['SubSectionHeader']))
        for h in insights.get('highlights', [])[:5]:
            elements.append(Paragraph(f"• {h}", self.styles['InsightBullet']))
        
        elements.append(Spacer(1, 12))
        
        # Warnings
        if insights.get('warnings'):
            elements.append(Paragraph("⚠ Warnings & Alerts", self.styles['SubSectionHeader']))
            for w in insights['warnings'][:5]:
                elements.append(Paragraph(f"• {w}", self.styles['WarningText']))
            elements.append(Spacer(1, 12))
        
        # Summary
        if insights.get('summary'):
            elements.append(Paragraph("📋 Summary Notes", self.styles['SubSectionHeader']))
            for s in insights['summary']:
                elements.append(Paragraph(f"• {s}", self.styles['InsightBullet']))
            elements.append(Spacer(1, 12))
        
        # Recommendation box
        if insights.get('recommendation'):
            elements.append(Paragraph("💡 Final Recommendation", self.styles['SubSectionHeader']))
            
            rec_box = Table([
                [Paragraph(insights['recommendation'], self.styles['BodyText'])]
            ], colWidths=[6*inch])
            rec_box.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), self.colors['gold_light']),
                ('BOX', (0, 0), (-1, -1), 2, self.colors['gold']),
                ('PADDING', (0, 0), (-1, -1), 16),
            ]))
            elements.append(rec_box)
        
        return elements


# Helper function (same as app.py)
def _corr_col(df):
    return 'Correlation (r)' if 'Correlation (r)' in df.columns else 'Correlation'
