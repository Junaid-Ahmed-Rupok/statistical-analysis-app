from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing, Rect, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
import io
import pandas as pd
import datetime

class PDFReportGenerator:
    """
    Professional PDF report generator for statistical analysis results.
    Uses io.BytesIO for in-memory generation.
    """
    
    def __init__(self):
        self.styles = self._create_styles()
        self.buffer = io.BytesIO()
        
    def _create_styles(self):
        """Create custom styles for the report."""
        styles = getSampleStyleSheet()
        
        # Custom styles
        styles.add(ParagraphStyle(
            name='NavyTitle',
            parent=styles['Title'],
            fontSize=28,
            textColor=colors.HexColor('#0A2342'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        styles.add(ParagraphStyle(
            name='GoldSubtitle',
            parent=styles['Normal'],
            fontSize=16,
            textColor=colors.HexColor('#F0A500'),
            spaceAfter=20,
            alignment=TA_CENTER
        ))
        
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#1B4F72'),
            spaceAfter=12,
            spaceBefore=20,
            borderPadding=10,
        ))
        
        styles.add(ParagraphStyle(
            name='BodyText2',
            parent=styles['Normal'],
            fontSize=11,
            leading=16,
            textColor=colors.HexColor('#1A1A2E'),
            alignment=TA_JUSTIFY
        ))
        
        styles.add(ParagraphStyle(
            name='MetricValue',
            parent=styles['Normal'],
            fontSize=24,
            textColor=colors.HexColor('#0A2342'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        styles.add(ParagraphStyle(
            name='MetricLabel',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor('#6C757D'),
            alignment=TA_CENTER
        ))
        
        return styles
    
    def _create_header_footer(self, canvas, doc):
        """Create page header and footer."""
        canvas.saveState()
        
        # Footer
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#6C757D'))
        canvas.drawString(
            inch, 0.5*inch,
            f"StatsPro | Confidential | Page {canvas.getPageNumber()}"
        )
        
        # Header line
        canvas.setStrokeColor(colors.HexColor('#F0A500'))
        canvas.setLineWidth(1)
        canvas.line(inch, 10.8*inch, 7.5*inch, 10.8*inch)
        
        canvas.restoreState()
    
    def generate(
        self,
        df: pd.DataFrame,
        overview: dict,
        stats_results: dict,
        insights: dict,
        figures: list,
        cleaned_df: pd.DataFrame = None
    ) -> io.BytesIO:
        """
        Generate complete PDF report.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            topMargin=0.8*inch,
            bottomMargin=0.8*inch,
            leftMargin=0.8*inch,
            rightMargin=0.8*inch
        )
        
        story = []
        
        # Title Page
        story.extend(self._create_title_page(overview))
        story.append(PageBreak())
        
        # Executive Summary
        story.extend(self._create_executive_summary(overview, insights))
        story.append(PageBreak())
        
        # Data Overview
        story.extend(self._create_data_overview(overview))
        story.append(PageBreak())
        
        # Statistical Tests
        story.extend(self._create_statistical_tests(stats_results))
        story.append(PageBreak())
        
        # Insights
        story.extend(self._create_insights_section(insights))
        
        doc.build(story, onFirstPage=self._create_header_footer,
                 onLaterPages=self._create_header_footer)
        
        buffer.seek(0)
        return buffer
    
    def _create_title_page(self, overview: dict):
        """Create report title page."""
        elements = []
        
        elements.append(Spacer(1, 2*inch))
        elements.append(Paragraph("StatsPro Analysis Report", 
                                self.styles['NavyTitle']))
        elements.append(Paragraph("Professional Statistical Analysis",
                                self.styles['GoldSubtitle']))
        
        elements.append(Spacer(1, 1*inch))
        
        # Dataset info
        info_data = [
            ['Dataset Information', ''],
            ['Rows', str(overview['rows'])],
            ['Columns', str(overview['columns'])],
            ['Quality Score', f"{overview['quality_score']:.1f}/100"],
            ['Generated', datetime.datetime.now().strftime('%Y-%m-%d %H:%M')],
        ]
        
        table = Table(info_data, colWidths=[3*inch, 3*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0A2342')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('SPAN', (0, 0), (-1, 0)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F4F6F9')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E0E6ED')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F4F6F9')]),
            ('PADDING', (0, 0), (-1, -1), 12),
        ]))
        
        elements.append(table)
        
        return elements
    
    def _create_executive_summary(self, overview: dict, insights: dict):
        """Create executive summary page."""
        elements = []
        
        elements.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        elements.append(Spacer(1, 20))
        
        # Key metrics in 3-column layout
        metrics = [
            ['Data Quality', 'Completeness', 'Readiness'],
            [
                f"{overview['quality_score']:.0f}/100",
                f"{100 - overview['missing_percentage']:.1f}%",
                f"{insights.get('readiness_score', 0):.0f}/100"
            ]
        ]
        
        table = Table(metrics, colWidths=[2.2*inch]*3)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0A2342')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#0A2342')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTSIZE', (0, 1), (-1, -1), 18),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FFF3CD')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E0E6ED')),
            ('PADDING', (0, 0), (-1, -1), 15),
        ]))
        
        elements.append(table)
        elements.append(Spacer(1, 30))
        
        # Key findings
        elements.append(Paragraph("Key Findings", self.styles['SectionHeader']))
        
        for highlight in insights.get('highlights', [])[:5]:
            elements.append(Paragraph(f"✓ {highlight}", self.styles['BodyText2']))
            elements.append(Spacer(1, 5))
        
        return elements
    
    def _create_data_overview(self, overview: dict):
        """Create data overview section."""
        elements = []
        
        elements.append(Paragraph("Data Overview", self.styles['SectionHeader']))
        elements.append(Spacer(1, 15))
        
        # Missing values table
        if not overview['missing_table'].empty:
            missing_df = overview['missing_table']
            table_data = [['Column', 'Missing Count', 'Missing %']]
            
            for _, row in missing_df.iterrows():
                table_data.append([
                    row['Column'],
                    str(row['Missing Count']),
                    f"{row['Missing Percentage']:.1f}%"
                ])
            
            table = Table(table_data, colWidths=[2.5*inch, 2*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B4F72')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E0E6ED')),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), 
                 [colors.white, colors.HexColor('#F4F6F9')]),
                ('PADDING', (0, 0), (-1, -1), 8),
            ]))
            
            elements.append(table)
        
        return elements
    
    def _create_statistical_tests(self, stats_results: dict):
        """Create statistical tests summary."""
        elements = []
        
        elements.append(Paragraph("Statistical Tests Results", 
                                self.styles['SectionHeader']))
        elements.append(Spacer(1, 15))
        
        # Normality test results
        if not stats_results['normality'].empty:
            elements.append(Paragraph("Normality Tests", 
                                    self.styles['Heading3']))
            
            norm_df = stats_results['normality']
            table_data = [['Column', 'Test', 'P-Value', 'Result']]
            
            for _, row in norm_df.iterrows():
                table_data.append([
                    row['Column'],
                    row['Test'],
                    f"{row['P-Value']:.4f}",
                    'Normal' if row['Normal'] else 'Non-normal'
                ])
            
            table = Table(table_data, colWidths=[1.5*inch]*4)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B4F72')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E0E6ED')),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            
            elements.append(table)
            elements.append(Spacer(1, 15))
        
        # Correlation summary
        if not stats_results['pearson_correlation'].empty:
            elements.append(Paragraph("Top Correlations", 
                                    self.styles['Heading3']))
            
            corr_df = stats_results['pearson_correlation'].nlargest(5, 'Correlation')
            table_data = [['Variable 1', 'Variable 2', 'Correlation', 'P-Value']]
            
            for _, row in corr_df.iterrows():
                table_data.append([
                    row['Variable 1'],
                    row['Variable 2'],
                    f"{row['Correlation']:.3f}",
                    f"{row['P-Value']:.4f}"
                ])
            
            table = Table(table_data, colWidths=[1.5*inch]*4)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1B4F72')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E0E6ED')),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            
            elements.append(table)
        
        return elements
    
    def _create_insights_section(self, insights: dict):
        """Create insights and recommendations section."""
        elements = []
        
        elements.append(Paragraph("AI-Powered Insights", 
                                self.styles['SectionHeader']))
        elements.append(Spacer(1, 15))
        
        # Warnings
        if insights.get('warnings'):
            elements.append(Paragraph("⚠️ Warnings", 
                                    self.styles['Heading3']))
            for warning in insights['warnings']:
                elements.append(Paragraph(f"• {warning}", 
                                        self.styles['BodyText2']))
            elements.append(Spacer(1, 10))
        
        # Highlights
        if insights.get('highlights'):
            elements.append(Paragraph("✅ Key Highlights", 
                                    self.styles['Heading3']))
            for highlight in insights['highlights']:
                elements.append(Paragraph(f"• {highlight}", 
                                        self.styles['BodyText2']))
            elements.append(Spacer(1, 10))
        
        # Recommendation
        if insights.get('recommendation'):
            elements.append(Paragraph("💡 Recommendation", 
                                    self.styles['Heading3']))
            elements.append(Paragraph(insights['recommendation'], 
                                    self.styles['BodyText2']))
        
        return elements