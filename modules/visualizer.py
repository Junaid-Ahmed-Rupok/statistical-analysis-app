import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

COLORS = ["#0A2342", "#F0A500", "#1B4F72", "#1E8449", "#C0392B", "#8E44AD"]

class Visualizer:
    
    def __init__(self):
        self.style_config()
    
    def style_config(self):
        plt.style.use('default')
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.color': '#F0F0F0',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'axes.labelsize': 11,
            'axes.labelcolor': '#6C757D',
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
        })
    
    def _add_watermark(self, fig):
        fig.text(0.95, 0.02, 'StatsPro', 
                fontsize=8, color='#D3D3D3', ha='right',
                style='italic', alpha=0.7)
    
    def histograms(self, df: pd.DataFrame, max_cols: int = 4):
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:12]
        n_cols = min(len(numeric_cols), max_cols)
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        fig.patch.set_facecolor('white')
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            data = df[col].dropna().values
            
            ax.hist(data, bins=30, color=COLORS[0], alpha=0.7, edgecolor='white', density=True)
            
            if len(data) > 1 and data.std() > 0:
                try:
                    kde = stats.gaussian_kde(data)
                    x_range = np.linspace(data.min(), data.max(), 200)
                    ax.plot(x_range, kde(x_range), color=COLORS[1], linewidth=2.5)
                except Exception:
                    pass
            
            ax.set_title(col, fontweight='bold', color=COLORS[0])
            ax.set_ylabel('Density')
        
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout(pad=2.0)
        self._add_watermark(fig)
        return fig
    
    def boxplots(self, df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:12]
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(numeric_cols) * 0.5)))
        fig.patch.set_facecolor('white')
        
        bp = ax.boxplot(
            [df[col].dropna().values for col in numeric_cols],
            labels=numeric_cols,
            patch_artist=True,
            medianprops={'color': COLORS[1], 'linewidth': 2.5},
            flierprops={'marker': 'o', 'markerfacecolor': COLORS[4], 'markersize': 4}
        )
        
        for patch in bp['boxes']:
            patch.set_facecolor(COLORS[0])
            patch.set_alpha(0.7)
        
        ax.set_title('Distribution Analysis', fontweight='bold', color=COLORS[0], pad=20)
        ax.set_ylabel('Values')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        self._add_watermark(fig)
        return fig
    
    def correlation_heatmap(self, df: pd.DataFrame, method: str = 'pearson'):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return None
        
        corr_matrix = df[numeric_cols].corr(method=method)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('white')
        
        colors = [COLORS[0], '#FFFFFF', COLORS[1]]
        cmap = plt.cm.colors.LinearSegmentedColormap.from_list('custom', colors)
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            annot=True,
            fmt='.2f',
            cmap=cmap,
            vmin=-1, vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8},
            ax=ax
        )
        
        ax.set_title(f'{method.capitalize()} Correlation Matrix', 
                    fontweight='bold', color=COLORS[0], pad=20)
        
        plt.tight_layout()
        self._add_watermark(fig)
        return fig
    
    def pairplot(self, df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
        
        if len(numeric_cols) < 2:
            return None
        
        g = sns.pairplot(
            df[numeric_cols],
            plot_kws={'alpha': 0.6, 'color': COLORS[0]},
            diag_kws={'color': COLORS[0], 'alpha': 0.7}
        )
        
        g.fig.patch.set_facecolor('white')
        g.fig.suptitle('Pairwise Relationships', 
                      fontweight='bold', color=COLORS[0], y=1.02, fontsize=14)
        
        self._add_watermark(g.fig)
        return g.fig
    
    def qq_plots(self, df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
        
        n_cols = min(len(numeric_cols), 3)
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        fig.patch.set_facecolor('white')
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            data = df[col].dropna().values
            
            if len(data) < 3 or data.std() == 0:
                ax.text(0.5, 0.5, 'Insufficient variance', ha='center', va='center',
                       transform=ax.transAxes, color='#999')
                ax.set_title(col, fontweight='bold', color=COLORS[0])
                continue
            
            stats.probplot(data, dist="norm", plot=ax)
            
            ax.get_lines()[0].set_markerfacecolor(COLORS[0])
            ax.get_lines()[0].set_markeredgecolor(COLORS[0])
            ax.get_lines()[0].set_alpha(0.6)
            ax.get_lines()[1].set_color(COLORS[1])
            ax.get_lines()[1].set_linewidth(2)
            
            ax.set_title(col, fontweight='bold', color=COLORS[0])
        
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        self._add_watermark(fig)
        return fig
    
    def vif_chart(self, vif_df: pd.DataFrame):
        if vif_df.empty:
            return None
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(vif_df) * 0.4)))
        fig.patch.set_facecolor('white')
        
        colors = []
        for vif in vif_df['VIF']:
            if vif < 5:
                colors.append(COLORS[3])
            elif vif < 10:
                colors.append(COLORS[1])
            else:
                colors.append(COLORS[4])
        
        bars = ax.barh(range(len(vif_df)), vif_df['VIF'], color=colors, alpha=0.8)
        ax.set_yticks(range(len(vif_df)))
        ax.set_yticklabels(vif_df['Feature'])
        
        for bar, vif in zip(bars, vif_df['VIF']):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2.,
                   f'{vif:.1f}', va='center', fontweight='bold')
        
        ax.axvline(x=5, color=COLORS[3], linestyle='--', alpha=0.5, label='Threshold (5)')
        ax.axvline(x=10, color=COLORS[4], linestyle='--', alpha=0.5, label='Threshold (10)')
        
        ax.set_xlabel('VIF Score')
        ax.set_title('Variance Inflation Factor (VIF)', 
                    fontweight='bold', color=COLORS[0], pad=20)
        ax.legend()
        
        plt.tight_layout()
        self._add_watermark(fig)
        return fig

    def bar_charts(self, df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:8]
        
        n_cols = min(len(numeric_cols), 3)
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.patch.set_facecolor('white')
        
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            value_counts = df[col].value_counts().nlargest(10)
            
            bars = ax.bar(range(len(value_counts)), value_counts.values, color=COLORS[0], alpha=0.8)
            
            for bar, val in zip(bars, value_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                       str(val), ha='center', va='bottom',
                       color=COLORS[1], fontweight='bold')
            
            ax.set_title(col, fontweight='bold', color=COLORS[0])
            ax.set_xticks([])
        
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        self._add_watermark(fig)
        return fig
    
    def violin_plots(self, df: pd.DataFrame):
        numeric_cols = df.select_dtypes(include=[np.number]).columns[:6]
        
        if len(numeric_cols) == 0:
            return None
        
        fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 3*len(numeric_cols)))
        fig.patch.set_facecolor('white')
        
        if len(numeric_cols) == 1:
            axes = [axes]
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            data = df[col].dropna().values
            
            parts = ax.violinplot(data, positions=[0], showmeans=True, showmedians=True)
            
            for pc in parts['bodies']:
                pc.set_facecolor(COLORS[idx % 2])
                pc.set_alpha(0.7)
            
            parts['cmeans'].set_color(COLORS[1])
            parts['cmedians'].set_color(COLORS[1])
            
            ax.set_title(col, fontweight='bold', color=COLORS[0])
            ax.set_ylabel('Values')
        
        plt.tight_layout()
        self._add_watermark(fig)
        return fig
    
    def missing_matrix(self, df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(12, max(6, len(df.columns) * 0.3)))
        fig.patch.set_facecolor('white')
        
        missing = df.isnull()
        
        ax.imshow(missing.T, aspect='auto', cmap='RdYlGn_r', interpolation='nearest', alpha=0.8)
        
        ax.set_xlabel('Row Index')
        ax.set_ylabel('Columns')
        ax.set_yticks(range(len(df.columns)))
        ax.set_yticklabels(df.columns)
        
        ax.set_title('Missing Value Matrix', fontweight='bold', color=COLORS[0], pad=20)
        
        plt.tight_layout()
        self._add_watermark(fig)
        return fig
