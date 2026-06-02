"""
INV-9956-101 Tracker 分析 Web 应用
晚期去势转移前列腺癌试验

核心分析维度：
- 受试者状态全景与流向
- 筛选失败原因分析
- EOT 原因分析
- Phase 维度统计

入组绩效分析：
- 月度筛选与入组趋势
- B组停止前后对比（2026-04-07）

中心绩效分析：
- 各中心筛选活跃度
- 筛选成功率与失败率对比

AR状态分析（核心）：
- 剂量组 × AR 状态统计
- AR+ 率计算

All subjects cohorts 分析：
- A/B/C 分组统计

时间轴分析：
- ICF → C1D1 筛选周期分析
- 各中心筛选周期对比
- 在组时间游泳图（分5mg/10mg，按AR和EOT状态分色）

安装依赖：pip install streamlit pandas matplotlib openpyxl seaborn
运行：streamlit run tracker_app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import font_manager
import numpy as np
from datetime import datetime
import io
import zipfile
import seaborn as sns
import platform
import os
from matplotlib.patches import Patch

sns.set_style("whitegrid")

def setup_chinese_font():
    FONT_DIR = os.path.join(os.path.dirname(__file__), 'fonts')
    FONT_FILE = os.path.join(FONT_DIR, 'NotoSansCJKsc-Regular.otf')
    
    if os.path.exists(FONT_FILE):
        font_manager.fontManager.addfont(FONT_FILE)
        font_prop = font_manager.FontProperties(fname=FONT_FILE)
        plt.rcParams['font.family'] = font_prop.get_name()
        return font_prop
    
    system = platform.system()
    if system == "Windows":
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    elif system == "Darwin":
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'STHeiti', 'Arial Unicode MS']
    else:
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    return None

CHINESE_FONT = setup_chinese_font()
plt.rcParams['axes.unicode_minus'] = False

COLOR_PALETTES = {
    'primary': ['#48CAE4', '#90E0EF', '#00B4D8', '#0077B6', '#023E8A'],
    'health': ['#95D5B2', '#74C69D', '#52B788', '#40916C', '#2D6A4F'],
    'accent': ['#FFB703', '#FB8500', '#F77F00', '#DC2F02', '#6A040F'],
    'purple': ['#E0AAFF', '#C77DFF', '#9D4EDD', '#7B2CBF', '#5A189A'],
    'warm': ['#FFCDB2', '#FFB4A2', '#FF8BA7', '#FF6D60', '#F72585'],
}

SWIMMER_COLORS = {
    'AR-_Treatment': '#95D5B2',
    'AR-_EOT': '#DC2F02',
    'AR+_Treatment': '#90E0EF',
    'AR+_EOT': '#FB8500',
}

saved_figures = {}
report_sections = []


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return buf.getvalue()


def df_to_markdown_table(dataframe):
    if dataframe is None or len(dataframe) == 0:
        return ""
    lines = []
    cols = list(dataframe.columns)
    if isinstance(dataframe.index, pd.MultiIndex):
        idx_names = dataframe.index.names
        idx_names = [n if n else "Index" for n in idx_names]
        header = "| " + " | ".join(idx_names + cols) + " |"
        sep = "| " + " | ".join(["---"] * (len(idx_names) + len(cols))) + " |"
        lines.append(header)
        lines.append(sep)
        for idx_tuple, row in dataframe.iterrows():
            idx_strs = [str(v) for v in idx_tuple]
            val_strs = [str(v) for v in row.values]
            lines.append("| " + " | ".join(idx_strs + val_strs) + " |")
    else:
        idx_name = dataframe.index.name if dataframe.index.name else ""
        header = "| " + idx_name + " | " + " | ".join(str(c) for c in cols) + " |"
        sep = "| --- | " + " | ".join(["---"] * len(cols)) + " |"
        lines.append(header)
        lines.append(sep)
        for idx_val, row in dataframe.iterrows():
            val_strs = [str(v) for v in row.values]
            lines.append("| " + str(idx_val) + " | " + " | ".join(val_strs) + " |")
    return "\n".join(lines) + "\n\n"


def series_to_markdown_table(series, col_name="人数"):
    if series is None or len(series) == 0:
        return ""
    lines = []
    lines.append(f"| {series.index.name or ''} | {col_name} |")
    lines.append("| --- | --- |")
    for idx_val, val in series.items():
        lines.append(f"| {idx_val} | {val} |")
    return "\n".join(lines) + "\n\n"


def get_swimmer_colors(time_data_sorted):
    colors = []
    for idx in time_data_sorted.index:
        ar_status = time_data_sorted.loc[idx, 'AR+/-'] if 'AR+/-' in time_data_sorted.columns else 'Unknown'
        eot_date = time_data_sorted.loc[idx, 'EOT date'] if 'EOT date' in time_data_sorted.columns else None
        status = time_data_sorted.loc[idx, 'Status'] if 'Status' in time_data_sorted.columns else ''
        is_eot = pd.notna(eot_date) or status == 'EOT'

        if ar_status == 'AR+':
            colors.append(SWIMMER_COLORS['AR+_EOT'] if is_eot else SWIMMER_COLORS['AR+_Treatment'])
        elif ar_status == 'AR-':
            colors.append(SWIMMER_COLORS['AR-_EOT'] if is_eot else SWIMMER_COLORS['AR-_Treatment'])
        else:
            colors.append(SWIMMER_COLORS['AR+_Treatment'] if not is_eot else SWIMMER_COLORS['AR+_EOT'])
    return colors


SWIMMER_LEGEND = [
    Patch(facecolor=SWIMMER_COLORS['AR+_Treatment'], edgecolor='white', label='AR+ 进行中（浅蓝色）'),
    Patch(facecolor=SWIMMER_COLORS['AR+_EOT'], edgecolor='white', label='AR+ EOT（橙色）'),
    Patch(facecolor=SWIMMER_COLORS['AR-_Treatment'], edgecolor='white', label='AR- 进行中（绿色）'),
    Patch(facecolor=SWIMMER_COLORS['AR-_EOT'], edgecolor='white', label='AR- EOT（红色）'),
]


def draw_swimmer_chart(ax, time_data_sorted, dose_label):
    colors = get_swimmer_colors(time_data_sorted)
    subject_ids = time_data_sorted['Subject ID'].values if 'Subject ID' in time_data_sorted.columns else [f'P{i+1}' for i in range(len(time_data_sorted))]
    days = time_data_sorted['在组天数'].values

    bars = ax.barh(range(len(time_data_sorted)), days, color=colors, alpha=0.9, edgecolor='white')
    ax.set_xlabel('在组天数', fontweight='bold')
    ax.set_ylabel('受试者', fontweight='bold')
    ax.set_title(f'{dose_label} 剂量组 - 在组时间', fontweight='bold', pad=10)
    ax.set_yticks(range(len(time_data_sorted)))
    ax.set_yticklabels(subject_ids, fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    for i, (bar, day) in enumerate(zip(bars, days)):
        ax.text(day + 1, i, f'{int(day)}天', va='center', fontweight='bold', fontsize=8)


st.set_page_config(
    page_title="INV-9956-101 Tracker 分析工具",
    page_icon="📊",
    layout="wide"
)

st.title("📊 INV-9956-101 Tracker 分析工具")

st.sidebar.header("定义")
st.sidebar.markdown("""
- **筛选**: 有 Date ICF
- **入组**: 有 C1D1 日期
- **筛选失败**: Status = 'Screen Failure'
- **筛选成功率**: 入组数 / 筛选数 × 100%
- **B组停止日期**: 2026-04-07
""")

uploaded_file = st.file_uploader(
    "上传 INV-9956 Tracker 数据文件",
    type=['csv', 'xlsx', 'xls'],
    help="支持 CSV 或 Excel 格式"
)

if uploaded_file is None:
    st.info("👆 请上传文件开始分析")
    st.stop()

try:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    if file_ext == 'csv':
        df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
    else:
        df = pd.read_excel(uploaded_file)
    st.success(f"成功读取文件：{uploaded_file.name}（{len(df)} 行）")
except Exception as e:
    st.error(f"读取文件失败：{e}")
    st.stop()

# ========== 数据清洗 ==========
st.header("数据清洗")
df.columns = df.columns.str.strip()

if 'Status' in df.columns:
    df['Status'] = df['Status'].str.strip()

if 'AR+/-' in df.columns:
    df['AR+/-'] = df['AR+/-'].str.strip()

if 'Cohort(mg)' in df.columns:
    df['Cohort(mg)'] = df['Cohort(mg)'].replace('Pending', np.nan)

date_columns = ['Date ICF', 'C1D1', 'Latest Date', 'EOT date']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

df['筛选'] = df['Date ICF'].notna()
df['入组'] = df['C1D1'].notna()
df['筛选失败'] = df['Status'] == 'Screen Failure' if 'Status' in df.columns else False

st.info(f"清洗后数据：{len(df)} 行 × {len(df.columns)} 列")

with st.expander("查看原始数据"):
    st.dataframe(df.head(10), width="stretch")

# ========== 1. Phase & 剂量组筛选入组表 ==========
st.header("1. Phase & 剂量组筛选入组表")
if 'Phase' in df.columns and 'Cohort(mg)' in df.columns:
    dose_data = df[df['Cohort(mg)'].notna()].copy()
    if len(dose_data) > 0:
        dose_data['Phase_Dose'] = dose_data['Phase'].apply(lambda x: f"Phase {int(x)}") + ' - ' + dose_data['Cohort(mg)'].astype(int).astype(str) + 'mg'

        phase_dose_stats = dose_data.groupby('Phase_Dose').agg({
            '筛选': 'sum', '入组': 'sum', '筛选失败': 'sum'
        }).astype(int)

        phase_dose_stats['_phase'] = phase_dose_stats.index.str.extract(r'Phase (\d)').astype(int)
        phase_dose_stats['_dose'] = phase_dose_stats.index.str.extract(r'(\d+)mg').astype(int)
        phase_dose_stats = phase_dose_stats.sort_values(['_phase', '_dose'])
        phase_dose_stats = phase_dose_stats.drop(['_phase', '_dose'], axis=1)

        phase_dose_stats['筛选成功率'] = (phase_dose_stats['入组'] / phase_dose_stats['筛选'] * 100).round(1)
        st.dataframe(phase_dose_stats, width="stretch")
        report_sections.append(("1. Phase & 剂量组筛选入组表", df_to_markdown_table(phase_dose_stats)))

# ========== 2. 中心筛选入组表（含总计）==========
st.header("2. 中心筛选入组表")
site_stats_display = None
if 'Site No.' in df.columns:
    site_stats = df.groupby('Site No.').agg({
        '筛选': 'sum', '入组': 'sum', '筛选失败': 'sum'
    }).astype(int).sort_values('筛选', ascending=False)
    site_stats['筛选成功率'] = (site_stats['入组'] / site_stats['筛选'] * 100).round(1)

    total = site_stats.sum(numeric_only=True)
    total['筛选成功率'] = (total['入组'] / total['筛选'] * 100).round(1)
    site_stats.loc['总计'] = total

    st.dataframe(site_stats, width="stretch")
    site_stats_display = site_stats.copy()
    report_sections.append(("2. 中心筛选入组表", df_to_markdown_table(site_stats_display)))
    
    site_stats_for_chart = site_stats.drop('总计', errors='ignore')
    
    if len(site_stats_for_chart) > 0:
        fig_site, ax_site = plt.subplots(figsize=(12, 6))
        
        x = range(len(site_stats_for_chart))
        width = 0.35
        
        bars1 = ax_site.bar([i - width/2 for i in x], 
                           site_stats_for_chart['筛选'], 
                           width, 
                           label='筛选人数', 
                           color=COLOR_PALETTES['primary'][2], 
                           alpha=0.8)
        
        bars2 = ax_site.bar([i + width/2 for i in x], 
                           site_stats_for_chart['入组'], 
                           width, 
                           label='入组人数', 
                           color=COLOR_PALETTES['health'][2], 
                           alpha=0.8)
        
        ax_site.set_xlabel('中心编号', fontweight='bold')
        ax_site.set_ylabel('人数', fontweight='bold')
        ax_site.set_title('各中心筛选与入组人数对比', fontweight='bold', pad=15)
        ax_site.set_xticks(x)
        ax_site.set_xticklabels(site_stats_for_chart.index.astype(str), rotation=45, ha='right')
        ax_site.legend(frameon=True, shadow=True)
        ax_site.grid(True, alpha=0.3, axis='y')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax_site.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig_site)
        saved_figures[f"各中心筛选入组对比_{datetime.now().strftime('%Y%m%d_%H%M')}.png"] = fig_to_png(fig_site)
        plt.close(fig_site)

# ========== 3. 剂量组 × AR 状态分析 ==========
st.header("3. 剂量组 × AR 状态分析")
if 'Cohort(mg)' in df.columns and 'AR+/-' in df.columns:
    dose_ar_data = df[df['Cohort(mg)'].notna()].copy()

    if len(dose_ar_data) > 0:
        dose_ar_pivot = pd.crosstab(dose_ar_data['Cohort(mg)'], dose_ar_data['AR+/-'])

        ar_order = ['AR+', 'AR-', 'Not Done']
        for col in ar_order:
            if col not in dose_ar_pivot.columns:
                dose_ar_pivot[col] = 0
        dose_ar_pivot = dose_ar_pivot[ar_order]

        dose_ar_pivot['总计'] = dose_ar_pivot.sum(axis=1)
        dose_ar_pivot['AR+率(%)'] = (dose_ar_pivot['AR+'] / dose_ar_pivot['总计'] * 100).round(1)

        st.dataframe(dose_ar_pivot.fillna(0).astype(int), width="stretch")
        report_sections.append(("3. 剂量组 × AR 状态分析", df_to_markdown_table(dose_ar_pivot.fillna(0).astype(int))))

# ========== 4. 剂量组 × 状态分布表 ==========
st.header("4. 剂量组 × 状态分布表")
if 'Status' in df.columns:
    with st.expander("查看 Status 列的所有唯一值"):
        status_counts = df['Status'].value_counts(dropna=False)
        st.dataframe(status_counts.to_frame('人数'), width="stretch")
    
    df_status = df.copy()
    df_status['Status'] = df_status['Status'].fillna('Unknown')
    
    if 'Cohort(mg)' in df_status.columns:
        def format_dose(val):
            if pd.isna(val):
                return 'Pending'
            try:
                return f'{int(val)}mg'
            except:
                return str(val)
        
        df_status['剂量组'] = df_status['Cohort(mg)'].apply(format_dose)
    else:
        df_status['剂量组'] = 'Pending'
    
    status_pivot_simple = pd.crosstab(df_status['剂量组'], df_status['Status'])
    
    status_pivot_simple['总计'] = status_pivot_simple.sum(axis=1)
    
    status_pivot_simple.loc['总计'] = status_pivot_simple.sum(axis=0)
    
    st.dataframe(status_pivot_simple, width="stretch")
    report_sections.append(("4. 剂量组 × 状态分布表", df_to_markdown_table(status_pivot_simple)))
else:
    st.info("📊 未找到 'Status' 列")

# ========== 5. Phase × 剂量 × 状态分布表 ==========
st.header("5. Phase × 剂量 × 状态分布表")
if 'Phase' in df.columns and 'Status' in df.columns and 'Cohort(mg)' in df.columns:
    dose_status_data = df[df['Cohort(mg)'].notna()].copy()

    if len(dose_status_data) > 0:
        dose_status_data['Phase_Label'] = dose_status_data['Phase'].apply(lambda x: f"Phase {int(x)}")
        dose_status_data['Dose'] = dose_status_data['Cohort(mg)'].astype(int).astype(str) + 'mg'

        status_pivot = pd.crosstab(
            [dose_status_data['Phase_Label'], dose_status_data['Dose']],
            dose_status_data['Status']
        )

        status_order = ['In Screening', 'Screen Failure', 'In Treatment', 'EOT']
        for col in status_order:
            if col not in status_pivot.columns:
                status_pivot[col] = 0
        status_pivot = status_pivot[status_order]

        st.dataframe(status_pivot.fillna(0).astype(int), width="stretch")
        report_sections.append(("4. Phase × 剂量 × 状态分布表", df_to_markdown_table(status_pivot.fillna(0).astype(int))))

# ========== 5. 月度筛选入组组合图 ==========
st.header("5. 月度筛选入组趋势（组合图）")
if 'Date ICF' in df.columns and 'C1D1' in df.columns:
    df['ICF_Month'] = df['Date ICF'].dt.to_period('M').dt.to_timestamp()
    df['C1D1_Month'] = df['C1D1'].dt.to_period('M').dt.to_timestamp()

    monthly_screened = df[df['筛选']].groupby('ICF_Month').size()
    monthly_enrolled = df[df['入组']].groupby('C1D1_Month').size()

    all_months = pd.Index(sorted(set(monthly_screened.index) | set(monthly_enrolled.index)))
    monthly_data = pd.DataFrame({
        '筛选': monthly_screened.reindex(all_months, fill_value=0),
        '入组': monthly_enrolled.reindex(all_months, fill_value=0)
    })

    monthly_data['累计筛选'] = monthly_data['筛选'].cumsum()
    monthly_data['累计入组'] = monthly_data['入组'].cumsum()

    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = range(len(monthly_data))
    width = 0.35

    bars1 = ax1.bar([i - width/2 for i in x], monthly_data['筛选'], width,
                     label='每月筛选', color=COLOR_PALETTES['primary'][1], alpha=0.8)
    bars2 = ax1.bar([i + width/2 for i in x], monthly_data['入组'], width,
                     label='每月入组', color=COLOR_PALETTES['health'][1], alpha=0.8)

    ax1.set_xlabel('月份', fontweight='bold')
    ax1.set_ylabel('每月人数', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.strftime('%Y-%m') for m in monthly_data.index], rotation=45)

    ax2 = ax1.twinx()
    line1 = ax2.plot(x, monthly_data['累计筛选'], 'o-', color=COLOR_PALETTES['primary'][3],
                     linewidth=2.5, markersize=8, label='累计筛选')
    line2 = ax2.plot(x, monthly_data['累计入组'], 's-', color=COLOR_PALETTES['health'][2],
                     linewidth=2.5, markersize=8, label='累计入组')
    ax2.set_ylabel('累计人数', fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, shadow=True)

    plt.title('月度筛选与入组趋势', fontweight='bold', pad=15)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(fig)
    saved_figures[f"月度筛选与入组趋势_{datetime.now().strftime('%Y%m%d_%H%M')}.png"] = fig_to_png(fig)
    plt.close(fig)

    monthly_display = monthly_data[['筛选', '入组', '累计筛选', '累计入组']].copy()
    monthly_display.index = [m.strftime('%Y-%m') for m in monthly_display.index]
    report_sections.append(("5. 月度筛选入组趋势", df_to_markdown_table(monthly_display.astype(int))))

# ========== 6. 筛选失败原因分析 ==========
st.header("6. 筛选失败原因分析")
screen_failures = df[df['筛选失败'] == True]

if len(screen_failures) > 0 and 'Screen Failure Reason' in screen_failures.columns:
    failure_counts = screen_failures['Screen Failure Reason'].value_counts()

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(failure_counts.to_frame('人数'), width="stretch")

    with col2:
        fig_pie, ax_pie = plt.subplots(figsize=(6, 5))
        colors = [COLOR_PALETTES['accent'][i % len(COLOR_PALETTES['accent'])]
                 for i in range(len(failure_counts))]
        wedges, texts, autotexts = ax_pie.pie(failure_counts.values, labels=failure_counts.index,
                                                  autopct='%1.1f%%', colors=colors,
                                                  shadow=True, startangle=90)
        ax_pie.set_title('筛选失败原因分布', fontweight='bold', pad=15)
        for wedge in wedges:
            wedge.set_edgecolor('white')
            wedge.set_linewidth(1.5)
        plt.tight_layout()
        st.pyplot(fig_pie)
        saved_figures[f"筛选失败原因分布_{datetime.now().strftime('%Y%m%d_%H%M')}.png"] = fig_to_png(fig_pie)
        plt.close(fig_pie)

    report_sections.append(("6. 筛选失败原因分析", series_to_markdown_table(failure_counts, '人数')))
else:
    st.info("📊 当前数据中没有筛选失败的受试者")

# ========== 7. B组停止前后对比 ==========
st.header("8. B组停止前后筛选成功率对比（2026-04-07）")
if 'Date ICF' in df.columns:
    cutoff_date = pd.Timestamp('2026-04-07')

    df['时间段'] = df['Date ICF'].apply(
        lambda x: 'B组停止前' if pd.notna(x) and x < cutoff_date else 'B组停止后'
    )

    period_comparison = df.groupby('时间段').agg({
        '筛选': 'sum', '入组': 'sum', '筛选失败': 'sum'
    }).astype(int)
    period_comparison['筛选成功率'] = (period_comparison['入组'] / period_comparison['筛选'] * 100).round(1)
    period_comparison['筛选失败率'] = (period_comparison['筛选失败'] / period_comparison['筛选'] * 100).round(1)

    st.dataframe(period_comparison, width="stretch")
    report_sections.append(("8. B组停止前后对比", df_to_markdown_table(period_comparison)))

# ========== 8. All Subjects Cohorts 分析 ==========
st.header("8. All Subjects Cohorts 分析")
if 'All subjects cohorts' in df.columns and 'Cohort(mg)' in df.columns:
    enrolled_data = df[df['入组'] == True].copy()
    
    if len(enrolled_data) > 0:
        cohort_pivot = pd.crosstab(
            enrolled_data['Cohort(mg)'],
            enrolled_data['All subjects cohorts']
        )
        
        cohort_pivot['总计'] = cohort_pivot.sum(axis=1)
        
        cohort_pivot.loc['总计'] = cohort_pivot.sum(axis=0)
        
        st.dataframe(cohort_pivot, width="stretch")
        report_sections.append(("8. All Subjects Cohorts 分析", df_to_markdown_table(cohort_pivot)))
    else:
        st.info("📊 当前没有入组的受试者")
else:
    st.info("📊 未找到 'All subjects cohorts' 或 'Cohort(mg)' 列")

# ========== 9. 在组时间游泳图（分5mg/10mg）==========
st.header("10. 在组时间游泳图（按剂量分组）")
if 'C1D1' in df.columns and 'Latest Date' in df.columns and 'Cohort(mg)' in df.columns:
    time_data = df[(df['C1D1'].notna()) & (df['Latest Date'].notna())].copy()

    if len(time_data) > 0:
        time_data['在组天数'] = (time_data['Latest Date'] - time_data['C1D1']).dt.days
        time_data = time_data[time_data['在组天数'] >= 0]

        time_5mg = time_data[time_data['Cohort(mg)'].astype(float) == 5]
        time_10mg = time_data[time_data['Cohort(mg)'].astype(float) == 10]

        col1, col2, col3 = st.columns(3)
        col1.metric("总样本数", len(time_data))
        col2.metric("5mg 样本", len(time_5mg))
        col3.metric("10mg 样本", len(time_10mg))

        if len(time_data) > 0:
            if len(time_5mg) > 0 or len(time_10mg) > 0:
                if len(time_5mg) > 0 and len(time_10mg) > 0:
                    fig = plt.figure(figsize=(14, max(8, max(len(time_5mg), len(time_10mg)) * 0.35)))

                    ax1 = plt.subplot(2, 1, 1)
                    plt.subplots_adjust(hspace=0.3)

                    time_5mg_sorted = time_5mg.sort_values('在组天数', ascending=False)
                    draw_swimmer_chart(ax1, time_5mg_sorted, '5mg')

                    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
                    time_10mg_sorted = time_10mg.sort_values('在组天数', ascending=False)
                    draw_swimmer_chart(ax2, time_10mg_sorted, '10mg')

                    fig.legend(handles=SWIMMER_LEGEND, loc='center right', bbox_to_anchor=(0.98, 0.5), frameon=True, shadow=True, fontsize=9)

                    plt.tight_layout()
                    st.pyplot(fig)
                    saved_figures[f"在组时间游泳图_{datetime.now().strftime('%Y%m%d_%H%M')}.png"] = fig_to_png(fig)
                    plt.close(fig)

                elif len(time_5mg) > 0:
                    fig, ax = plt.subplots(figsize=(12, max(6, len(time_5mg) * 0.35)))
                    time_5mg_sorted = time_5mg.sort_values('在组天数', ascending=False)
                    draw_swimmer_chart(ax, time_5mg_sorted, '5mg')
                    ax.legend(handles=SWIMMER_LEGEND, loc='lower right', frameon=True, shadow=True, fontsize=9)

                    plt.tight_layout()
                    st.pyplot(fig)
                    saved_figures[f"在组时间游泳图_5mg_{datetime.now().strftime('%Y%m%d_%H%M')}.png"] = fig_to_png(fig)
                    plt.close(fig)

                elif len(time_10mg) > 0:
                    fig, ax = plt.subplots(figsize=(12, max(6, len(time_10mg) * 0.35)))
                    time_10mg_sorted = time_10mg.sort_values('在组天数', ascending=False)
                    draw_swimmer_chart(ax, time_10mg_sorted, '10mg')
                    ax.legend(handles=SWIMMER_LEGEND, loc='lower right', frameon=True, shadow=True, fontsize=9)

                    plt.tight_layout()
                    st.pyplot(fig)
                    saved_figures[f"在组时间游泳图_10mg_{datetime.now().strftime('%Y%m%d_%H%M')}.png"] = fig_to_png(fig)
                    plt.close(fig)

# ========== 10. EOT原因归类饼图 ==========
st.header("10. EOT 原因分布")
if 'EOT reson type' in df.columns and 'Status' in df.columns:
    eot_data = df[df['Status'] == 'EOT'].copy()

    if len(eot_data) > 0:
        eot_counts = eot_data['EOT reson type'].value_counts()

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(eot_counts.to_frame('人数'), width="stretch")

        with col2:
            fig, ax = plt.subplots(figsize=(6, 5))
            colors = [COLOR_PALETTES['primary'][i % len(COLOR_PALETTES['primary'])]
                     for i in range(len(eot_counts))]
            wedges, texts, autotexts = ax.pie(eot_counts.values, labels=eot_counts.index,
                                             autopct='%1.1f%%', colors=colors,
                                             shadow=True, startangle=90)
            ax.set_title('EOT 原因分布', fontweight='bold', pad=15)
            for wedge in wedges:
                wedge.set_edgecolor('white')
                wedge.set_linewidth(1.5)
            plt.tight_layout()
            st.pyplot(fig)
            saved_figures[f"EOT原因分布_{datetime.now().strftime('%Y%m%d_%H%M')}.png"] = fig_to_png(fig)
            plt.close(fig)

        report_sections.append(("10. EOT 原因分布", series_to_markdown_table(eot_counts, '人数')))
    else:
        st.info("📊 当前数据中没有 EOT 的受试者")

# ========== 11. 疗效数据（按剂量和AR状态）==========
st.header("12. 疗效数据（按剂量和AR状态）")
if 'Best Response' in df.columns and 'Cohort(mg)' in df.columns:
    response_data = df[df['Best Response'].notna()].copy()

    if len(response_data) > 0:
        response_data = response_data[response_data['Cohort(mg)'].notna()]

        response_data['分组'] = response_data['Cohort(mg)'].astype(int).astype(str) + 'mg - ' + response_data.get('AR+/-', pd.Series(['Unknown'] * len(response_data), index=response_data.index)).astype(str)

        if len(response_data) > 0:
            response_pivot = pd.crosstab(response_data['分组'], response_data['Best Response'])

            response_order = ['CR', 'PR', 'SD', 'PD', 'NE', 'uPD']
            for col in response_order:
                if col not in response_pivot.columns:
                    response_pivot[col] = 0
            existing_cols = [col for col in response_order if col in response_pivot.columns or col in response_data['Best Response'].values]
            response_pivot = response_pivot[[col for col in existing_cols if col in response_pivot.columns]]

            response_pivot['总计'] = response_pivot.sum(axis=1)

            if 'CR' in response_pivot.columns and 'PR' in response_pivot.columns:
                response_pivot['ORR(%)'] = (response_pivot['CR'] + response_pivot['PR']) / response_pivot['总计'] * 100
            else:
                response_pivot['ORR(%)'] = 0
                if 'PR' in response_pivot.columns:
                    response_pivot['ORR(%)'] = response_pivot['PR'] / response_pivot['总计'] * 100

            dcr_cols = []
            if 'CR' in response_pivot.columns:
                dcr_cols.append('CR')
            if 'PR' in response_pivot.columns:
                dcr_cols.append('PR')
            if 'SD' in response_pivot.columns:
                dcr_cols.append('SD')

            if dcr_cols:
                response_pivot['DCR(%)'] = response_pivot[dcr_cols].sum(axis=1) / response_pivot['总计'] * 100
            else:
                response_pivot['DCR(%)'] = 0

            response_pivot['ORR(%)'] = response_pivot['ORR(%)'].round(1)
            response_pivot['DCR(%)'] = response_pivot['DCR(%)'].round(1)

            st.dataframe(response_pivot, width="stretch")
            report_sections.append(("12. 疗效数据", df_to_markdown_table(response_pivot)))
    else:
        st.info("📊 当前数据中没有疗效评估结果")

# ========== 12. Protocol版本筛选入组 ==========
st.header("12. Protocol 版本筛选入组")
if 'Protocol' in df.columns:
    protocol_data = df[df['Protocol'].notna()].copy()

    if len(protocol_data) > 0:
        protocol_stats = protocol_data.groupby('Protocol').agg({
            '筛选': 'sum', '入组': 'sum', '筛选失败': 'sum'
        }).astype(int)
        protocol_stats['筛选成功率'] = (protocol_stats['入组'] / protocol_stats['筛选'] * 100).round(1)
        st.dataframe(protocol_stats, width="stretch")
        report_sections.append(("12. Protocol 版本筛选入组", df_to_markdown_table(protocol_stats)))
    else:
        st.info("📊 当前数据中没有 Protocol 版本信息")

# ========== 13. 来源分析 ==========
st.header("14. 来源分析")
if 'from' in df.columns:
    source_data = df[df['from'].notna()].copy()

    if len(source_data) > 0:
        source_stats = source_data.groupby('from').agg({
            '筛选': 'sum', '入组': 'sum', '筛选失败': 'sum'
        }).astype(int).sort_values('筛选', ascending=False)
        source_stats['筛选成功率'] = (source_stats['入组'] / source_stats['筛选'] * 100).round(1)
        source_stats['筛选失败率'] = (source_stats['筛选失败'] / source_stats['筛选'] * 100).round(1)
        st.dataframe(source_stats, width="stretch")
        report_sections.append(("13. 来源分析", df_to_markdown_table(source_stats)))
    else:
        st.info("📊 当前数据中没有来源信息")

# ========== 14. 国家筛选入组对比 ==========
st.header("14. 国家筛选入组对比")
if 'Country' in df.columns:
    country_data = df[df['Country'].notna()].copy()

    if len(country_data) > 0:
        country_stats = country_data.groupby('Country').agg({
            '筛选': 'sum', '入组': 'sum'
        }).astype(int)

        display_countries = ['China', 'USA']
        country_stats = country_stats.loc[[c for c in display_countries if c in country_stats.index]]

        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(country_stats, width="stretch")

        with col2:
            fig, ax = plt.subplots(figsize=(8, 6))

            x = range(len(country_stats))
            width = 0.35

            bars1 = ax.bar([i - width/2 for i in x], country_stats['筛选'], width,
                           label='筛选人数', color=COLOR_PALETTES['primary'][2], alpha=0.8)
            bars2 = ax.bar([i + width/2 for i in x], country_stats['入组'], width,
                           label='入组人数', color=COLOR_PALETTES['health'][2], alpha=0.8)

            ax.set_xlabel('国家', fontweight='bold')
            ax.set_ylabel('人数', fontweight='bold')
            ax.set_title('国家筛选入组对比', fontweight='bold', pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels(country_stats.index)
            ax.legend(frameon=True, shadow=True)
            ax.grid(True, alpha=0.3, axis='y')

            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{int(height)}', ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            st.pyplot(fig)
            saved_figures[f"国家筛选入组对比_{datetime.now().strftime('%Y%m%d_%H%M')}.png"] = fig_to_png(fig)
            plt.close(fig)

        report_sections.append(("15. 国家筛选入组对比", df_to_markdown_table(country_stats)))
    else:
        st.info("📊 当前数据中国家信息为空")
else:
    st.info("📊 未找到 'Country' 列")

# ========== 15. 筛选时长分析 ==========
st.header("16. 各中心筛选时长分析")
if 'Site No.' in df.columns and 'Date ICF' in df.columns and 'C1D1' in df.columns:
    screening_time = df[(df['Date ICF'].notna()) & (df['C1D1'].notna())].copy()

    if len(screening_time) > 0:
        screening_time['筛选天数'] = (screening_time['C1D1'] - screening_time['Date ICF']).dt.days
        screening_time = screening_time[screening_time['筛选天数'] >= 0]

        if len(screening_time) > 0:
            screening_stats = screening_time.groupby('Site No.').agg({
                '筛选天数': ['min', 'max', 'mean', 'count']
            }).round(1)
            screening_stats.columns = ['最短天数', '最长天数', '平均天数', '人数']
            screening_stats = screening_stats.sort_values('平均天数', ascending=False)

            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(screening_stats, width="stretch")

            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                sites = screening_stats.index
                colors = [COLOR_PALETTES['primary'][i % len(COLOR_PALETTES['primary'])]
                         for i in range(len(sites))]

                bars = ax.bar(range(len(sites)), screening_stats['平均天数'].values,
                             color=colors, alpha=0.8, edgecolor='white')

                ax.set_xlabel('中心', fontweight='bold')
                ax.set_ylabel('平均筛选天数', fontweight='bold')
                ax.set_title('各中心平均筛选时长（C1D1 - ICF）', fontweight='bold', pad=15)
                ax.set_xticks(range(len(sites)))
                ax.set_xticklabels(sites, rotation=45)

                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{height:.1f}', ha='center', va='bottom', fontweight='bold')

                plt.tight_layout()
                st.pyplot(fig)
                saved_figures[f"各中心平均筛选时长_{datetime.now().strftime('%Y%m%d_%H%M')}.png"] = fig_to_png(fig)
                plt.close(fig)

            report_sections.append(("16. 各中心筛选时长分析", df_to_markdown_table(screening_stats)))
        else:
            st.info("📊 没有有效的筛选时长数据")

# ========== 下载报告 ==========
st.header("下载完整报告")

def generate_markdown():
    lines = []
    lines.append("# INV-9956-101 Tracker 分析报告\n\n")
    lines.append(f"> 分析日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    lines.append(f"> 数据文件: {uploaded_file.name}\n\n")
    lines.append("## 定义\n\n")
    lines.append("- 筛选: 有 Date ICF\n")
    lines.append("- 入组: 有 C1D1 日期\n")
    lines.append("- 筛选失败: Status = 'Screen Failure'\n")
    lines.append("- 筛选成功率: 入组数 / 筛选数 × 100%\n")
    lines.append("- B组停止日期: 2026-04-07\n\n")
    for title, content in report_sections:
        lines.append(f"## {title}\n\n")
        lines.append(content)
    return ''.join(lines)

markdown_content = generate_markdown()

col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="📄 下载 Markdown 报告",
        data=markdown_content,
        file_name=f"inv9956_tracker_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown"
    )

with col2:
    if saved_figures:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr(
                f"inv9956_tracker_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                markdown_content.encode('utf-8')
            )
            for filename, png_bytes in saved_figures.items():
                zip_file.writestr(filename, png_bytes)

        zip_buffer.seek(0)
        st.download_button(
            label="📦 下载完整报告包（Markdown + 所有图表）",
            data=zip_buffer.getvalue(),
            file_name=f"inv9956_tracker_complete_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
            mime="application/zip"
        )
