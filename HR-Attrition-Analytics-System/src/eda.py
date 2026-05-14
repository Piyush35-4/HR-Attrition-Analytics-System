import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.preprocessing import load_data, drop_useless_columns, handle_missing_values

OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Set visual style
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 130


def load_clean_data():
    """Load and lightly clean the dataset for EDA (no encoding yet)."""
    df = load_data(DATA_PATH)
    df = drop_useless_columns(df)
    df = handle_missing_values(df)
    return df


def plot_attrition_distribution(df):
    """
    Chart 1: Attrition Distribution

    Shows how many employees left (Yes) vs stayed (No).
    This helps understand class imbalance — usually fewer employees leave.
    """
    counts = df['Attrition'].value_counts()
    colors = ['#2196F3', '#F44336']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    axes[0].bar(counts.index, counts.values, color=colors, width=0.4, edgecolor='white')
    axes[0].set_title('Employee Attrition Counts', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Attrition')
    axes[0].set_ylabel('Number of Employees')
    for i, (label, val) in enumerate(zip(counts.index, counts.values)):
        axes[0].text(i, val + 5, str(val), ha='center', fontsize=11)

    # Pie chart
    axes[1].pie(
        counts.values, labels=counts.index, autopct='%1.1f%%',
        colors=colors, startangle=140, textprops={'fontsize': 12}
    )
    axes[1].set_title('Attrition Proportion', fontsize=13, fontweight='bold')

    plt.suptitle('Attrition Distribution\n(Insight: Dataset is imbalanced — fewer employees leave)',
                 fontsize=11, style='italic', y=1.01)
    plt.tight_layout()
    _save('attrition_distribution.png')


def plot_age_vs_attrition(df):
    """
    Chart 2: Age vs Attrition

    Younger employees (20–35) tend to have higher attrition.
    This often correlates with career exploration and job-hopping behavior.
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(
        data=df, x='Age', hue='Attrition', bins=20,
        palette={'Yes': '#F44336', 'No': '#2196F3'},
        kde=True, element='step'
    )
    plt.title('Age Distribution by Attrition\n(Insight: Younger employees tend to leave more)',
              fontsize=13, fontweight='bold')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.legend(title='Attrition', labels=['Left (Yes)', 'Stayed (No)'])
    plt.tight_layout()
    _save('age_vs_attrition.png')


def plot_income_vs_attrition(df):
    """
    Chart 3: Monthly Income vs Attrition

    Employees with lower salaries are more likely to leave.
    This is one of the strongest predictors in the dataset.
    """
    plt.figure(figsize=(10, 5))
    sns.boxplot(
        data=df, x='Attrition', y='MonthlyIncome',
        palette={'Yes': '#F44336', 'No': '#2196F3'}
    )
    plt.title('Monthly Income vs Attrition\n(Insight: Lower earners are more likely to leave)',
              fontsize=13, fontweight='bold')
    plt.xlabel('Attrition')
    plt.ylabel('Monthly Income (USD)')
    plt.tight_layout()
    _save('income_vs_attrition.png')


def plot_overtime_vs_attrition(df):
    """
    Chart 4: OverTime vs Attrition

    Employees who work overtime show significantly higher attrition.
    Overtime is one of the top attrition predictors.
    """
    ct = df.groupby(['OverTime', 'Attrition']).size().unstack()
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    ct_pct.plot(kind='bar', stacked=True,
                color=['#2196F3', '#F44336'],
                figsize=(8, 5), edgecolor='white')
    plt.title('Overtime vs Attrition (%)\n(Insight: Overtime employees leave far more often)',
              fontsize=13, fontweight='bold')
    plt.xlabel('Works Overtime')
    plt.ylabel('Percentage (%)')
    plt.xticks(rotation=0)
    plt.legend(title='Attrition', labels=['Stayed (No)', 'Left (Yes)'])
    plt.tight_layout()
    _save('overtime_vs_attrition.png')


def plot_department_attrition(df):
    """
    Chart 5: Department-wise Attrition

    Helps HR understand which departments have the highest risk.
    Sales typically has higher attrition than R&D.
    """
    dept_attr = df[df['Attrition'] == 'Yes']['Department'].value_counts()
    dept_total = df['Department'].value_counts()
    dept_rate = (dept_attr / dept_total * 100).sort_values(ascending=False)

    plt.figure(figsize=(9, 5))
    bars = plt.bar(dept_rate.index, dept_rate.values,
                   color=['#F44336', '#FF9800', '#2196F3'], edgecolor='white')
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.3,
                 f'{bar.get_height():.1f}%',
                 ha='center', fontsize=11)
    plt.title('Attrition Rate by Department\n(Insight: Sales has the highest attrition rate)',
              fontsize=13, fontweight='bold')
    plt.xlabel('Department')
    plt.ylabel('Attrition Rate (%)')
    plt.tight_layout()
    _save('department_attrition.png')


def plot_correlation_heatmap(df):
    """
    Chart 6: Correlation Heatmap

    Shows how numerical features relate to each other.
    Features highly correlated with Attrition are the best predictors.
    (Attrition is encoded as 1=Yes, 0=No for this chart.)
    """
    df_encoded = df.copy()
    df_encoded['Attrition'] = df_encoded['Attrition'].map({'Yes': 1, 'No': 0})

    # Keep only numerical columns
    num_cols = df_encoded.select_dtypes(include='number').columns.tolist()
    corr = df_encoded[num_cols].corr()

    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Hide upper triangle
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f',
        cmap='RdYlGn', linewidths=0.5,
        annot_kws={'size': 7}, vmin=-1, vmax=1
    )
    plt.title('Feature Correlation Heatmap\n(Insight: Darker red = stronger negative correlation with the target)',
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save('correlation_heatmap.png')


def _save(filename):
    """Helper to save the current matplotlib figure."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    path = os.path.join(OUTPUTS_DIR, filename)
    plt.savefig(path, bbox_inches='tight', dpi=130)
    plt.close()
    print(f"[INFO] Chart saved → {path}")


def run_eda():
    """Run the complete EDA pipeline."""
    print("\n" + "="*50)
    print("  HR ATTRITION — EXPLORATORY DATA ANALYSIS")
    print("="*50)

    df = load_clean_data()

    print(f"\n[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Attrition breakdown:\n{df['Attrition'].value_counts()}\n")

    plot_attrition_distribution(df)
    plot_age_vs_attrition(df)
    plot_income_vs_attrition(df)
    plot_overtime_vs_attrition(df)
    plot_department_attrition(df)
    plot_correlation_heatmap(df)

    print("\n[INFO] All EDA charts saved to outputs/")


if __name__ == '__main__':
    run_eda()
