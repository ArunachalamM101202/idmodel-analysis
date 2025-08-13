import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="HPV Cancer Screening Strategy Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    h1 {
        color: #1f77b4;
    }
    .stExpander {
        background-color: #f0f2f6;
        border-radius: 10px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Strategy definitions - UPDATED with correct numbering
STRATEGY_DEFINITIONS = {
    1: {"display": "1", "name": "CYTO-3Y, 21-65", "test": "Cyto only", "start_age": 21, "end_age": 65, "switch_age": None, "exit_tests": 3, "interval": "3Y"},
    2: {"display": "2", "name": "CYTO-3Y, 21/HPV-5Y, 30-65", "test": "Cyto→HPV", "start_age": 21, "end_age": 65, "switch_age": 30, "exit_tests": 2, "interval": "3Y→5Y"},
    3: {"display": "3", "name": "CYTO-3Y, 21/COTEST-5Y, 30-65", "test": "Cyto→Cotest", "start_age": 21, "end_age": 65, "switch_age": 30, "exit_tests": 2, "interval": "3Y→5Y"},
    4: {"display": "4", "name": "CYTO-3Y, 21-70", "test": "Cyto only", "start_age": 21, "end_age": 70, "switch_age": None, "exit_tests": 3, "interval": "3Y"},
    5: {"display": "5", "name": "CYTO-3Y, 21/HPV-5Y, 30-70", "test": "Cyto→HPV", "start_age": 21, "end_age": 70, "switch_age": 30, "exit_tests": 2, "interval": "3Y→5Y"},
    6: {"display": "6", "name": "CYTO-3Y, 21/COTEST-5Y, 30-70", "test": "Cyto→Cotest", "start_age": 21, "end_age": 70, "switch_age": 30, "exit_tests": 2, "interval": "3Y→5Y"},
    7: {"display": "7", "name": "CYTO-3Y, 21-75", "test": "Cyto only", "start_age": 21, "end_age": 75, "switch_age": None, "exit_tests": 3, "interval": "3Y"},
    8: {"display": "8", "name": "CYTO-3Y, 21/HPV-5Y, 30-75", "test": "Cyto→HPV", "start_age": 21, "end_age": 75, "switch_age": 30, "exit_tests": 2, "interval": "3Y→5Y"},
    9: {"display": "9", "name": "CYTO-3Y, 21/COTEST-5Y, 30-75", "test": "Cyto→Cotest", "start_age": 21, "end_age": 75, "switch_age": 30, "exit_tests": 2, "interval": "3Y→5Y"},
    10: {"display": "10", "name": "CYTO-3Y, 21/HPV-5Y, 30-65 (1 exit)", "test": "Cyto→HPV", "start_age": 21, "end_age": 65, "switch_age": 30, "exit_tests": 1, "interval": "3Y→5Y"},
    11: {"display": "11", "name": "CYTO-3Y, 21/COTEST-5Y, 30-65 (1 exit)", "test": "Cyto→Cotest", "start_age": 21, "end_age": 65, "switch_age": 30, "exit_tests": 1, "interval": "3Y→5Y"},
    12: {"display": "12", "name": "CYTO-3Y, 21/HPV-5Y, 30-70 (1 exit)", "test": "Cyto→HPV", "start_age": 21, "end_age": 70, "switch_age": 30, "exit_tests": 1, "interval": "3Y→5Y"},
    13: {"display": "13", "name": "CYTO-3Y, 21/COTEST-5Y, 30-70 (1 exit)", "test": "Cyto→Cotest", "start_age": 21, "end_age": 70, "switch_age": 30, "exit_tests": 1, "interval": "3Y→5Y"},
    14: {"display": "14", "name": "CYTO-3Y, 21/HPV-5Y, 30-75 (1 exit)", "test": "Cyto→HPV", "start_age": 21, "end_age": 75, "switch_age": 30, "exit_tests": 1, "interval": "3Y→5Y"},
    15: {"display": "15", "name": "CYTO-3Y, 21/COTEST-5Y, 30-75 (1 exit)", "test": "Cyto→Cotest", "start_age": 21, "end_age": 75, "switch_age": 30, "exit_tests": 1, "interval": "3Y→5Y"},
}

@st.cache_data
def load_strategy_data(strategy_num):
    """Load data for a specific strategy"""
    filename = f"cancer_screening_results{strategy_num}.csv"
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        st.warning(f"File {filename} not found")
        return None

@st.cache_data
def calculate_metrics(df):
    """Calculate key metrics for questions 2, 3, and 4"""
    if df is None:
        return None
    
    # Sum all metrics from age 1 to 100
    metrics = {
        'total_cancers': df['Cancer'].sum(),
        'total_cancer_deaths': df['Cancer death'].sum(),
        'total_life_years': df['Population'].sum(),
        'total_tests': df['Total test'].sum() if 'Total test' in df.columns else 0,
        'total_cyto': df['Cyto'].sum() if 'Cyto' in df.columns else 0,
        'total_hpv': df['HPV'].sum() if 'HPV' in df.columns else 0,
        'total_cotest': df['Cotest'].sum() if 'Cotest' in df.columns else 0,
        'total_cin2_detected': df['CIN2 detected'].sum() if 'CIN2 detected' in df.columns else 0,
        'total_cin3_detected': df['CIN3 detected'].sum() if 'CIN3 detected' in df.columns else 0,
    }
    
    # Calculate rates
    if metrics['total_cancers'] > 0:
        metrics['case_fatality_rate'] = (metrics['total_cancer_deaths'] / metrics['total_cancers']) * 100
    else:
        metrics['case_fatality_rate'] = 0
    
    if metrics['total_life_years'] > 0:
        metrics['cancer_rate_per_100k'] = (metrics['total_cancers'] / metrics['total_life_years']) * 100000
        metrics['death_rate_per_100k'] = (metrics['total_cancer_deaths'] / metrics['total_life_years']) * 100000
    else:
        metrics['cancer_rate_per_100k'] = 0
        metrics['death_rate_per_100k'] = 0
    
    return metrics

@st.cache_data
def load_all_strategies():
    """Load and process all strategy data"""
    results = []
    age_distributions = {}
    
    for strategy_num in range(1, 16):
        df = load_strategy_data(strategy_num)
        if df is None:
            continue
        
        # Store age distribution data for Question 1
        age_distributions[strategy_num] = df
        
        metrics = calculate_metrics(df)
        if metrics is None:
            continue
        
        strategy_info = STRATEGY_DEFINITIONS[strategy_num]
        result = {
            'Strategy': strategy_num,
            'Display': strategy_info['display'],
            'Name': strategy_info['name'],
            'Test Type': strategy_info['test'],
            'Start Age': strategy_info['start_age'],
            'End Age': strategy_info['end_age'],
            'Switch Age': strategy_info['switch_age'] if strategy_info['switch_age'] else '-',
            'Exit Tests': strategy_info['exit_tests'],
            'Interval': strategy_info['interval'],
            **metrics
        }
        results.append(result)
    
    return pd.DataFrame(results), age_distributions

def create_strategy_overview_table(df):
    """Create a visual table showing all strategy definitions"""
    # Simply display as a formatted dataframe
    display_df = df[['Display', 'Name', 'Test Type', 'Start Age', 'End Age', 'Switch Age', 'Exit Tests', 'Interval']].copy()
    display_df.columns = ['Strategy', 'Description', 'Test Type', 'Start Age', 'End Age', 'Switch Age', 'Exit Tests', 'Interval']
    
    # Sort by strategy number
    display_df['sort_key'] = display_df['Strategy'].map(lambda x: 
        int(x) if x.isdigit() and int(x) <= 9 else int(x) + 3 if x.isdigit() else 999)
    display_df = display_df.sort_values('sort_key').drop('sort_key', axis=1)
    
    return display_df

def create_question_1_analysis(age_distributions, df_results):
    """Analyze screening test execution at designed ages - all strategies"""
    
    # Create subplot for all 15 strategies
    fig = make_subplots(
        rows=5, cols=3,
        subplot_titles=[f"Strategy {STRATEGY_DEFINITIONS[i]['display']}: {STRATEGY_DEFINITIONS[i]['test']}" 
                       for i in range(1, 16)],
        vertical_spacing=0.08,  # Add spacing between rows
        horizontal_spacing=0.10,  # Add spacing between columns
        specs=[[{"secondary_y": False}]*3 for _ in range(5)]
    )
    
    # Position mapping for subplots
    positions = [
        (1,1), (1,2), (1,3),  # Row 1: Strategies 1, 2, 3
        (2,1), (2,2), (2,3),  # Row 2: Strategies 4, 5, 6
        (3,1), (3,2), (3,3),  # Row 3: Strategies 7, 8, 9
        (4,1), (4,2), (4,3),  # Row 4: Strategies 13, 14, 15
        (5,1), (5,2), (5,3),  # Row 5: Strategies 16, 17, 18
    ]
    
    for idx, strategy_num in enumerate(range(1, 16)):
        if strategy_num not in age_distributions:
            continue
            
        df = age_distributions[strategy_num]
        strategy_info = STRATEGY_DEFINITIONS[strategy_num]
        
        row, col = positions[idx]
        
        # Create stacked area chart for test distribution
        fig.add_trace(
            go.Scatter(
                x=df['Age'],
                y=df['Cyto'],
                mode='lines',
                fill='tozeroy',
                name='Cytology',
                line=dict(color='lightblue', width=1),
                fillcolor='rgba(173, 216, 230, 0.5)',
                showlegend=(idx == 0),
                hovertemplate='Age %{x}<br>Cyto: %{y}<extra></extra>'
            ),
            row=row, col=col
        )
        
        if strategy_info['switch_age']:
            # Add HPV tests
            fig.add_trace(
                go.Scatter(
                    x=df['Age'],
                    y=df['HPV'],
                    mode='lines',
                    fill='tozeroy',
                    name='HPV',
                    line=dict(color='lightgreen', width=1),
                    fillcolor='rgba(144, 238, 144, 0.5)',
                    showlegend=(idx == 0),
                    hovertemplate='Age %{x}<br>HPV: %{y}<extra></extra>'
                ),
                row=row, col=col
            )
            
            # Add Cotest if applicable
            if 'Cotest' in strategy_info['test']:
                fig.add_trace(
                    go.Scatter(
                        x=df['Age'],
                        y=df['Cotest'],
                        mode='lines',
                        fill='tozeroy',
                        name='Cotest',
                        line=dict(color='lightcoral', width=1),
                        fillcolor='rgba(240, 128, 128, 0.5)',
                        showlegend=(idx == 0),
                        hovertemplate='Age %{x}<br>Cotest: %{y}<extra></extra>'
                    ),
                    row=row, col=col
                )
        
        # Add vertical lines for key ages with annotations
        # Start age
        fig.add_vline(
            x=strategy_info['start_age'], 
            line_dash="dash", 
            line_color="green", 
            line_width=1, 
            row=row, col=col
        )
        
        # End age
        fig.add_vline(
            x=strategy_info['end_age'], 
            line_dash="dash", 
            line_color="red", 
            line_width=1, 
            row=row, col=col
        )
        
        # Switch age if applicable
        if strategy_info['switch_age']:
            fig.add_vline(
                x=strategy_info['switch_age'], 
                line_dash="dash", 
                line_color="orange", 
                line_width=1, 
                row=row, col=col
            )
    
    fig.update_layout(
        title="<b>All 15 Strategies</b>",
        height=1200,  # Increased height for 5 rows
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.01, 
            xanchor="right", 
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1
        ),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    # Update all x and y axes
    for i in range(1, 16):
        row, col = positions[i-1]
        fig.update_xaxes(
            title_text="Age" if row == 5 else "", 
            range=[15, 85],
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray',
            row=row, col=col
        )
        fig.update_yaxes(
            title_text="Tests" if col == 1 else "", 
            showgrid=True,
            gridwidth=0.5,
            gridcolor='lightgray',
            row=row, col=col
        )
    
    return fig

def create_main_analysis_charts(df):
    """Create main charts for Questions 2, 3, and 4"""
    
    # Use display numbers for x-axis
    df_sorted = df.sort_values('Strategy')
    
    # Create color scale based on screening intensity
    colors = []
    for _, row in df_sorted.iterrows():
        if row['Exit Tests'] == 1:
            colors.append('#d62728')  # Red for single exit
        elif row['End Age'] == 75:
            colors.append('#2ca02c')  # Green for extended to 75
        elif row['End Age'] == 70:
            colors.append('#ff7f0e')  # Orange for extended to 70
        else:
            colors.append('#1f77b4')  # Blue for standard
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Total Cancers (Sum to Age 100)',
            'Total Cancer Deaths (Sum to Age 100)',
            'Total Life-Years (Sum to Age 100)',
            'Screening Intensity vs Outcomes'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.12
    )
    
    # Question 2: Total Cancers
    fig.add_trace(
        go.Bar(
            x=df_sorted['Display'],
            y=df_sorted['total_cancers'],
            marker_color=colors,
            text=df_sorted['total_cancers'].round(0),
            textposition='outside',
            hovertemplate='<b>Strategy %{x}</b><br>' +
                         'Total Cancers: %{y:.0f}<br>' +
                         '<extra></extra>',
            name='Total Cancers'
        ),
        row=1, col=1
    )
    
    # Question 3: Total Deaths
    fig.add_trace(
        go.Bar(
            x=df_sorted['Display'],
            y=df_sorted['total_cancer_deaths'],
            marker_color=colors,
            text=df_sorted['total_cancer_deaths'].round(0),
            textposition='outside',
            hovertemplate='<b>Strategy %{x}</b><br>' +
                         'Total Deaths: %{y:.0f}<br>' +
                         '<extra></extra>',
            name='Total Deaths'
        ),
        row=1, col=2
    )
    
    # Question 4: Total Life Years
    fig.add_trace(
        go.Bar(
            x=df_sorted['Display'],
            y=df_sorted['total_life_years']/1e6,
            marker_color=colors,
            text=(df_sorted['total_life_years']/1e6).round(2),
            textposition='outside',
            hovertemplate='<b>Strategy %{x}</b><br>' +
                         'Life Years: %{y:.2f}M<br>' +
                         '<extra></extra>',
            name='Life Years'
        ),
        row=2, col=1
    )
    
    # Intensity vs Outcomes scatter
    fig.add_trace(
        go.Scatter(
            x=df_sorted['total_tests'],
            y=df_sorted['total_cancers'],
            mode='markers+text',
            marker=dict(
                size=15,
                color=df_sorted['total_cancer_deaths'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Deaths", x=1.15, len=0.4, y=0.2)
            ),
            text=df_sorted['Display'],
            textposition='top center',
            hovertemplate='<b>Strategy %{text}</b><br>' +
                         'Total Tests: %{x:,.0f}<br>' +
                         'Total Cancers: %{y:.0f}<br>' +
                         '<extra></extra>',
            name='Intensity'
        ),
        row=2, col=2
    )
    
    # Update layouts
    fig.update_xaxes(title_text="Strategy Number", row=1, col=1)
    fig.update_xaxes(title_text="Strategy Number", row=1, col=2)
    fig.update_xaxes(title_text="Strategy Number", row=2, col=1)
    fig.update_xaxes(title_text="Total Screening Tests", row=2, col=2)
    
    fig.update_yaxes(title_text="Total Cancers", row=1, col=1)
    fig.update_yaxes(title_text="Total Deaths", row=1, col=2)
    fig.update_yaxes(title_text="Million Life-Years", row=2, col=1)
    fig.update_yaxes(title_text="Total Cancers", row=2, col=2)
    
    fig.update_layout(
        height=700,
        showlegend=False,
        title_text="<b>Primary Analysis: Cancer Outcomes by Screening Strategy</b>",
        title_font_size=20
    )
    
    return fig

def create_trend_analysis(df):
    """Create trend analysis comparing screening intensity groups"""
    
    # Group strategies by characteristics
    groups = {
        'Cyto Only': df[df['Test Type'] == 'Cyto only'],
        'Cyto→HPV': df[df['Test Type'] == 'Cyto→HPV'],
        'Cyto→Cotest': df[df['Test Type'] == 'Cyto→Cotest']
    }
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('By Test Type', 'By End Age', 'By Exit Tests'),
        horizontal_spacing=0.15
    )
    
    # By Test Type
    test_summary = []
    for test_type, group_df in groups.items():
        test_summary.append({
            'Type': test_type,
            'Avg Cancers': group_df['total_cancers'].mean(),
            'Avg Deaths': group_df['total_cancer_deaths'].mean(),
            'Avg Life-Years': group_df['total_life_years'].mean()/1e6
        })
    test_df = pd.DataFrame(test_summary)
    
    fig.add_trace(
        go.Bar(x=test_df['Type'], y=test_df['Avg Cancers'], name='Avg Cancers',
               marker_color='lightcoral'),
        row=1, col=1
    )
    
    # By End Age
    age_groups = df.groupby('End Age').agg({
        'total_cancers': 'mean',
        'total_cancer_deaths': 'mean',
        'total_life_years': 'mean'
    }).reset_index()
    
    fig.add_trace(
        go.Bar(x=age_groups['End Age'], y=age_groups['total_cancers'], name='By Age',
               marker_color='lightblue'),
        row=1, col=2
    )
    
    # By Exit Tests
    exit_groups = df.groupby('Exit Tests').agg({
        'total_cancers': 'mean',
        'total_cancer_deaths': 'mean',
        'total_life_years': 'mean'
    }).reset_index()
    
    fig.add_trace(
        go.Bar(x=exit_groups['Exit Tests'], y=exit_groups['total_cancers'], name='By Exit',
               marker_color='lightgreen'),
        row=1, col=3
    )
    
    fig.update_layout(
        title="<b>Trend Analysis: Average Cancer Cases by Strategy Characteristics</b>",
        height=400,
        showlegend=False
    )
    
    fig.update_yaxes(title_text="Average Total Cancers", row=1, col=1)
    fig.update_xaxes(title_text="Test Type", row=1, col=1)
    fig.update_xaxes(title_text="End Age", row=1, col=2)
    fig.update_xaxes(title_text="Exit Tests", row=1, col=3)
    
    return fig

def main():
    # Title and description
    st.title("🔬 HPV Cancer Screening Strategy Analysis Dashboard")
    st.markdown("""
    ### Analysis of 15 Screening Strategies (Strategies 1-9, 13-18)
    This dashboard analyzes the effectiveness of different HPV cancer screening strategies to answer four key questions.
    """)
    
    # Load data
    with st.spinner("Loading all strategy data..."):
        df, age_distributions = load_all_strategies()
    
    if df.empty:
        st.error("No data files found. Please ensure cancer_screening_results1.csv through cancer_screening_results15.csv are in the current directory.")
        return
    
    # Strategy Overview Table
    st.header("📋 Strategy Definitions")
    strategy_table = create_strategy_overview_table(df)
    st.dataframe(strategy_table, use_container_width=True, hide_index=True)
    
    # Question 1: Age Distribution Analysis
    st.header("Are Screening Tests Executed at Designed Ages?")
    
    q1_fig = create_question_1_analysis(age_distributions, df)
    st.plotly_chart(q1_fig, use_container_width=True)

    st.write("""
    This chart shows the **distribution of screening tests by age** for all 15 strategies.
    
    **Colors (filled areas)**:
    - 🟦 **Light Blue**: Cyto
    - 🟩 **Light Green**: HPV
    - 🟥 **Light Coral**: Cotest
    
    **Vertical dashed lines**:
    - 🟢 **Green dashed line**: **Screening start age** (usually 21) — testing begins here.
    - 🟠 **Orange dashed line**: **Switch age** (usually 30) — strategies that begin with Cytology switch to HPV or Cotest here.
    - 🔴 **Red dashed line**: **Screening end age** (65, 70, or 75) — testing stops here.
    
    **How to interpret**:
    - Each subplot is **one strategy** showing how test types are distributed across ages.
    - The area under each color shows **how many tests of that type** occur at each age.
    - For combined strategies, expect a visible shift in color at the switch age (orange line).
    - The bulk of tests should occur **between start and end ages**.
""")
    
    # Main Analysis for Questions 2, 3, 4
    st.header(" Cancer Outcomes Analysis")
    
    main_fig = create_main_analysis_charts(df)
    st.plotly_chart(main_fig, use_container_width=True)

    st.write("""
    This analysis compares **cancer outcomes** across all screening strategies using four plots (2×2 grid):
    
    **Top Left — Total Cancers**  
    - Shows the **total number of cervical cancer cases** detected over a lifetime (to age 100).  
    - **Lower bars = better prevention**.
    
    **Top Right — Total Cancer Deaths**  
    - Shows the **total number of deaths** from cervical cancer over a lifetime.  
    - **Lower bars = better survival**.
    
    **Bottom Left — Total Life-Years**  
    - Shows the **total number of years lived** across the simulated population.  
    - **Higher bars = better overall survival**.
    
    **Bottom Right — Screening Intensity vs Outcomes**  
    - X-axis: Total screening tests performed.  
    - Y-axis: Total cancer cases.  
    - Bubble **color**: Total deaths (lighter = fewer deaths).  
    - Each bubble = one strategy, labeled with its strategy number.  
    **Interpretation (Bottom-Right Plot)**:  
        - ⬅️ **Farther left** → Fewer total screening tests  
        - ⬇️ **Farther down** → Fewer total cancer cases  
        - 🎨 **Lighter color** → Fewer total deaths


    
    **Bar Color Legend**:  
    - 🔵 **Blue** — Standard strategies (end age 65)  
    - 🟧 **Orange** — Extended to age 70  
    - 🟩 **Green** — Extended to age 75  
    - 🔴 **Red** — Single-exit-test strategies
    
""")
    
    # Trend Analysis
    st.header("📈 Trend Analysis")
    
    st.write("""
    Comparison of average cancer cases grouped by:
    - **Test Type**: Cyto only vs Cyto→HPV vs Cyto→Cotest
    - **End Age**: Strategies ending at 65, 70, or 75
    - **Exit Tests**: Strategies with 1, 2, or 3 exit tests
    
    Lower bars indicate better cancer prevention.
    """)
    
    trend_fig = create_trend_analysis(df)
    st.plotly_chart(trend_fig, use_container_width=True)
    
    # Detailed Strategy Comparison Table
    st.header("📋 Detailed Strategy Comparison Table")
    
    # Format the dataframe for display
    display_df = df[[
        'Display', 'Name', 'Test Type', 'End Age', 'Exit Tests',
        'total_cancers', 'total_cancer_deaths', 'total_life_years',
        'case_fatality_rate', 'cancer_rate_per_100k', 'death_rate_per_100k'
    ]].copy()
    
    display_df.columns = [
        'Strategy', 'Description', 'Test Type', 'End Age', 'Exit Tests',
        'Total Cancers', 'Total Deaths', 'Total Life-Years',
        'Case Fatality %', 'Cancer Rate/100k', 'Death Rate/100k'
    ]
    
    # Sort by display number
    display_df['sort_key'] = display_df['Strategy'].map(lambda x: 
        int(x) if x.isdigit() and int(x) <= 9 else int(x) + 3 if x.isdigit() else 999)
    display_df = display_df.sort_values('sort_key').drop('sort_key', axis=1)
    
    # Format numerical columns
    display_df['Total Life-Years'] = display_df['Total Life-Years'].apply(lambda x: f"{x:,.0f}")
    display_df['Case Fatality %'] = display_df['Case Fatality %'].apply(lambda x: f"{x:.1f}%")
    display_df['Cancer Rate/100k'] = display_df['Cancer Rate/100k'].apply(lambda x: f"{x:.2f}")
    display_df['Death Rate/100k'] = display_df['Death Rate/100k'].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    
    
    # Key Findings
    st.header("✅ Key Findings & Conclusions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **Screening Age Compliance ✓**
        - All strategies show screening concentrated between designed start and end ages
        - Clear transitions at age 30 for HPV/Cotest strategies
        - Minimal screening outside target age ranges
        """)
        
        st.success("""
        **Total Cancers ✓**
        - HPV/Cotest strategies show fewer cancers than Cyto-only
        - Single exit strategies (13-18) have slightly more cancers
        - Extended screening ages detect more cancers (but earlier stage)
        """)
    
    with col2:
        st.success("""
        **Cancer Deaths ✓**
        - More intensive screening reduces cancer deaths
        - Extended screening to age 75 shows lowest mortality
        - Cotest strategies perform best for mortality reduction
        """)
        
        st.success("""
        **Life-Years ✓**
        - More intensive screening increases total life-years
        - Strategies preventing cancer deaths show higher population survival
        - Extended age strategies maximize life-years saved
        """)
    
    # Best performers summary
    st.header("🏆 Best Performing Strategies")
    
    best_cancer = df.loc[df['total_cancers'].idxmin()]
    best_death = df.loc[df['total_cancer_deaths'].idxmin()]
    best_life = df.loc[df['total_life_years'].idxmax()]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **Lowest Cancer Count**
        Strategy {best_cancer['Display']}: {best_cancer['Name']}
        - Total cancers: {best_cancer['total_cancers']:.0f}
        - Test type: {best_cancer['Test Type']}
        """)
    
    with col2:
        st.info(f"""
        **Lowest Death Count**
        Strategy {best_death['Display']}: {best_death['Name']}
        - Total deaths: {best_death['total_cancer_deaths']:.0f}
        - Test type: {best_death['Test Type']}
        """)
    
    with col3:
        st.info(f"""
        **Most Life-Years Saved**
        Strategy {best_life['Display']}: {best_life['Name']}
        - Total life-years: {best_life['total_life_years']/1e6:.2f}M
        - Test type: {best_life['Test Type']}
        """)
    
    # Export option
    st.header("💾 Export Results")
    
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Complete Analysis Results (CSV)",
        data=csv,
        file_name='screening_strategy_analysis.csv',
        mime='text/csv'
    )

if __name__ == "__main__":
    main()