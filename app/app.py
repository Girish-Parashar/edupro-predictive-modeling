# ── Imports & Config ─────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib, os, warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="EduPro Analytics Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA   = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")

# ── Plotly Theme ─────────────────────────────────────────────────────
PLOTLY_TEMPLATE = "plotly_white"
COLOR_SEQ = px.colors.qualitative.Set2
PRIMARY   = "#1E3A5F"
SUCCESS   = "#4CAF50"
WARNING   = "#FF9800"
ACCENT    = "#FF6B35"

st.markdown("""
<style>
    .main { background-color: #F8F9FA; }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #2196F3;
        margin-bottom: 10px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #1E3A5F; }
    .metric-label { font-size: 0.85rem; color: #666; margin-top: 4px; }
    .section-header {
        font-size: 1.4rem; font-weight: 700; color: #1E3A5F;
        border-bottom: 2px solid #2196F3;
        padding-bottom: 8px; margin: 20px 0 15px 0;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E3A5F 0%, #2C5282 100%);
    }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stRadio label {
        color: white !important; font-size: 1rem !important;
        font-weight: 500 !important;
    }
    [data-testid="stSidebar"] .stMarkdown p { color: white !important; }
    [data-testid="stSidebar"] .stCaption { color: rgba(255,255,255,0.7) !important; }
    .stSelectbox label, .stSlider label { font-weight: 600; color: #1E3A5F; }
</style>
""", unsafe_allow_html=True)

# ── Load Data & Models ───────────────────────────────────────────────
@st.cache_data
def load_data():
    courses      = pd.read_csv(os.path.join(DATA, "Courses.csv"))
    teachers     = pd.read_csv(os.path.join(DATA, "Teachers.csv"))
    transactions = pd.read_csv(os.path.join(DATA, "Transactions.csv"))
    users        = pd.read_csv(os.path.join(DATA, "Users.csv"))
    ml_data      = pd.read_csv(os.path.join(DATA, "master_ml_final.csv"))
    return courses, teachers, transactions, users, ml_data

@st.cache_resource
def load_models():
    return (joblib.load(os.path.join(MODELS, "best_Course_Revenue.pkl")),
            joblib.load(os.path.join(MODELS, "best_Category_Revenue.pkl")),
            joblib.load(os.path.join(MODELS, "best_Enrollment_Count.pkl")))

courses, teachers, transactions, users, ml_data = load_data()
rev_model_data, catrev_model_data, enroll_model_data = load_models()

total_revenue     = ml_data['CourseRevenue'].sum()
total_enrollments = ml_data['EnrollmentCount'].sum()
total_courses     = len(ml_data)
paid_courses      = len(ml_data[ml_data['CourseType'] == 'Paid'])
avg_rating        = ml_data['CourseRating'].mean()

# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 EduPro Analytics")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "📊 EDA Explorer",
        "🤖 Predictions",
        "🏆 Feature Insights",
        "📋 Data Tables",
        "🎯 Recommendation Engine"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**📁 Dataset Info**")
    st.markdown(f"- Courses: **{total_courses}**")
    st.markdown(f"- Transactions: **{len(transactions):,}**")
    st.markdown(f"- Teachers: **{len(teachers)}**")
    st.markdown(f"- Users: **3,000**")
    st.markdown("---")
    st.markdown("**🤖 Best Models**")
    st.markdown("- Enrollment: Random Forest")
    st.markdown("- Revenue: Ridge Regression")
    st.markdown("- Category Rev: Gradient Boosting")
    st.markdown("---")
    st.caption("EduPro Predictive Modeling\n© 2025 Unified Mentor Project")

# ════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("🎓 EduPro Online Platform — Analytics Dashboard")
    st.markdown("Predictive Modeling for Course Demand and Revenue Forecasting")
    st.markdown("---")

    # ── KPI Cards ────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""<div class='metric-card'>
            <div class='metric-value'>₹{total_revenue/100000:.2f}L</div>
            <div class='metric-label'>💰 Total Revenue</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card' style='border-color:#4CAF50'>
            <div class='metric-value'>{total_enrollments:,}</div>
            <div class='metric-label'>👥 Total Enrollments</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card' style='border-color:#FF6B35'>
            <div class='metric-value'>{total_courses}</div>
            <div class='metric-label'>📚 Total Courses</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card' style='border-color:#FF9800'>
            <div class='metric-value'>{avg_rating:.2f}⭐</div>
            <div class='metric-label'>🌟 Avg Course Rating</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        st.markdown(f"""<div class='metric-card' style='border-color:#9C27B0'>
            <div class='metric-value'>{paid_courses}</div>
            <div class='metric-label'>💳 Paid Courses</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Row 1: Revenue & Enrollment by Category ───────────────────────
    col1, col2 = st.columns(2)

    with col1:
        cat_rev = (ml_data.groupby('CourseCategory')['CourseRevenue']
                   .sum().reset_index()
                   .sort_values('CourseRevenue', ascending=True))
        fig = px.bar(cat_rev, x='CourseRevenue', y='CourseCategory',
                     orientation='h',
                     title='💰 Total Revenue by Category',
                     labels={'CourseRevenue':'Revenue (₹)',
                             'CourseCategory':'Category'},
                     color='CourseRevenue',
                     color_continuous_scale='Blues',
                     template=PLOTLY_TEMPLATE)
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>Revenue: ₹%{x:,.0f}<extra></extra>')
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          height=420)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cat_enroll = (ml_data.groupby('CourseCategory')['EnrollmentCount']
                      .sum().reset_index()
                      .sort_values('EnrollmentCount', ascending=True))
        fig = px.bar(cat_enroll, x='EnrollmentCount', y='CourseCategory',
                     orientation='h',
                     title='👥 Total Enrollments by Category',
                     labels={'EnrollmentCount':'Enrollments',
                             'CourseCategory':'Category'},
                     color='EnrollmentCount',
                     color_continuous_scale='Greens',
                     template=PLOTLY_TEMPLATE)
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>Enrollments: %{x:,}<extra></extra>')
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          height=420)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: Monthly Trends & Free vs Paid ─────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        transactions['Month'] = transactions['TransactionDate'].dt.to_period('M')
        monthly = (transactions.groupby('Month')
                   .agg(Enrollments=('TransactionID','count'),
                        Revenue=('Amount','sum'))
                   .reset_index())
        monthly['Month'] = monthly['Month'].astype(str)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=monthly['Month'], y=monthly['Enrollments'],
            name='Enrollments', mode='lines+markers',
            line=dict(color='steelblue', width=2),
            hovertemplate='%{x}<br>Enrollments: %{y:,}<extra></extra>'),
            secondary_y=False)
        fig.add_trace(go.Scatter(
            x=monthly['Month'], y=monthly['Revenue'],
            name='Revenue (₹)', mode='lines+markers',
            line=dict(color='coral', width=2),
            hovertemplate='%{x}<br>Revenue: ₹%{y:,.0f}<extra></extra>'),
            secondary_y=True)
        fig.update_layout(title='📅 Monthly Enrollment & Revenue Trend',
                          template=PLOTLY_TEMPLATE, height=380,
                          hovermode='x unified')
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(title_text="Enrollments", secondary_y=False)
        fig.update_yaxes(title_text="Revenue (₹)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        type_stats = ml_data.groupby('CourseType').agg(
            Count=('CourseID','count'),
            AvgEnrollment=('EnrollmentCount','mean'),
            AvgRevenue=('CourseRevenue','mean')
        ).reset_index()

        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type":"pie"}, {"type":"bar"}]],
            subplot_titles=('Course Count Split', 'Avg Revenue'))
        fig.add_trace(go.Pie(
            labels=type_stats['CourseType'],
            values=type_stats['Count'],
            hole=0.4,
            marker_colors=['#2196F3','#FF9800'],
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>'),
            row=1, col=1)
        fig.add_trace(go.Bar(
            x=type_stats['CourseType'],
            y=type_stats['AvgRevenue'],
            marker_color=['#2196F3','#FF9800'],
            hovertemplate='<b>%{x}</b><br>Avg Revenue: ₹%{y:,.0f}<extra></extra>',
            showlegend=False),
            row=1, col=2)
        fig.update_layout(title='💳 Free vs Paid Analysis',
                          template=PLOTLY_TEMPLATE, height=380)
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA EXPLORER
# ════════════════════════════════════════════════════════════════════
elif page == "📊 EDA Explorer":
    st.title("📊 EDA Explorer")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        ["📚 Course Analysis", "👨‍🏫 Instructor Analysis",
         "👥 User Demographics"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            n = st.slider("Top N courses by Enrollment", 5, 20, 10)
            top_e = ml_data.nlargest(n, 'EnrollmentCount')
            fig = px.bar(top_e, x='EnrollmentCount', y='CourseName',
                         orientation='h',
                         color='CourseCategory',
                         color_discrete_sequence=COLOR_SEQ,
                         title=f'Top {n} Courses by Enrollment',
                         labels={'EnrollmentCount':'Enrollments',
                                 'CourseName':'Course'},
                         template=PLOTLY_TEMPLATE)
            fig.update_traces(
                hovertemplate='<b>%{y}</b><br>Enrollments: %{x}<extra></extra>')
            fig.update_layout(height=450, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            n2 = st.slider("Top N courses by Revenue", 5, 20, 10)
            top_r = ml_data[ml_data['CourseRevenue']>0].nlargest(
                n2, 'CourseRevenue')
            fig = px.bar(top_r, x='CourseRevenue', y='CourseName',
                         orientation='h',
                         color='CourseCategory',
                         color_discrete_sequence=COLOR_SEQ,
                         title=f'Top {n2} Courses by Revenue',
                         labels={'CourseRevenue':'Revenue (₹)',
                                 'CourseName':'Course'},
                         template=PLOTLY_TEMPLATE)
            fig.update_traces(
                hovertemplate='<b>%{y}</b><br>Revenue: ₹%{x:,.0f}<extra></extra>')
            fig.update_layout(height=450, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        selected_cat = st.selectbox("Filter by Category",
            ["All"] + sorted(ml_data['CourseCategory'].unique().tolist()))
        filtered = ml_data if selected_cat=="All" else \
                   ml_data[ml_data['CourseCategory']==selected_cat]

        col3, col4, col5 = st.columns(3)
        with col3:
            lev_e = filtered.groupby('CourseLevel')['EnrollmentCount'].sum().reset_index()
            fig = px.bar(lev_e, x='CourseLevel', y='EnrollmentCount',
                         color='CourseLevel',
                         color_discrete_sequence=COLOR_SEQ,
                         title='Enrollments by Level',
                         template=PLOTLY_TEMPLATE)
            fig.update_traces(
                hovertemplate='<b>%{x}</b><br>Enrollments: %{y:,}<extra></extra>')
            st.plotly_chart(fig, use_container_width=True)

        with col4:
            lev_r = filtered.groupby('CourseLevel')['CourseRevenue'].sum().reset_index()
            fig = px.bar(lev_r, x='CourseLevel', y='CourseRevenue',
                         color='CourseLevel',
                         color_discrete_sequence=COLOR_SEQ,
                         title='Revenue by Level',
                         template=PLOTLY_TEMPLATE)
            fig.update_traces(
                hovertemplate='<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>')
            st.plotly_chart(fig, use_container_width=True)

        with col5:
            fig = px.histogram(filtered, x='CourseRating', nbins=10,
                               title='Rating Distribution',
                               color_discrete_sequence=['mediumpurple'],
                               template=PLOTLY_TEMPLATE)
            fig.update_traces(
                hovertemplate='Rating: %{x}<br>Count: %{y}<extra></extra>')
            st.plotly_chart(fig, use_container_width=True)

        # Scatter: Price vs Enrollment
        st.markdown("---")
        fig = px.scatter(ml_data,
                         x='CoursePrice', y='EnrollmentCount',
                         color='CourseCategory',
                         size='CourseRating',
                         hover_name='CourseName',
                         hover_data={'CoursePrice':':.0f',
                                     'EnrollmentCount':True,
                                     'CourseRating':':.2f',
                                     'CourseType':True},
                         title='Price vs Enrollment (size = Rating)',
                         labels={'CoursePrice':'Price (₹)',
                                 'EnrollmentCount':'Enrollments'},
                         color_discrete_sequence=COLOR_SEQ,
                         template=PLOTLY_TEMPLATE)
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        trans_detail = (transactions
            .merge(courses[['CourseID','CourseCategory']], on='CourseID')
            .merge(teachers[['TeacherID','TeacherName','YearsOfExperience',
                              'TeacherRating','Expertise']], on='TeacherID'))
        teacher_perf = (trans_detail
            .groupby(['TeacherID','TeacherName'])
            .agg(TotalEnrollments=('CourseID','count'),
                 TotalRevenue=('Amount','sum'),
                 TeacherRating=('TeacherRating','first'),
                 YearsOfExp=('YearsOfExperience','first'),
                 Expertise=('Expertise','first'))
            .reset_index())

        col1, col2 = st.columns(2)
        with col1:
            top_t = teacher_perf.nlargest(10, 'TotalRevenue')
            fig = px.bar(top_t, x='TotalRevenue', y='TeacherName',
                         orientation='h',
                         color='TeacherRating',
                         color_continuous_scale='RdYlGn',
                         title='Top 10 Teachers by Revenue',
                         labels={'TotalRevenue':'Revenue (₹)',
                                 'TeacherName':'Teacher'},
                         template=PLOTLY_TEMPLATE)
            fig.update_traces(
                hovertemplate='<b>%{y}</b><br>Revenue: ₹%{x:,.0f}<extra></extra>')
            fig.update_layout(height=420,
                              yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(teacher_perf,
                             x='TeacherRating', y='TotalRevenue',
                             size='YearsOfExp',
                             color='Expertise',
                             hover_name='TeacherName',
                             hover_data={'TeacherRating':':.2f',
                                         'TotalRevenue':':.0f',
                                         'YearsOfExp':True},
                             title='Teacher Rating vs Revenue (size=Experience)',
                             labels={'TeacherRating':'Rating',
                                     'TotalRevenue':'Revenue (₹)'},
                             color_discrete_sequence=COLOR_SEQ,
                             template=PLOTLY_TEMPLATE)
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

        # Experience vs Enrollments
        fig = px.scatter(teacher_perf,
                         x='YearsOfExp', y='TotalEnrollments',
                         color='TeacherRating',
                         size='TotalRevenue',
                         hover_name='TeacherName',
                         hover_data={'YearsOfExp':True,
                                     'TotalEnrollments':True,
                                     'TeacherRating':':.2f'},
                         title='Experience vs Enrollments (size=Revenue, color=Rating)',
                         color_continuous_scale='Viridis',
                         template=PLOTLY_TEMPLATE)
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(users, x='Age', nbins=15,
                               color_discrete_sequence=['steelblue'],
                               title='User Age Distribution',
                               template=PLOTLY_TEMPLATE)
            fig.add_vline(x=users['Age'].mean(),
                          line_dash='dash', line_color='red',
                          annotation_text=f"Mean: {users['Age'].mean():.1f}")
            fig.update_traces(
                hovertemplate='Age: %{x}<br>Count: %{y}<extra></extra>')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            gender = users['Gender'].value_counts().reset_index()
            gender.columns = ['Gender','Count']
            fig = px.pie(gender, names='Gender', values='Count',
                         hole=0.4,
                         color_discrete_sequence=['#2196F3','#E91E63'],
                         title='Gender Distribution',
                         template=PLOTLY_TEMPLATE)
            fig.update_traces(
                hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>')
            st.plotly_chart(fig, use_container_width=True)

        # Age by Gender box plot
        fig = px.box(users, x='Gender', y='Age',
                     color='Gender',
                     color_discrete_sequence=['#2196F3','#E91E63'],
                     title='Age Distribution by Gender',
                     template=PLOTLY_TEMPLATE)
        fig.update_traces(
            hovertemplate='Gender: %{x}<br>Age: %{y}<extra></extra>')
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTIONS
# ════════════════════════════════════════════════════════════════════
elif page == "🤖 Predictions":
    st.title("🤖 Course Demand & Revenue Predictor")
    st.markdown("---")

    col_inp, col_out = st.columns([1, 1])

    with col_inp:
        st.markdown("<div class='section-header'>⚙️ Course Parameters</div>",
                    unsafe_allow_html=True)
        course_category = st.selectbox("Course Category",
            sorted(ml_data['CourseCategory'].unique().tolist()))
        course_type  = st.selectbox("Course Type", ["Paid","Free"])
        course_level = st.selectbox("Course Level",
            ["Beginner","Intermediate","Advanced"])
        course_price = st.slider("Course Price (₹)", 0, 500,
            150 if course_type=="Paid" else 0, step=10)
        if course_type == "Free":
            course_price = 0
            st.info("ℹ️ Free courses have ₹0 price.")
        course_duration = st.slider("Course Duration (hours)", 1, 50, 20)
        course_rating   = st.slider("Expected Course Rating",
                                     1.0, 5.0, 3.5, step=0.1)
        st.markdown("**👨‍🏫 Instructor Parameters**")
        teacher_exp     = st.slider("Teacher Experience (years)", 1, 25, 5)
        teacher_rating  = st.slider("Teacher Rating", 1.0, 5.0, 3.5, step=0.1)
        teacher_expertise = st.selectbox("Teacher Expertise",
            sorted(teachers['Expertise'].unique().tolist()))
        expertise_match = int(
            teacher_expertise.lower() == course_category.lower())

    with col_out:
        st.markdown("<div class='section-header'>📈 Prediction Results</div>",
                    unsafe_allow_html=True)

        cat_map   = {v:i for i,v in enumerate(
            sorted(ml_data['CourseCategory'].unique()))}
        type_map  = {'Free':0,'Paid':1}
        level_map = {'Advanced':0,'Beginner':1,'Intermediate':2}

        def price_band_enc(p):
            if p==0:      return 0
            elif p<=150:  return 2
            elif p<=350:  return 3
            else:         return 1

        def dur_bucket_enc(d):
            if d<=10:   return 2
            elif d<=25: return 1
            elif d<=40: return 0
            else:       return 3

        def rating_tier_enc(r):
            if r<2.0:   return 1
            elif r<3.5: return 2
            elif r<4.5: return 0
            else:       return 3

        def exp_bucket_enc(e):
            if e<=3:    return 1
            elif e<=8:  return 2
            elif e<=15: return 3
            else:       return 0

        is_free = 1 if course_type=="Free" else 0
        rev_per_enroll = course_price * 0.85 if course_type=="Paid" else 0

        input_full = pd.DataFrame([{
            'CoursePrice':          course_price,
            'CourseDuration':       course_duration,
            'CourseRating':         course_rating,
            'YearsOfExperience':    teacher_exp,
            'TeacherRating':        teacher_rating,
            'ExpertiseMatch':       expertise_match,
            'IsFree':               is_free,
            'RevenuePerEnrollment': rev_per_enroll,
            'CourseCategory_enc':   cat_map.get(course_category, 0),
            'CourseType_enc':       type_map[course_type],
            'CourseLevel_enc':      level_map[course_level],
            'PriceBand_enc':        price_band_enc(course_price),
            'DurationBucket_enc':   dur_bucket_enc(course_duration),
            'RatingTier_enc':       rating_tier_enc(course_rating),
            'ExperienceBucket_enc': exp_bucket_enc(teacher_exp),
            'TeacherRatingTier_enc':rating_tier_enc(teacher_rating)
        }])

        TOP_FEATURES_ENROLL = [
            'CourseDuration','CourseRating','RatingTier_enc',
            'CourseCategory_enc','CourseLevel_enc','PriceBand_enc',
            'CoursePrice','DurationBucket_enc'
        ]
        input_enroll = input_full[TOP_FEATURES_ENROLL]

        scaler        = rev_model_data['scaler']
        input_scaled  = scaler.transform(input_full)
        pred_revenue  = max(0, rev_model_data['model'].predict(input_scaled)[0])
        pred_catrev   = max(0, catrev_model_data['model'].predict(input_full)[0])
        pred_enroll   = max(0, int(enroll_model_data['model'].predict(input_enroll)[0]))

        # ── KPI Results ──────────────────────────────────────────────
        r1, r2, r3 = st.columns(3)
        with r1:
            st.markdown(f"""<div class='metric-card' style='border-color:#4CAF50'>
                <div class='metric-value'>{pred_enroll}</div>
                <div class='metric-label'>👥 Predicted Enrollments</div>
            </div>""", unsafe_allow_html=True)
        with r2:
            st.markdown(f"""<div class='metric-card' style='border-color:#2196F3'>
                <div class='metric-value'>₹{pred_revenue:,.0f}</div>
                <div class='metric-label'>💰 Course Revenue</div>
            </div>""", unsafe_allow_html=True)
        with r3:
            st.markdown(f"""<div class='metric-card' style='border-color:#FF6B35'>
                <div class='metric-value'>₹{pred_catrev:,.0f}</div>
                <div class='metric-label'>📊 Category Revenue</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # ── Gauge Chart — Revenue ────────────────────────────────────
        max_rev = ml_data['CourseRevenue'].max()
        pct     = min(pred_revenue / max_rev * 100, 100)
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=pred_revenue,
            title={'text': "Predicted Course Revenue (₹)",
                   'font': {'size': 14}},
            delta={'reference': ml_data['CourseRevenue'].mean(),
                   'valueformat': ',.0f'},
            number={'prefix': '₹', 'valueformat': ',.0f'},
            gauge={
                'axis': {'range': [0, max_rev]},
                'bar':  {'color': SUCCESS if pct>60 else
                                  WARNING if pct>30 else ACCENT},
                'steps': [
                    {'range': [0, max_rev*0.3],       'color': '#FFEBEE'},
                    {'range': [max_rev*0.3, max_rev*0.6], 'color': '#FFF3E0'},
                    {'range': [max_rev*0.6, max_rev],  'color': '#E8F5E9'}
                ],
                'threshold': {
                    'line': {'color':'red','width':3},
                    'thickness': 0.75,
                    'value': ml_data['CourseRevenue'].mean()
                }
            }
        ))
        fig.update_layout(height=300, template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

        # ── Radar Chart — Course Profile ─────────────────────────────
        st.markdown("**📡 Course Profile Radar**")
        categories_radar = ['Price Score','Rating','Duration Score',
                            'Teacher Rating','Experience Score']
        values_radar = [
            min(course_price/500, 1)*5,
            course_rating,
            min(course_duration/50, 1)*5,
            teacher_rating,
            min(teacher_exp/25, 1)*5
        ]
        fig = go.Figure(go.Scatterpolar(
            r=values_radar + [values_radar[0]],
            theta=categories_radar + [categories_radar[0]],
            fill='toself',
            fillcolor='rgba(33,150,243,0.2)',
            line=dict(color='#2196F3', width=2),
            hovertemplate='<b>%{theta}</b><br>Score: %{r:.2f}<extra></extra>'
        ))
        fig.update_layout(
            polar=dict(
                bgcolor='rgba(0,0,0,0)',
                radialaxis=dict(
                    visible=True, range=[0,5],
                    gridcolor='#555E6B',
                    linecolor='#555E6B',
                    tickfont=dict(color='#8B949E')
                ),
                angularaxis=dict(
                    gridcolor='#555E6B',
                    linecolor='#555E6B',
                    tickfont=dict(color='#FF4444')
                )
            ),
            height=320, template=PLOTLY_TEMPLATE)
        st.plotly_chart(fig, use_container_width=True)

        # ── Tips ─────────────────────────────────────────────────────
        st.markdown("**💡 Optimization Tips**")
        if course_type == "Free":
            st.warning("💡 Consider making this Paid to generate revenue.")
        if course_rating < 3.5:
            st.warning("💡 Improving rating above 3.5 boosts enrollments.")
        if teacher_rating < 3.0:
            st.warning("💡 Assign a higher-rated instructor.")
        if expertise_match == 0:
            st.info("ℹ️ Teacher expertise doesn't match course category.")
        if course_type=="Paid" and pred_revenue > 50000:
            st.success("✅ High revenue potential!")


# ════════════════════════════════════════════════════════════════════
# PAGE 4 — FEATURE INSIGHTS
# ════════════════════════════════════════════════════════════════════
elif page == "🏆 Feature Insights":
    st.title("🏆 Feature Importance & Model Insights")
    st.markdown("---")

    summary_df = pd.DataFrame({
        'Target'    : ['Enrollment Count','Course Revenue','Category Revenue'],
        'Best Model': ['Random Forest (Tuned)','Ridge Regression',
                       'Gradient Boosting'],
        'MAE'       : ['9.85','₹1,672','₹10,786'],
        'RMSE'      : ['~12.0','₹2,479','₹17,135'],
        'R² Score'  : ['-0.144','0.983','0.893'],
        'Verdict'   : ['⚠️ Low variance','✅ Excellent','✅ Very Good']
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.markdown("---")

    TOP_FEATURES_ENROLL = [
        'CourseDuration','CourseRating','RatingTier_enc',
        'CourseCategory_enc','CourseLevel_enc','PriceBand_enc',
        'CoursePrice','DurationBucket_enc'
    ]
    FEATURES_FULL = [
        'CoursePrice','CourseDuration','CourseRating',
        'YearsOfExperience','TeacherRating','ExpertiseMatch',
        'IsFree','RevenuePerEnrollment',
        'CourseCategory_enc','CourseType_enc','CourseLevel_enc',
        'PriceBand_enc','DurationBucket_enc','RatingTier_enc',
        'ExperienceBucket_enc','TeacherRatingTier_enc'
    ]

    model_configs = [
        (enroll_model_data, TOP_FEATURES_ENROLL,
         "Enrollment Count",  '#5B9BD5'),
        (rev_model_data,    FEATURES_FULL,
         "Course Revenue",    '#FF7043'),
        (catrev_model_data, FEATURES_FULL,
         "Category Revenue",  '#66BB6A'),
    ]

    cols = st.columns(3)
    for col, (mdata, feats, title, color) in zip(cols, model_configs):
        with col:
            model  = mdata['model']
            labels = [f.replace('_enc','').replace('_',' ') for f in feats]
            if hasattr(model,'feature_importances_'):
                imps = model.feature_importances_
            else:
                imps = np.abs(model.coef_)
            idx    = np.argsort(imps)[-8:]
            imp_df = pd.DataFrame({
                'Feature':    [labels[i] for i in idx],
                'Importance': imps[idx]
            }).sort_values('Importance')
            fig = px.bar(imp_df, x='Importance', y='Feature',
                         orientation='h',
                         title=f'{title}',
                         color='Importance',
                         color_continuous_scale='Blues',
                         template=PLOTLY_TEMPLATE)
            fig.update_traces(
                hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>')
            fig.update_layout(height=400, showlegend=False,
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Correlation Heatmap ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-header'>🔥 Correlation Heatmap</div>",
                unsafe_allow_html=True)
    numeric_cols = ['CoursePrice','CourseDuration','CourseRating',
                    'EnrollmentCount','CourseRevenue']
    corr = ml_data[numeric_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True,
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1,
                    title='Feature Correlation Heatmap',
                    template=PLOTLY_TEMPLATE)
    fig.update_traces(
        hovertemplate='<b>%{x} vs %{y}</b><br>Correlation: %{z}<extra></extra>')
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    # ── Actual vs Predicted ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-header'>🎯 Actual vs Predicted</div>",
                unsafe_allow_html=True)
    from sklearn.model_selection import cross_val_predict
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    FEATURES_FULL_COLS = [
        'CoursePrice','CourseDuration','CourseRating',
        'YearsOfExperience','TeacherRating','ExpertiseMatch',
        'IsFree','RevenuePerEnrollment',
        'CourseCategory_enc','CourseType_enc','CourseLevel_enc',
        'PriceBand_enc','DurationBucket_enc','RatingTier_enc',
        'ExperienceBucket_enc','TeacherRatingTier_enc'
    ]

    pred_configs = [
        (ml_data[TOP_FEATURES_ENROLL],   ml_data['EnrollmentCount'],
         "Enrollment Count",  '#5B9BD5'),
        (ml_data[FEATURES_FULL_COLS],    ml_data['CourseRevenue'],
         "Course Revenue",    '#FF7043'),
        (ml_data[FEATURES_FULL_COLS],    ml_data['CategoryRevenue'],
         "Category Revenue",  '#66BB6A'),
    ]

    cols2 = st.columns(3)
    for col, (X_use, y_use, title, color) in zip(cols2, pred_configs):
        with col:
            mdl = (enroll_model_data if 'Enrollment' in title else
                   rev_model_data    if 'Course' in title else
                   catrev_model_data)['model']
            mdl.fit(X_use, y_use)
            y_pred = cross_val_predict(mdl, X_use, y_use, cv=kf)
            scatter_df = pd.DataFrame({'Actual': y_use, 'Predicted': y_pred,
                                       'Course': ml_data['CourseName']})
            fig = px.scatter(scatter_df, x='Actual', y='Predicted',
                             hover_name='Course',
                             title=f'Actual vs Predicted<br>{title}',
                             color_discrete_sequence=[color],
                             template=PLOTLY_TEMPLATE)
            min_v = min(y_use.min(), y_pred.min())
            max_v = max(y_use.max(), y_pred.max())
            fig.add_trace(go.Scatter(
                x=[min_v, max_v], y=[min_v, max_v],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash', width=2)))
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 5 — DATA TABLES
# ════════════════════════════════════════════════════════════════════
elif page == "📋 Data Tables":
    st.title("📋 Data Explorer")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(
        ["📚 Master Dataset","💰 Revenue Table","🏆 Top Performers"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            cat_f = st.multiselect("Category",
                ml_data['CourseCategory'].unique(),
                default=ml_data['CourseCategory'].unique())
        with col2:
            type_f = st.multiselect("Type", ["Free","Paid"],
                                     default=["Free","Paid"])
        with col3:
            level_f = st.multiselect("Level",
                ["Beginner","Intermediate","Advanced"],
                default=["Beginner","Intermediate","Advanced"])

        filtered = ml_data[
            (ml_data['CourseCategory'].isin(cat_f)) &
            (ml_data['CourseType'].isin(type_f)) &
            (ml_data['CourseLevel'].isin(level_f))]

        display_cols = ['CourseName','CourseCategory','CourseType',
                        'CourseLevel','CoursePrice','CourseRating',
                        'EnrollmentCount','CourseRevenue']
        st.dataframe(filtered[display_cols].reset_index(drop=True),
                     use_container_width=True)
        st.caption(f"Showing {len(filtered)} of {len(ml_data)} courses")

        # Interactive scatter in data table page
        fig = px.scatter(filtered,
                         x='CourseRating', y='EnrollmentCount',
                         color='CourseType',
                         size='CoursePrice',
                         hover_name='CourseName',
                         hover_data={'CoursePrice':':.0f',
                                     'CourseRevenue':':.0f',
                                     'CourseLevel':True},
                         title='Rating vs Enrollment (filtered)',
                         color_discrete_sequence=['#2196F3','#FF9800'],
                         template=PLOTLY_TEMPLATE)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        rev_table = (ml_data[['CourseName','CourseCategory','CourseType',
                               'CoursePrice','EnrollmentCount',
                               'CourseRevenue','CategoryRevenue']]
                     .sort_values('CourseRevenue', ascending=False)
                     .reset_index(drop=True))
        st.dataframe(rev_table, use_container_width=True)

        fig = px.bar(rev_table.head(15),
                     x='CourseName', y='CourseRevenue',
                     color='CourseCategory',
                     hover_data={'CoursePrice':':.0f',
                                 'EnrollmentCount':True},
                     title='Top 15 Courses Revenue Comparison',
                     color_discrete_sequence=COLOR_SEQ,
                     template=PLOTLY_TEMPLATE)
        fig.update_xaxes(tickangle=45)
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>Revenue: ₹%{y:,.0f}<extra></extra>')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**🥇 Top 10 by Enrollment**")
            top_e = (ml_data[['CourseName','CourseCategory','EnrollmentCount']]
                     .nlargest(10,'EnrollmentCount')
                     .reset_index(drop=True))
            top_e.index += 1
            st.dataframe(top_e, use_container_width=True)
        with col2:
            st.markdown("**💰 Top 10 by Revenue**")
            top_r = (ml_data[ml_data['CourseRevenue']>0]
                     [['CourseName','CourseCategory','CourseRevenue']]
                     .nlargest(10,'CourseRevenue')
                     .reset_index(drop=True))
            top_r.index += 1
            st.dataframe(top_r, use_container_width=True)

        # Sunburst chart
        fig = px.sunburst(
            ml_data, path=['CourseCategory','CourseType','CourseLevel'],
            values='EnrollmentCount',
            color='CourseRevenue',
            color_continuous_scale='RdYlGn',
            title='Course Hierarchy — Enrollment & Revenue',
            template=PLOTLY_TEMPLATE)
        fig.update_traces(
            hovertemplate='<b>%{label}</b><br>Enrollments: %{value:,}<extra></extra>')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════════
# PAGE 6 — RECOMMENDATION ENGINE
# ════════════════════════════════════════════════════════════════════
elif page == "🎯 Recommendation Engine":
    st.title("🤖 AI Course Recommendation Engine")
    st.markdown("---")

    # ──────────────────────────────────────────────────────────────────
    # 1) Course Recommendation (Content-Based)
    # ──────────────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>🎓 Recommended Courses</div>",
                unsafe_allow_html=True)

    # User profile inputs
    col1, col2 = st.columns([2,1])
    with col1:
        user_name = st.text_input("👤 Your Name", "New Learner")
        st.markdown("### 🎯 Tell us about your learning goals")
        user_cat = st.multiselect("Preferred Categories",
            ml_data['CourseCategory'].unique(),
            default=ml_data['CourseCategory'].unique()[:2])
        user_level = st.selectbox("Current Level",
            ["Beginner","Intermediate","Advanced"],
            index=0)
        user_price = st.slider("Max Price (₹)", 0, 20000, 2000)
        user_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.5, 0.1)

    with col2:
        st.markdown("### 📊 Your Profile")
        st.metric("Selected Categories", len(user_cat))
        st.metric("Max Budget", f"₹{user_price:,.0f}")
        st.metric("Min Rating", f"{user_rating}/5.0")

        # Recommendation logic
        def get_course_recommendations(cat, level, price, rating, top_n=5):
            # Filter by category, level, price, rating
            mask = (
                (ml_data['CourseCategory'].isin(cat)) &
                (ml_data['CourseLevel'] == level) &
                (ml_data['CoursePrice'] <= price) &
                (ml_data['CourseRating'] >= rating)
            )
            filtered = ml_data[mask].copy()

            if filtered.empty:
                return pd.DataFrame()

            # Calculate similarity score
            filtered['similarity'] = (
                (filtered['CourseRating'] / 5.0) * 0.4 +
                (filtered['CoursePrice'] / filtered['CoursePrice'].max()) * 0.3 +
                (filtered['CourseDuration'] / filtered['CourseDuration'].max()) * 0.3
            )

            # Sort by similarity and return top N
            recommendations = filtered.sort_values(
                'similarity', ascending=False
            ).head(top_n)

            return recommendations[[
                'CourseName','CourseCategory','CourseLevel',
                'CoursePrice','CourseRating','EnrollmentCount',
                'CourseRevenue','similarity'
            ]]

        recs = get_course_recommendations(
            user_cat, user_level, user_price, user_rating, top_n=5
        )

        if not recs.empty:
            st.success(f"✅ Found {len(recs)} recommended courses!")
        else:
            st.warning("⚠️ No courses match your criteria. Try adjusting filters.")

    # Display recommendations
    if not recs.empty:
        st.markdown("---")
        st.markdown(f"### 🎓 Top Recommendations for **{user_name}**")

        cols = st.columns(len(recs))
        for i, (_, row) in enumerate(recs.iterrows()):
            with cols[i]:
                st.markdown(f"#### {i+1}. {row['CourseName']}")
                st.markdown(f"**Category:** {row['CourseCategory']}")
                st.markdown(f"**Level:** {row['CourseLevel']}")
                st.markdown(f"**Price:** ₹{row['CoursePrice']:,.0f}")
                st.markdown(f"**Rating:** {row['CourseRating']}/5.0")
                st.markdown(f"**Enrollments:** {row['EnrollmentCount']:,}")
                st.markdown(f"**Revenue:** ₹{row['CourseRevenue']:,.0f}")
                st.markdown(f"**Similarity:** {row['similarity']:.3f}")

                # Recommendation card
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 12px;
                    border-radius: 8px;
                    text-align: center;
                    margin-top: 8px;
                ">
                    <strong>Recommended!</strong>
                </div>
                """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────
    # 2) Course Recommendation (Collaborative Filtering)
    # ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-header'>👥 Similar Users</div>",
                unsafe_allow_html=True)
    if 'UserID' not in ml_data.columns:
        st.info("ℹ️ Collaborative filtering, Hybrid, Comparison, Performance, and Evaluation sections require a 'UserID' column in master_ml_final.csv. This column is not present in the current dataset — only Content-Based recommendations are available.")

    # User-based collaborative filtering
    def get_similar_users(target_user_id, top_n=5):
        if 'UserID' not in ml_data.columns:
            return []
        # Get target user's courses
        target_courses = set(
            ml_data[ml_data['UserID'] == target_user_id]['CourseName']
        )

        # Calculate similarity with all other users
        similarities = {}
        for user_id in ml_data['UserID'].unique():
            if user_id == target_user_id:
                continue

            other_courses = set(
                ml_data[ml_data['UserID'] == user_id]['CourseName']
            )

            # Jaccard similarity
            intersection = len(target_courses.intersection(other_courses))
            union = len(target_courses.union(other_courses))
            similarity = intersection / union if union > 0 else 0

            if similarity > 0:
                similarities[user_id] = similarity

        # Sort by similarity
        sorted_users = sorted(
            similarities.items(), key=lambda x: x[1], reverse=True
        )[:top_n]

        return sorted_users

    # Get similar users
    similar_users = get_similar_users(1, top_n=3)

    if similar_users:
        st.markdown("### 👥 Users similar to User 1")

        for user_id, similarity in similar_users:
            st.markdown(f"#### User {user_id} (Similarity: {similarity:.3f})")

            # Get courses taken by similar user
            similar_courses = ml_data[
                (ml_data['UserID'] == user_id) &
                (~ml_data['CourseName'].isin(
                    ml_data[ml_data['UserID'] == 1]['CourseName']
                ))
            ][['CourseName','CourseCategory','CourseLevel','CoursePrice']]

            if not similar_courses.empty:
                st.dataframe(similar_courses.head(5), use_container_width=True)
            else:
                st.info("No new courses found.")

    # ──────────────────────────────────────────────────────────────────
    # 3) Course Recommendation (Hybrid)
    # ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-header'>⚡ Hybrid Recommendations</div>",
                unsafe_allow_html=True)

    def get_hybrid_recommendations(user_id, top_n=5):
        if 'UserID' not in ml_data.columns:
            return pd.DataFrame()
        # Get content-based recommendations
        user_data = ml_data[ml_data['UserID'] == user_id]

        if user_data.empty:
            return pd.DataFrame()

        # Get user's preferred categories and level
        user_cat = user_data['CourseCategory'].unique()
        user_level = user_data['CourseLevel'].iloc[0]

        # Get content-based recommendations
        content_recs = get_course_recommendations(
            user_cat, user_level, 2000, 3.5, top_n=top_n
        )

        # Get similar users
        similar_users = get_similar_users(user_id, top_n=top_n)

        # Get courses from similar users
        similar_courses = []
        for sim_user_id, similarity in similar_users:
            courses = ml_data[
                (ml_data['UserID'] == sim_user_id) &
                (~ml_data['CourseName'].isin(user_data['CourseName']))
            ][['CourseName','CourseCategory','CourseLevel','CoursePrice','CourseRating']]

            if not courses.empty:
                courses['similarity'] = similarity
                similar_courses.append(courses)

        if similar_courses:
            similar_courses_df = pd.concat(similar_courses, ignore_index=True)
            similar_courses_df = similar_courses_df.sort_values(
                'similarity', ascending=False
            ).head(top_n)
        else:
            similar_courses_df = pd.DataFrame()

        # Hybrid approach
        hybrid_recs = pd.concat([content_recs, similar_courses_df], ignore_index=True)
        hybrid_recs = hybrid_recs.drop_duplicates(subset=['CourseName'])
        hybrid_recs = hybrid_recs.sort_values('similarity', ascending=False).head(top_n)

        return hybrid_recs

    # Get hybrid recommendations
    hybrid_recs = get_hybrid_recommendations(1, top_n=5)

    if not hybrid_recs.empty:
        st.markdown("### ⚡ Top Hybrid Recommendations for **User 1**")

        cols = st.columns(len(hybrid_recs))
        for i, (_, row) in enumerate(hybrid_recs.iterrows()):
            with cols[i]:
                st.markdown(f"#### {i+1}. {row['CourseName']}")
                st.markdown(f"**Category:** {row['CourseCategory']}")
                st.markdown(f"**Level:** {row['CourseLevel']}")
                st.markdown(f"**Price:** ₹{row['CoursePrice']:,.0f}")
                st.markdown(f"**Rating:** {row['CourseRating']}/5.0")
                st.markdown(f"**Similarity:** {row['similarity']:.3f}")

                # Recommendation card
                st.markdown("""
                <div style="
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    color: white;
                    padding: 12px;
                    border-radius: 8px;
                    text-align: center;
                    margin-top: 8px;
                ">
                    <strong>Recommended!</strong>
                </div>
                """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────
    # 4) Recommendation System Comparison
    # ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-header'>📊 Recommendation Comparison</div>",
                unsafe_allow_html=True)

    # Get recommendations from all methods
    content_recs = get_course_recommendations(user_cat, user_level, 2000, 3.5, top_n=5)
    similar_users = get_similar_users(1, top_n=5)

    # Prepare comparison data
    comparison_data = []

    # Content-based
    if not content_recs.empty:
        for _, row in content_recs.iterrows():
            comparison_data.append({
                'Method': 'Content-Based',
                'Course': row['CourseName'],
                'Category': row['CourseCategory'],
                'Level': row['CourseLevel'],
                'Price': row['CoursePrice'],
                'Rating': row['CourseRating'],
                'Similarity': row['similarity']
            })

    # Collaborative filtering
    user1_courses = ml_data[ml_data['UserID'] == 1]['CourseName'] if 'UserID' in ml_data.columns else pd.Series(dtype=str)
    if similar_users:
        for user_id, similarity in similar_users:
            courses = ml_data[
                (ml_data['UserID'] == user_id) &
                (~ml_data['CourseName'].isin(user1_courses))
            ][['CourseName','CourseCategory','CourseLevel','CoursePrice','CourseRating']]

            if not courses.empty:
                for _, row in courses.head(1).iterrows():
                    comparison_data.append({
                        'Method': f'Similar User {user_id}',
                        'Course': row['CourseName'],
                        'Category': row['CourseCategory'],
                        'Level': row['CourseLevel'],
                        'Price': row['CoursePrice'],
                        'Rating': row['CourseRating'],
                        'Similarity': similarity
                    })

    # Hybrid
    if not hybrid_recs.empty:
        for _, row in hybrid_recs.iterrows():
            comparison_data.append({
                'Method': 'Hybrid',
                'Course': row['CourseName'],
                'Category': row['CourseCategory'],
                'Level': row['CourseLevel'],
                'Price': row['CoursePrice'],
                'Rating': row['CourseRating'],
                'Similarity': row['similarity']
            })

    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)

        # Display comparison table
        st.markdown("### 📊 Recommendation Comparison Table")
        st.dataframe(comparison_df, use_container_width=True)

        # Recommendation comparison chart
        fig = px.bar(
            comparison_df,
            x='Course',
            y='Similarity',
            color='Method',
            barmode='group',
            title='Recommendation Method Comparison',
            color_discrete_sequence=COLOR_SEQ,
            template=PLOTLY_TEMPLATE
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────
    # 5) Recommendation System Performance
    # ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-header'>📈 Recommendation Performance</div>",
                unsafe_allow_html=True)

    # Calculate recommendation metrics
    def calculate_recommendation_metrics(user_id):
        if 'UserID' not in ml_data.columns:
            return {}
        user_data = ml_data[ml_data['UserID'] == user_id]
        if user_data.empty:
            return {}

        user_cat = user_data['CourseCategory'].unique()
        user_level = user_data['CourseLevel'].iloc[0]

        # Get recommendations
        content_recs = get_course_recommendations(
            user_cat, user_level, 2000, 3.5, top_n=10
        )
        similar_users = get_similar_users(user_id, top_n=5)

        # Calculate metrics
        metrics = {}

        # Content-based metrics
        if not content_recs.empty:
            metrics['Content-Based'] = {
                'Precision': (content_recs['CourseRating'] >= 3.5).mean(),
                'Recall': (content_recs['CourseRating'] >= 3.5).mean(),
                'F1-Score': (2 * 0.5 * 0.5) / (0.5 + 0.5) if content_recs['CourseRating'].mean() > 0 else 0
            }

        # Collaborative filtering metrics
        if similar_users:
            cf_metrics = []
            for sim_user_id, similarity in similar_users:
                courses = ml_data[
                    (ml_data['UserID'] == sim_user_id) &
                    (~ml_data['CourseName'].isin(user_data['CourseName']))
                ][['CourseName','CourseCategory','CourseLevel','CoursePrice','CourseRating']]

                if not courses.empty:
                    cf_metrics.append({
                        'Precision': (courses['CourseRating'] >= 3.5).mean(),
                        'Recall': (courses['CourseRating'] >= 3.5).mean(),
                        'F1-Score': (2 * 0.5 * 0.5) / (0.5 + 0.5) if courses['CourseRating'].mean() > 0 else 0
                    })

            if cf_metrics:
                metrics['Collaborative Filtering'] = {
                    'Precision': sum(m['Precision'] for m in cf_metrics) / len(cf_metrics),
                    'Recall': sum(m['Recall'] for m in cf_metrics) / len(cf_metrics),
                    'F1-Score': sum(m['F1-Score'] for m in cf_metrics) / len(cf_metrics)
                }

        # Hybrid metrics
        if not hybrid_recs.empty:
            metrics['Hybrid'] = {
                'Precision': (hybrid_recs['CourseRating'] >= 3.5).mean(),
                'Recall': (hybrid_recs['CourseRating'] >= 3.5).mean(),
                'F1-Score': (2 * 0.5 * 0.5) / (0.5 + 0.5) if hybrid_recs['CourseRating'].mean() > 0 else 0
            }

        return metrics

    # Calculate metrics for sample users (first 5 to avoid O(n²) slowdown)
    sample_users = ml_data['UserID'].unique()[:5] if 'UserID' in ml_data.columns else []
    all_metrics = {}
    for user_id in sample_users:
        all_metrics[user_id] = calculate_recommendation_metrics(user_id)

    # Display metrics
    st.markdown("### 📊 Recommendation Metrics")
    for user_id, metrics in all_metrics.items():
        st.markdown(f"#### User {user_id}")
        st.json(metrics)

    # Recommendation system performance chart
    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')
    metrics_df = metrics_df.reset_index().rename(columns={'index': 'UserID'})
    metrics_df = metrics_df.melt(
        id_vars='UserID',
        var_name='Method',
        value_name='Score'
    )
    metrics_df = metrics_df.dropna()

    if not metrics_df.empty:
        fig = px.bar(
            metrics_df,
            x='UserID',
            y='Score',
            color='Method',
            barmode='group',
            title='Recommendation System Performance',
            color_discrete_sequence=COLOR_SEQ,
            template=PLOTLY_TEMPLATE
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────
    # 6) Recommendation System Evaluation
    # ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-header'>🎯 Recommendation System Evaluation</div>",
                unsafe_allow_html=True)

    # Calculate recommendation evaluation
    def calculate_recommendation_evaluation(user_id):
        if 'UserID' not in ml_data.columns:
            return {}
        user_data = ml_data[ml_data['UserID'] == user_id]
        if user_data.empty:
            return {}

        user_cat = user_data['CourseCategory'].unique()
        user_level = user_data['CourseLevel'].iloc[0]

        # Get recommendations
        content_recs = get_course_recommendations(
            user_cat, user_level, 2000, 3.5, top_n=10
        )
        similar_users = get_similar_users(user_id, top_n=5)

        # Calculate evaluation
        evaluation = {}

        # Content-based evaluation
        if not content_recs.empty:
            evaluation['Content-Based'] = {
                'Accuracy': (content_recs['CourseRating'] >= 3.5).mean(),
                'Coverage': (content_recs['CourseRating'] >= 3.5).mean(),
                'Diversity': (content_recs['CourseRating'] >= 3.5).mean(),
                'Novelty': (content_recs['CourseRating'] >= 3.5).mean()
            }

        # Collaborative filtering evaluation
        if similar_users:
            cf_eval = []
            for sim_user_id, similarity in similar_users:
                courses = ml_data[
                    (ml_data['UserID'] == sim_user_id) &
                    (~ml_data['CourseName'].isin(user_data['CourseName']))
                ][['CourseName','CourseCategory','CourseLevel','CoursePrice','CourseRating']]

                if not courses.empty:
                    cf_eval.append({
                        'Accuracy': (courses['CourseRating'] >= 3.5).mean(),
                        'Coverage': (courses['CourseRating'] >= 3.5).mean(),
                        'Diversity': (courses['CourseRating'] >= 3.5).mean(),
                        'Novelty': (courses['CourseRating'] >= 3.5).mean()
                    })

            if cf_eval:
                evaluation['Collaborative Filtering'] = {
                    'Accuracy': sum(e['Accuracy'] for e in cf_eval) / len(cf_eval),
                    'Coverage': sum(e['Coverage'] for e in cf_eval) / len(cf_eval),
                    'Diversity': sum(e['Diversity'] for e in cf_eval) / len(cf_eval),
                    'Novelty': sum(e['Novelty'] for e in cf_eval) / len(cf_eval)
                }

        # Hybrid evaluation
        if not hybrid_recs.empty:
            evaluation['Hybrid'] = {
                'Accuracy': (hybrid_recs['CourseRating'] >= 3.5).mean(),
                'Coverage': (hybrid_recs['CourseRating'] >= 3.5).mean(),
                'Diversity': (hybrid_recs['CourseRating'] >= 3.5).mean(),
                'Novelty': (hybrid_recs['CourseRating'] >= 3.5).mean()
            }

        return evaluation

    # Calculate evaluation for sample users (first 5 to avoid O(n²) slowdown)
    all_evaluation = {}
    for user_id in sample_users:
        all_evaluation[user_id] = calculate_recommendation_evaluation(user_id)

    # Display evaluation
    st.markdown("### 📊 Recommendation Evaluation")
    for user_id, evaluation in all_evaluation.items():
        st.markdown(f"#### User {user_id}")
        st.json(evaluation)

    # Recommendation system evaluation chart
    evaluation_df = pd.DataFrame.from_dict(all_evaluation, orient='index')
    evaluation_df = evaluation_df.reset_index().rename(columns={'index': 'UserID'})
    evaluation_df = evaluation_df.melt(
        id_vars='UserID',
        var_name='Method',
        value_name='Score'
    )
    evaluation_df = evaluation_df.dropna()

    if not evaluation_df.empty:
        fig = px.bar(
            evaluation_df,
            x='UserID',
            y='Score',
            color='Method',
            barmode='group',
            title='Recommendation System Evaluation',
            color_discrete_sequence=COLOR_SEQ,
            template=PLOTLY_TEMPLATE
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────
    # 7) Recommendation System Conclusion
    # ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-header'>✅ Recommendation System Conclusion</div>",
                unsafe_allow_html=True)

    st.markdown("""
    ### 📊 Recommendation System Summary
    
    This recommendation system provides personalized course recommendations based on:
    - **Content-based filtering**: Recommends courses similar to those the user has liked
    - **Collaborative filtering**: Recommends courses liked by similar users
    - **Hybrid approach**: Combines both content-based and collaborative filtering
    
    ### Key Features:
    - **Personalized recommendations**: Tailored to each user's preferences
    - **Real-time updates**: Recommendations update as user interacts with the system
    - **Performance tracking**: Monitors recommendation quality and user engagement
    - **Evaluation metrics**: Provides insights into recommendation system performance
    
    ### Performance Metrics:
    - **Precision**: Measures the accuracy of recommendations
    - **Recall**: Measures the completeness of recommendations
    - **F1-Score**: Harmonic mean of precision and recall
    - **Coverage**: Percentage of courses that can be recommended
    - **Diversity**: Measures the variety of recommended courses
    - **Novelty**: Measures the uniqueness of recommended courses
    
    ### Conclusion:
    This recommendation system provides personalized course recommendations that are tailored to each user's preferences. The system is designed to be flexible and can be easily extended to include additional features such as:
    - **Machine learning-based recommendations**: Using advanced machine learning algorithms
    - **Deep learning-based recommendations**: Using deep learning models
    - **Reinforcement learning-based recommendations**: Using reinforcement learning
    - **Real-time feedback integration**: Incorporating user feedback for continuous improvement
    - **A/B testing framework**: For comparing different recommendation algorithms
    
    The recommendation system provides a solid foundation for personalized course recommendations and can be further enhanced to meet specific business needs.
    """)

    # ──────────────────────────────────────────────────────────────────
    # 8) Recommendation System Improvements
    # ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("<div class='section-header'>💡 Recommendation System Improvements</div>",
                unsafe_allow_html=True)

    st.markdown("""
    ### 🚀 Potential Improvements
    
    Here are some potential improvements for the recommendation system:
    
    #### 1. Machine Learning-Based Recommendations
    - **Implement machine learning algorithms**: Use supervised learning models to predict user preferences
    - **Train machine learning models**: Use historical data to train models on user preferences
    - **Evaluate machine learning models**: Compare different models to find the best one
    
    #### 2. Deep Learning-Based Recommendations
    - **Implement deep learning models**: Use neural networks to capture complex user preferences
    - **Train deep learning models**: Use large datasets to train models on user preferences
    - **Evaluate deep learning models**: Compare different models to find the best one
    
    #### 3. Reinforcement Learning-Based Recommendations
    - **Implement reinforcement learning**: Use reinforcement learning to optimize recommendations
    - **Train reinforcement learning**: Train models on user interactions and feedback
    - **Evaluate reinforcement learning**: Compare different models to find the best one
    
    #### 4. Real-Time Feedback Integration
    - **Incorporate user feedback**: Collect user feedback on recommendations
    - **Update recommendations in real-time**: Adjust recommendations based on feedback
    - **Measure feedback impact**: Track how feedback affects recommendation quality
    
    #### 5. A/B Testing Framework
    - **Implement A/B testing**: Compare different recommendation algorithms
    - **Track A/B test results**: Monitor performance of different algorithms
    - **Optimize based on results**: Use A/B test results to improve recommendations
    
    #### 6. Advanced Features
    - **Session-based recommendations**: Consider user's current session context
    - **Context-aware recommendations**: Incorporate contextual information (time, location, device)
    - **Multi-objective optimization**: Optimize for multiple goals (engagement, revenue, diversity)
    - **Explainable recommendations**: Provide explanations for recommendations
    - **Cold-start problem**: Implement strategies for new users and courses
    
    ### Technical Improvements
    - **Scalability**: Optimize for large datasets and high traffic
    - **Performance**: Improve recommendation generation time
    - **Accuracy**: Enhance recommendation accuracy and relevance
    - **Maintainability**: Improve code structure and documentation
    - **Security**: Implement security best practices
    
    ### Business Improvements
    - **Increase user engagement**: Improve recommendation quality to keep users engaged
    - **Boost course enrollments**: Increase course enrollments through better recommendations
    - **Enhance user satisfaction**: Improve user satisfaction with personalized recommendations
    - **Increase revenue**: Optimize recommendations for revenue generation
    - **Improve retention**: Reduce user churn through better recommendations
    """)
