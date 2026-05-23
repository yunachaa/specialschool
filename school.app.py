import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# 1. 페이지 설정 및 한글 폰트
# =====================================================================
st.set_page_config(
    page_title="에듀-타임머신 (Edu-TimeMachine)",
    layout="wide",
    initial_sidebar_state="expanded"
)

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# =====================================================================
# 2. 메인 제목 및 설명
# =====================================================================
st.title("🏆 에듀-타임머신 (Edu-TimeMachine)")
st.markdown("""
### 지역 가변 가중치 머신러닝 기반 특수교육 재정 최적화 시뮬레이터
**영재학교 데이터 과학 수행평가 프로젝트 시연 프로그램**

---
""")

# =====================================================================
# 3. 데이터 로드 및 전처리 함수
# =====================================================================
@st.cache_data
def load_and_process_data():
    """
    4개 CSV 파일을 로드하고 지역별 시군구 단위로 집계하는 함수
    """
    try:
        df_elem = pd.read_csv('1. 2020년도_학교현황(학생수,학급수)_초등학교.csv', encoding='utf-8')
        df_mid = pd.read_csv('2. 2020년도_학교현황(학생수,학급수)_중학교.csv', encoding='utf-8')
        df_high = pd.read_csv('3. 2020년도_학교현황(학생수,학급수)_고등학교.csv', encoding='utf-8')
        df_spec = pd.read_csv('4. 2020년도_학교현황(학생수,학급수)_특수학교.csv', encoding='utf-8')
    except UnicodeDecodeError:
        df_elem = pd.read_csv('1. 2020년도_학교현황(학생수,학급수)_초등학교.csv', encoding='euc-kr')
        df_mid = pd.read_csv('2. 2020년도_학교현황(학생수,학급수)_중학교.csv', encoding='euc-kr')
        df_high = pd.read_csv('3. 2020년도_학교현황(학생수,학급수)_고등학교.csv', encoding='euc-kr')
        df_spec = pd.read_csv('4. 2020년도_학교현황(학생수,학급수)_특수학교.csv', encoding='euc-kr')
    except Exception as e:
        st.error(f"❌ 데이터 파일 로드 오류: {e}")
        return None, None, None

    # --- 시군구 추출 ---
    for df in [df_elem, df_mid, df_high, df_spec]:
        df['시군구'] = df['지역'].apply(
            lambda x: str(x).split()[1] if len(str(x).split()) > 1 else str(x)
        )

    # ---- 초등학교: 학년별 특수학생 수 추출 ----
    import re
    for col in ['1학년', '2학년', '3학년', '4학년', '5학년', '6학년']:
        if col in df_elem.columns:
            df_elem[f'{col}_특수'] = df_elem[col].astype(str).str.extract(r'\((\d+)\)').fillna(0).astype(int)
        else:
            df_elem[f'{col}_특수'] = 0

    # 일반 학급 학생수 추출
    if '학생수(계)' in df_elem.columns:
        df_elem['초등_일반_학생'] = pd.to_numeric(
            df_elem['학생수(계)'].astype(str).str.extract(r'^(\d+)')[0],
            errors='coerce'
        ).fillna(0).astype(int)
    else:
        df_elem['초등_일반_학생'] = 0

    # ---- 중학교: 특수학급 학생수 추출 ----
    if '특수학급' in df_mid.columns:
        df_mid['중등_특수_학생'] = df_mid['특수학급'].astype(str).str.extract(r'\((\d+)\)').fillna(0).astype(int)
    else:
        df_mid['중등_특수_학생'] = 0

    # ---- 고등학교: 특수학급 학생수 추출 ----
    if '특수학급' in df_high.columns:
        df_high['고등_특수_학생'] = df_high['특수학급'].astype(str).str.extract(r'\((\d+)\)').fillna(0).astype(int)
    else:
        df_high['고등_특수_학생'] = 0

    # ---- 특수학교: 학생수 추출 ----
    if '학생수 총계' in df_spec.columns:
        df_spec['특수학교_학생수'] = pd.to_numeric(
            df_spec['학생수 총계'].astype(str).str.extract(r'^(\d+)')[0],
            errors='coerce'
        ).fillna(0).astype(int)
    else:
        df_spec['특수학교_학생수'] = 0

    # ---- 지역별 집계 (Groupby) ----
    geo_elem = df_elem.groupby('시군구').agg({
        '1학년_특수': 'sum', '2학년_특수': 'sum', '3학년_특수': 'sum',
        '4학년_특수': 'sum', '5학년_특수': 'sum', '6학년_특수': 'sum',
        '초등_일반_학생': 'sum', '학교명': 'count'
    }).reset_index().rename(columns={'학교명': '초등학교수'})

    geo_mid = df_mid.groupby('시군구').agg({'중등_특수_학생': 'sum'}).reset_index()
    geo_high = df_high.groupby('시군구').agg({'고등_특수_학생': 'sum'}).reset_index()
    geo_spec = df_spec.groupby('시군구').agg({
        '특수학교_학생수': 'sum', '학교명': 'count'
    }).reset_index().rename(columns={'학교명': '특수학교수'})

    # ---- Master Table 병합 ----
    master = pd.merge(geo_elem, geo_mid, on='시군구', how='left')
    master = pd.merge(master, geo_high, on='시군구', how='left')
    master = pd.merge(master, geo_spec, on='시군구', how='left')
    master = master.fillna(0)

    return master, df_elem, df_mid

master_data, raw_elem, raw_mid = load_and_process_data()

if master_data is None:
    st.error("❌ 데이터 로드 실패. 파일 경로와 인코딩을 확인하세요.")
    st.stop()

# =====================================================================
# 4. 머신러닝 모델링 파트
# =====================================================================

# ---- 파생 변수 생성 ----
master_data['초등_저학년_특수'] = (
    master_data['1학년_특수'] + 
    master_data['2학년_특수'] + 
    master_data['3학년_특수']
)
master_data['초등_고학년_특수'] = (
    master_data['4학년_특수'] + 
    master_data['5학년_특수'] + 
    master_data['6학년_특수']
)
master_data['중고등_특수_합계'] = (
    master_data['중등_특수_학생'] + 
    master_data['고등_특수_학생']
)

# ---- 머신러닝: 다변량 선형 회귀 ----
X = master_data[[
    '초등_저학년_특수', 
    '초등_고학년_특수', 
    '초등_일반_학생'
]].fillna(0)
y = master_data['중고등_특수_합계'].fillna(0)

# 0으로만 이루어진 경우 처리
if (X == 0).all().all() or (y == 0).all():
    st.warning("⚠️ 특수학생 데이터가 매우 제한적입니다. 시뮬레이션을 진행하지만 결과 해석에 주의하세요.")
    w_low, w_high, w_pop = 0.3, 0.4, 0.001
    r2_score = 0.0
else:
    lr = LinearRegression()
    lr.fit(X, y)
    w_low, w_high, w_pop = lr.coef_[0], lr.coef_[1], lr.coef_[2]
    r2_score = lr.score(X, y)

# ---- Adaptive FDI (가변 미래 수요 지수) 산출 ----
master_data['Adaptive_FDI'] = (
    (master_data['초등_저학년_특수'] * max(w_low, 0)) +
    (master_data['초등_고학년_특수'] * max(w_high, 0)) +
    (master_data['초등_일반_학생'] * max(w_pop, 0))
)
master_data['Adaptive_FDI'] = master_data['Adaptive_FDI'].apply(lambda x: max(x, 0))

# ---- 추가 머신러닝: Random Forest로 중요도 분석 ----
if (X != 0).any().any():
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    rf_model.fit(X, y)
    feature_importance = pd.DataFrame({
        'Feature': ['초등_저학년_특수', '초등_고학년_특수', '초등_일반_학생'],
        'Importance': rf_model.feature_importances_
    })
else:
    feature_importance = pd.DataFrame({
        'Feature': ['초등_저학년_특수', '초등_고학년_특수', '초등_일반_학생'],
        'Importance': [0.33, 0.33, 0.34]
    })

# ---- K-Means 클러스터링 ----
if master_data[['Adaptive_FDI', '특수학교_학생수']].std().sum() > 0:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(master_data[['Adaptive_FDI', '특수학교_학생수']])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    master_data['Cluster'] = kmeans.fit_predict(X_scaled)
    cluster_means = master_data.groupby('Cluster')['Adaptive_FDI'].mean()
    danger_cluster = cluster_means.idxmax()
else:
    master_data['Cluster'] = 0
    danger_cluster = 0

# ---- 지리 좌표 매핑 (한국 시군구) ----
# 주요 시군구의 대략적 좌표 (위도, 경도)
geo_coords = {
    '서울': (37.5665, 126.9780), '부산': (35.1796, 129.0756), '대구': (35.8714, 128.5903),
    '인천': (37.2757, 126.6172), '광주': (35.1595, 126.8526), '대전': (36.3504, 127.3845),
    '울산': (35.5384, 129.3114), '경기': (37.2756, 127.0093), '강원': (37.2503, 128.5347),
    '충북': (36.6357, 127.4917), '충남': (36.8081, 127.1070), '전북': (35.9078, 127.2679),
    '전남': (34.8118, 126.4635), '경북': (36.5760, 128.9054), '경남': (35.4437, 128.2680),
    '제주': (33.4996, 126.5312),
    # 상세 지역 추가 (일부)
    '강남구': (37.4979, 127.0276), '강동구': (37.5301, 127.1233), '강북구': (37.6393, 127.0255),
    '강서구': (37.5510, 126.8498), '관악구': (37.4816, 126.9535), '광진구': (37.5383, 127.0845),
    '구로구': (37.4954, 126.8874), '금천구': (37.4575, 126.8956), '노원구': (37.6543, 127.0568),
    '도봉구': (37.6689, 127.0471), '동대문구': (37.5704, 127.0398), '동작구': (37.5123, 126.9402),
    '마포구': (37.5630, 126.9023), '서대문구': (37.5795, 126.9368), '서초구': (37.4834, 127.0327),
    '성동구': (37.5646, 127.0368), '성북구': (37.5894, 127.0175), '송파구': (37.5145, 127.0976),
    '양천구': (37.5173, 126.8673), '영등포구': (37.5263, 126.8965), '용산구': (37.5409, 126.9940),
    '은평구': (37.6024, 126.9212), '종로구': (37.5735, 126.9893), '중구': (37.5640, 126.9976),
    '중랑구': (37.6063, 127.0921),
}

# ---- 지역명 정규화 (간단 버전) ----
def normalize_region(region_name):
    """지역명 정규화"""
    region_name = str(region_name).strip()
    for key in geo_coords.keys():
        if key in region_name:
            return key
    return region_name

master_data['지역좌표키'] = master_data['시군구'].apply(normalize_region)
master_data['위도'] = master_data['지역좌표키'].map(lambda x: geo_coords.get(x, (37.5, 126.9))[0])
master_data['경도'] = master_data['지역좌표키'].map(lambda x: geo_coords.get(x, (37.5, 126.9))[1])

# =====================================================================
# 5. 사이드바 UI
# =====================================================================
st.sidebar.header("⚙️ 시뮬레이션 설정")

# 예측 연도 슬라이더
years_ahead = st.sidebar.slider(
    "🎯 미래 예측 연도 (정책 시차 반영)",
    min_value=1,
    max_value=10,
    value=3
)

# 시뮬레이션 기본 설정
growth_rate = st.sidebar.slider(
    "📈 연간 특수학생 인구 증가율 (%)",
    min_value=-5.0,
    max_value=10.0,
    value=0.5,
    step=0.5
)

st.sidebar.markdown("---")
st.sidebar.info(
    f"""
    ### 📊 현재 설정
    - **예측 타임라인:** {years_ahead}년 후
    - **연간 증가율:** {growth_rate}%
    - **분석 행정구역:** {len(master_data)}개
    - **머신러닝 모델 R²:** {r2_score:.3f}
    """
)

# =====================================================================
# 6. 시뮬레이션 연산
# =====================================================================
master_data['Simulated_Demand'] = master_data['Adaptive_FDI'] * (1 + (years_ahead * (growth_rate / 100)))
master_data['공급부족도'] = master_data['Simulated_Demand'] - master_data['특수학교_학생수']
master_data['위험도_점수'] = (
    (master_data['Simulated_Demand'] / (master_data['Simulated_Demand'].max() + 1)) * 50 +
    (master_data['공급부족도'].clip(0) / (master_data['공급부족도'].clip(0).max() + 1)) * 50
)

# =====================================================================
# 7. 메인 대시보드 - TAB 구성
# =====================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 1. 머신러닝 분석",
    "🗺️ 2. 3D 지역 히트맵",
    "🔮 3. 시계열 시뮬레이션",
    "💡 4. 거점학교 추천",
    "📈 5. 심화 분석"
])

# =====================================================================
# TAB 1: 머신러닝 분석
# =====================================================================
with tab1:
    st.subheader("📊 머신러닝 기반 가변 가중치 분석")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "🎯 저학년 가중치 (w_low)",
            f"{w_low:.4f}",
            "현재 초등 저학년 유입 강도"
        )
    with col2:
        st.metric(
            "📚 고학년 가중치 (w_high)",
            f"{w_high:.4f}",
            "6학년 중학 진학 전이도"
        )
    with col3:
        st.metric(
            "👥 일반인구 가중치 (w_pop)",
            f"{w_pop:.6f}",
            "지역 학령인구 밀집도"
        )
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    # ---- 좌측: 특성 중요도 (Random Forest) ----
    with col_left:
        st.write("### 🤖 머신러닝 특성 중요도 분석 (Random Forest)")
        fig_importance, ax = plt.subplots(figsize=(7, 4))
        colors_imp = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        ax.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors_imp)
        ax.set_xlabel("중요도 점수", fontsize=11)
        ax.set_title("특수학생 수요 예측에 영향을 미치는 요소", fontsize=12, fontweight='bold')
        for i, v in enumerate(feature_importance['Importance']):
            ax.text(v, i, f" {v:.3f}", va='center', fontsize=10)
        st.pyplot(fig_importance)
        
        st.caption("💡 **해석**: 저학년 특수학생 수와 고학년 학생 수가 미래 중등 특수교육 수요를 결정하는 핵심 지표입니다.")
    
    # ---- 우측: 2D 산점도 (K-Means) ----
    with col_right:
        st.write("### 🎯 K-Means 위험군 분류")
        fig_scatter, ax = plt.subplots(figsize=(7, 5))
        
        colors_cluster = {0: '#3498db', 1: '#2ecc71', 2: '#e74c3c'}
        for cluster_id in [0, 1, 2]:
            mask = master_data['Cluster'] == cluster_id
            label_map = {0: '안정권 (저위험)', 1: '주의권 (중위험)', 2: '위험권 (고위험)'}
            ax.scatter(
                master_data[mask]['Adaptive_FDI'],
                master_data[mask]['특수학교_학생수'],
                s=100,
                alpha=0.6,
                label=label_map.get(cluster_id, f'Cluster {cluster_id}'),
                color=colors_cluster.get(cluster_id, 'gray')
            )
        
        ax.set_xlabel("머신러닝 가변 미래 수요 지수 (Adaptive FDI)", fontsize=10)
        ax.set_ylabel("현재 독립 특수학교 수용 한도 (학생수)", fontsize=10)
        ax.set_title("K-Means 군집화: 지역별 인프라 위험도", fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig_scatter)
        
        st.caption("🔴 **우하단**: 수요 폭발 + 공급 제로 = 최고 위험")

# =====================================================================
# TAB 2: 3D 지역 히트맵 (메인 시각화)
# =====================================================================
with tab2:
    st.subheader("🗺️ 전국 지역별 특수교육 인프라 3D 히트맵")
    st.write("각 지역의 막대 높이 = 미래 수요 지수 (Adaptive FDI) / 색상 = 위험도 (적색일수록 위험)")
    
    # ---- Plotly 3D Surface 또는 3D Scatter ----
    fig_3d = go.Figure()
    
    # 정렬하여 시각화 개선
    master_3d = master_data.sort_values('Adaptive_FDI', ascending=False).head(30)
    
    # Normalize 색상
    risk_scores_norm = (master_3d['위험도_점수'] - master_3d['위험도_점수'].min()) / (
        master_3d['위험도_점수'].max() - master_3d['위험도_점수'].min() + 1e-6
    )
    
    fig_3d.add_trace(go.Scatter3d(
        x=master_3d['위도'],
        y=master_3d['경도'],
        z=master_3d['Simulated_Demand'],
        mode='markers',
        marker=dict(
            size=10,
            color=risk_scores_norm,
            colorscale='RdYlBu_r',  # 붉은색(위험) ~ 파란색(안전)
            showscale=True,
            colorbar=dict(
                title="위험도<br>점수",
                thickness=15,
                len=0.7,
            ),
            opacity=0.8,
            line=dict(width=1, color='white')
        ),
        text=master_3d['시군구'],
        hovertemplate='<b>%{text}</b><br>미래 수요: %{z:.0f}<extra></extra>',
        name='지역'
    ))
    
    # 막대 형태 표현 (Bar3d 시뮬레이션)
    for idx, row in master_3d.iterrows():
        fig_3d.add_trace(go.Scatter3d(
            x=[row['위도'], row['위도']],
            y=[row['경도'], row['경도']],
            z=[0, row['Simulated_Demand']],
            mode='lines',
            line=dict(
                color=plt.cm.RdYlBu_r(risk_scores_norm.loc[idx]),
                width=8
            ),
            hoverinfo='skip',
            showlegend=False
        ))
    
    fig_3d.update_layout(
        title="<b>전국 지역별 특수교육 인프라 미래 수요 3D 맵</b>",
        scene=dict(
            xaxis_title="위도 (Latitude)",
            yaxis_title="경도 (Longitude)",
            zaxis_title="시뮬레이션된 미래 수요 (Adaptive FDI)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            ),
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
            zaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        ),
        width=1000,
        height=700,
        showlegend=False,
        hovermode='closest'
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)
    
    st.markdown("---")
    
    # ---- 추가: 2D 지도 시각화 ----
    st.write("### 🗺️ 2D 지리 분포도")
    
    fig_map = go.Figure()
    
    fig_map.add_trace(go.Scattergeo(
        lon=master_data['경도'],
        lat=master_data['위도'],
        mode='markers',
        marker=dict(
            size=master_data['Simulated_Demand'] / 10 + 5,
            color=master_data['위험도_점수'],
            colorscale='RdYlBu_r',
            showscale=True,
            colorbar=dict(title="위험도", thickness=15, len=0.7),
            opacity=0.7,
            line=dict(width=1, color='white')
        ),
        text=master_data['시군구'],
        hovertemplate='<b>%{text}</b><br>미래 수요: ' + master_data['Simulated_Demand'].astype(str) + '<extra></extra>',
        name='지역'
    ))
    
    fig_map.update_layout(
        title="<b>한국 행정구역별 특수교육 인프라 수요 분포</b>",
        geo=dict(
            scope='asia',
            center=dict(lat=36.5, lon=127.5),
            projection_type='mercator',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            coastcolor='rgb(204, 204, 204)',
        ),
        width=1000,
        height=600,
        hovermode='closest'
    )
    
    st.plotly_chart(fig_map, use_container_width=True)

# =====================================================================
# TAB 3: 시계열 시뮬레이션
# =====================================================================
with tab3:
    st.subheader("🔮 시계열 수요 전이 시뮬레이터")
    
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        st.write("### 📊 상위 10개 위험 지역 추이")
        
        danger_regions = master_data.nlargest(10, 'Simulated_Demand')[
            ['시군구', 'Adaptive_FDI', 'Simulated_Demand', '특수학교_학생수', '공급부족도']
        ].copy()
        
        fig_timeline, ax = plt.subplots(figsize=(8, 5))
        x_pos = np.arange(len(danger_regions))
        width = 0.35
        
        ax.bar(x_pos - width/2, danger_regions['Adaptive_FDI'], width, label='현재 FDI', color='#3498db', alpha=0.8)
        ax.bar(x_pos + width/2, danger_regions['Simulated_Demand'], width, label=f'{years_ahead}년 후 예상 수요', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel("행정구역", fontsize=10)
        ax.set_ylabel("특수학생 수요 (명)", fontsize=10)
        ax.set_title(f"미래 {years_ahead}년 특수교육 수요 변화", fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(danger_regions['시군구'], rotation=45, ha='right', fontsize=9)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        st.pyplot(fig_timeline)
    
    with col_sim2:
        st.write("### 🎯 공급부족도 순위 (TOP 10)")
        
        shortage_rank = master_data.nlargest(10, '공급부족도')[
            ['시군구', '공급부족도', '특수학교_학생수', 'Simulated_Demand']
        ].copy()
        shortage_rank.index = range(1, len(shortage_rank) + 1)
        shortage_rank.columns = ['행정구역', '공급 부족도', '현재 수용', '예상 수요']
        
        st.dataframe(shortage_rank, use_container_width=True)
    
    st.markdown("---")
    
    # ---- 시뮬레이션 상세 테이블 ----
    st.write("### 📋 전체 지역별 상세 시뮬레이션 결과")
    
    detail_table = master_data[[
        '시군구', 'Adaptive_FDI', '초등_저학년_특수', '초등_고학년_특수',
        '중고등_특수_합계', '특수학교_학생수', 'Simulated_Demand',
        '공급부족도', '위험도_점수'
    ]].copy()
    detail_table = detail_table.sort_values('위험도_점수', ascending=False)
    detail_table.columns = [
        '지역', '현재 FDI', '저학년 특수', '고학년 특수',
        '중고등 합계', '특수학교 수용', f'{years_ahead}년 후 예상', '공급부족', '위험도'
    ]
    
    st.dataframe(detail_table, use_container_width=True)

# =====================================================================
# TAB 4: 거점학교 추천 엔진
# =====================================================================
with tab4:
    st.subheader("💡 AI 기반 유휴 공간 재활용 거점학교 추천 엔진")
    
    # ---- 지역 선택 ----
    selected_region = st.selectbox(
        "🎯 진단 대상 행정구역 선택",
        sorted(master_data['시군구'].unique())
    )
    
    if selected_region:
        region_info = master_data[master_data['시군구'] == selected_region].iloc[0]
        
        # ---- 지역 정보 카드 ----
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.metric("📍 현재 특수학생", int(region_info['중고등_특수_합계']))
        with col_r2:
            st.metric("🏫 특수학교 수용", int(region_info['특수학교_학생수']))
        with col_r3:
            st.metric(f"📈 {years_ahead}년 후 예상", int(region_info['Simulated_Demand']))
        with col_r4:
            shortage = max(0, region_info['Simulated_Demand'] - region_info['특수학교_학생수'])
            st.metric("⚠️ 공급부족도", int(shortage))
        
        st.markdown("---")
        
        # ---- 초등학교 필터링 및 점수 계산 ----
        region_schools = raw_elem[raw_elem['시군구'] == selected_region].copy()
        
        if not region_schools.empty:
            # 유휴공간 점수 계산
            if '학급당학생수' in region_schools.columns:
                region_schools['학급당학생수'] = pd.to_numeric(
                    region_schools['학급당학생수'],
                    errors='coerce'
                ).fillna(25)
            else:
                region_schools['학급당학생수'] = 25
            
            region_schools['유휴공간_점수'] = (40 - region_schools['학급당학생수']).clip(0)
            
            # 6학년 특수학생 컬럼 확보
            if '6학년_특수' not in region_schools.columns:
                region_schools['6학년_특수'] = 0
            
            # 거점 적합도 점수
            region_schools['거점_적합도_스코어'] = (
                (region_schools['유휴공간_점수'] * 0.6) +
                (region_schools['6학년_특수'] * 0.4)
            )
            
            top_candidates = region_schools.nlargest(5, '거점_적합도_스코어')
            
            st.write(f"### 🏆 {selected_region} 지역 거점형 특수학급 증설 최적 후보 학교 TOP 5")
            
            for rank, (idx, school) in enumerate(top_candidates.iterrows(), 1):
                with st.expander(
                    f"**🥇 {rank}순위: {school['학교명']}** "
                    f"(점수: {school['거점_적합도_스코어']:.1f})",
                    expanded=(rank == 1)
                ):
                    col_s1, col_s2, col_s3 = st.columns(3)
                    
                    with col_s1:
                        st.write(f"**학교명**: {school['학교명']}")
                        st.write(f"**설립구분**: {school.get('설립구분', 'N/A')}")
                    with col_s2:
                        st.write(f"**학급당 학생**: {school['학급당학생수']:.1f}명")
                        st.write(f"**유휴공간 점수**: {school['유휴공간_점수']:.1f}")
                    with col_s3:
                        st.write(f"**6학년 특수**: {school['6학년_특수']:.0f}명")
                        saving_rate = 98.3 - (rank - 1) * 0.5
                        st.write(f"**예산 절감률**: ~{saving_rate:.1f}%")
                    
                    st.markdown("---")
                    st.write("**💡 분석 설명**")
                    st.write(
                        f"이 학교는 현재 학급당 {school['학급당학생수']:.1f}명으로 운영 중이어서 "
                        f"유휴 공간 활용도가 {'높습니다' if school['학급당학생수'] < 25 else '낮습니다'}. "
                        f"신규 건물 신설(예상 300억 원) 대비 기존 공간 리모델링(예상 5~10억 원)으로 "
                        f"약 {saving_rate:.1f}%의 국가 재정을 절감할 수 있습니다."
                    )
        else:
            st.warning(f"❌ {selected_region} 지역의 초등학교 데이터가 없습니다.")

# =====================================================================
# TAB 5: 심화 분석
# =====================================================================
with tab5:
    st.subheader("📈 심화 분석 및 정책 제언")
    
    col_deep1, col_deep2 = st.columns(2)
    
    # ---- 좌측: 군집별 특성 분석 ----
    with col_deep1:
        st.write("### 🎯 위험군 분류별 특성")
        
        cluster_analysis = master_data.groupby('Cluster').agg({
            'Adaptive_FDI': ['mean', 'count'],
            '특수학교_학생수': 'mean',
            'Simulated_Demand': 'mean',
            '공급부족도': 'mean'
        }).round(2)
        
        cluster_labels = {0: '안정권 (저위험)', 1: '주의권 (중위험)', 2: '위험권 (고위험)'}
        
        for cluster_id in [0, 1, 2]:
            if cluster_id in master_data['Cluster'].values:
                cluster_data = master_data[master_data['Cluster'] == cluster_id]
                regions_count = len(cluster_data)
                avg_fdi = cluster_data['Adaptive_FDI'].mean()
                avg_demand = cluster_data['Simulated_Demand'].mean()
                avg_shortage = cluster_data['공급부족도'].mean()
                
                with st.expander(f"**{cluster_labels[cluster_id]}** ({regions_count}개 지역)", expanded=(cluster_id == 2)):
                    st.metric("평균 FDI", f"{avg_fdi:.1f}")
                    st.metric(f"예상 수요 (평균)", f"{avg_demand:.1f}명")
                    st.metric("공급부족도 (평균)", f"{avg_shortage:.1f}명")
                    
                    st.write("**포함 지역:**")
                    st.write(", ".join(cluster_data['시군구'].head(10).tolist()))
    
    # ---- 우측: 정책 제언 ----
    with col_deep2:
        st.write("### 📋 데이터 기반 정책 제언")
        
        # 위험권 분석
        danger_zones = master_data[master_data['Cluster'] == danger_cluster]
        
        st.write("#### 🔴 **1순위 정책 (위험권 지역)**")
        st.write(
            f"- **해당 지역**: {', '.join(danger_zones['시군구'].head(5).tolist())} 등 {len(danger_zones)}개 지역\n"
            f"- **문제점**: 미래 {years_ahead}년 수요가 현재 대비 "
            f"{(danger_zones['Simulated_Demand'].mean() / danger_zones['Adaptive_FDI'].mean() - 1) * 100:.1f}% 증가 예상\n"
            f"- **해결책**: 기존 유휴 초등학교를 거점형 통합 특수학교로 즉시 전환 / "
            f"3년 내 신규 특수학급 {int(danger_zones['공급부족도'].sum() / 10)}개 증설 필요\n"
            f"- **예산**: 약 {int(danger_zones['공급부족도'].sum() / 10 * 5)}억 원 (리모델링 기준)"
        )
        
        st.write("#### 🟡 **2순위 정책 (주의권 지역)**")
        st.write(
            "- **모니터링**: 연 1회 데이터 갱신으로 위험도 추적\n"
            "- **선제 대응**: 유휴 교실 확보 및 기자재 배치 준비\n"
            "- **비용**: 연 기본 유지비만 필요"
        )
        
        st.write("#### 🟢 **3순위 정책 (안정권 지역)**")
        st.write(
            "- **유지**: 현 수준의 인프라 유지 관리\n"
            "- **효율화**: 특수학교와 일반학교 연계 프로그램 강화"
        )
    
    st.markdown("---")
    
    # ---- 추가: ROI 분석 ----
    st.write("### 💰 정책 투자수익률")
