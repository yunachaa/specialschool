# =====================================================================
# 에듀-타임머신 (Edu-TimeMachine)
# 지역 가변 가중치 머신러닝 기반 특수교육 재정 최적화 시뮬레이터
# Streamlit Cloud 완벽 호환 버전
# =====================================================================

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Streamlit Cloud 호환)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 시스템 폰트 사용 (Linux 서버 환경)
try:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import cm

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    st.error("❌ Plotly 설치 오류. requirements.txt를 확인하세요.")

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# =====================================================================
# 1. 페이지 설정
# =====================================================================
st.set_page_config(
    page_title="에듀-타임머신",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "영재학교 데이터 과학 수행평가 프로젝트"
    }
)

# =====================================================================
# 2. 캐시 설정 (Streamlit Cloud 메모리 최적화)
# =====================================================================
@st.cache_resource
def init_session():
    """세션 초기화"""
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    return st.session_state

session = init_session()

# =====================================================================
# 3. 타이틀
# =====================================================================
st.title("🏆 에듀-타임머신 (Edu-TimeMachine)")
st.markdown("""
### 지역 가변 가중치 머신러닝 기반 특수교육 재정 최적화 시뮬레이터
**영재학교 데이터 과학 수행평가 프로젝트 시연 프로그램**
""")

# =====================================================================
# 4. 데이터 로드 및 전처리 (캐시됨)
# =====================================================================
@st.cache_data(show_spinner=False)
def load_and_process_data():
    """
    CSV 파일 로드 및 전처리
    Streamlit Cloud 환경에서 안정적으로 작동
    """
    try:
        # 현재 디렉토리에서 CSV 파일 찾기
        current_dir = os.getcwd()
        
        file_mapping = {
            '초등': '1. 2020년도_학교현황(학생수,학급수)_초등학교.csv',
            '중': '2. 2020년도_학교현황(학생수,학급수)_중학교.csv',
            '고': '3. 2020년도_학교현황(학생수,학급수)_고등학교.csv',
            '특': '4. 2020년도_학교현황(학생수,학급수)_특수학교.csv',
        }
        
        # 파일 존재 확인 및 로드
        dataframes = {}
        for key, filename in file_mapping.items():
            file_path = os.path.join(current_dir, filename)
            
            if not os.path.exists(file_path):
                # 다른 경로 시도
                possible_paths = [
                    filename,
                    f"./data/{filename}",
                    f"../data/{filename}"
                ]
                
                file_found = False
                for path in possible_paths:
                    if os.path.exists(path):
                        file_path = path
                        file_found = True
                        break
                
                if not file_found:
                    st.warning(f"⚠️ {filename} 파일을 찾을 수 없습니다.")
                    return None, None, None
            
            # 인코딩 자동 감지
            try:
                dataframes[key] = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    dataframes[key] = pd.read_csv(file_path, encoding='euc-kr')
                except:
                    try:
                        dataframes[key] = pd.read_csv(file_path, encoding='cp949')
                    except Exception as e:
                        st.error(f"❌ {filename} 인코딩 오류: {e}")
                        return None, None, None
        
        df_elem = dataframes['초등']
        df_mid = dataframes['중']
        df_high = dataframes['고']
        df_spec = dataframes['특']
        
        # --- 시군구 추출 ---
        for df in [df_elem, df_mid, df_high, df_spec]:
            if '지역' in df.columns:
                df['시군구'] = df['지역'].astype(str).apply(
                    lambda x: x.split()[1] if len(x.split()) > 1 else x
                )
            else:
                st.warning("⚠️ '지역' 컬럼이 없습니다.")
                df['시군구'] = '미분류'
        
        # ---- 초등학교: 학년별 특수학생 수 추출 ----
        for col in ['1학년', '2학년', '3학년', '4학년', '5학년', '6학년']:
            if col in df_elem.columns:
                df_elem[f'{col}_특수'] = pd.to_numeric(
                    df_elem[col].astype(str).str.extract(r'\((\d+)\)', expand=False),
                    errors='coerce'
                ).fillna(0).astype(int)
            else:
                df_elem[f'{col}_특수'] = 0
        
        # 일반 학급 학생수
        if '학생수(계)' in df_elem.columns:
            df_elem['초등_일반_학생'] = pd.to_numeric(
                df_elem['학생수(계)'].astype(str).str.extract(r'^(\d+)', expand=False),
                errors='coerce'
            ).fillna(0).astype(int)
        else:
            df_elem['초등_일반_학생'] = 0
        
        # ---- 중학교: 특수학급 ----
        if '특수학급' in df_mid.columns:
            df_mid['중등_특수_학생'] = pd.to_numeric(
                df_mid['특수학급'].astype(str).str.extract(r'\((\d+)\)', expand=False),
                errors='coerce'
            ).fillna(0).astype(int)
        else:
            df_mid['중등_특수_학생'] = 0
        
        # ---- 고등학교: 특수학급 ----
        if '특수학급' in df_high.columns:
            df_high['고등_특수_학생'] = pd.to_numeric(
                df_high['특수학급'].astype(str).str.extract(r'\((\d+)\)', expand=False),
                errors='coerce'
            ).fillna(0).astype(int)
        else:
            df_high['고등_특수_학생'] = 0
        
        # ---- 특수학교 ----
        if '학생수 총계' in df_spec.columns:
            df_spec['특수학교_학생수'] = pd.to_numeric(
                df_spec['학생수 총계'].astype(str).str.extract(r'^(\d+)', expand=False),
                errors='coerce'
            ).fillna(0).astype(int)
        else:
            df_spec['특수학교_학생수'] = 0
        
        # ---- 지역별 집계 ----
        geo_elem = df_elem.groupby('시군구', as_index=False).agg({
            '1학년_특수': 'sum',
            '2학년_특수': 'sum',
            '3학년_특수': 'sum',
            '4학년_특수': 'sum',
            '5학년_특수': 'sum',
            '6학년_특수': 'sum',
            '초등_일반_학생': 'sum',
            '학교명': 'count'
        }).rename(columns={'학교명': '초등학교수'})
        
        geo_mid = df_mid.groupby('시군구', as_index=False).agg({'중등_특수_학생': 'sum'})
        geo_high = df_high.groupby('시군구', as_index=False).agg({'고등_특수_학생': 'sum'})
        geo_spec = df_spec.groupby('시군구', as_index=False).agg({
            '특수학교_학생수': 'sum',
            '학교명': 'count'
        }).rename(columns={'학교명': '특수학교수'})
        
        # ---- Master Table 병합 ----
        master = pd.merge(geo_elem, geo_mid, on='시군구', how='left')
        master = pd.merge(master, geo_high, on='시군구', how='left')
        master = pd.merge(master, geo_spec, on='시군구', how='left')
        master = master.fillna(0)
        
        return master, df_elem, df_mid
    
    except Exception as e:
        st.error(f"❌ 데이터 로드 오류: {str(e)}")
        return None, None, None

# 데이터 로드
with st.spinner("📊 데이터 로드 중..."):
    master_data, raw_elem, raw_mid = load_and_process_data()

if master_data is None or master_data.empty:
    st.error("""
    ❌ **데이터를 찾을 수 없습니다.**
    
    다음을 확인하세요:
    1. CSV 파일이 GitHub 저장소의 **루트 디렉토리**에 있는가?
    2. 파일명이 정확한가?
       - `1. 2020년도_학교현황(학생수,학급수)_초등학교.csv`
       - `2. 2020년도_학교현황(학생수,학급수)_중학교.csv`
       - `3. 2020년도_학교현황(학생수,학급수)_고등학교.csv`
       - `4. 2020년도_학교현황(학생수,학급수)_특수학교.csv`
    """)
    st.stop()

# =====================================================================
# 5. 머신러닝 모델링
# =====================================================================

# 파생 변수
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

# 머신러닝 모델
X = master_data[[
    '초등_저학년_특수',
    '초등_고학년_특수',
    '초등_일반_학생'
]].fillna(0).values

y = master_data['중고등_특수_합계'].fillna(0).values

if (X != 0).any() and (y != 0).any():
    lr = LinearRegression()
    lr.fit(X, y)
    w_low, w_high, w_pop = lr.coef_[0], lr.coef_[1], lr.coef_[2]
    r2_score = lr.score(X, y)
else:
    w_low, w_high, w_pop = 0.3, 0.4, 0.001
    r2_score = 0.0

# Adaptive FDI
master_data['Adaptive_FDI'] = (
    (master_data['초등_저학년_특수'] * max(w_low, 0)) +
    (master_data['초등_고학년_특수'] * max(w_high, 0)) +
    (master_data['초등_일반_학생'] * max(w_pop, 0))
).clip(lower=0)

# Random Forest
try:
    if (X != 0).any():
        rf_model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
        rf_model.fit(X, y)
        feature_importance = pd.DataFrame({
            'Feature': ['초등_저학년_특수', '초등_고학년_특수', '초등_일반_학생'],
            'Importance': rf_model.feature_importances_
        })
    else:
        raise ValueError("No variance in features")
except:
    feature_importance = pd.DataFrame({
        'Feature': ['초등_저학년_특수', '초등_고학년_특수', '초등_일반_학생'],
        'Importance': [0.33, 0.33, 0.34]
    })

# K-Means
try:
    if master_data[['Adaptive_FDI', '특수학교_학생수']].std().sum() > 0:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(master_data[['Adaptive_FDI', '특수학교_학생수']])
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        master_data['Cluster'] = kmeans.fit_predict(X_scaled)
        danger_cluster = master_data.groupby('Cluster')['Adaptive_FDI'].mean().idxmax()
    else:
        master_data['Cluster'] = 0
        danger_cluster = 0
except:
    master_data['Cluster'] = 0
    danger_cluster = 0

# 지리 좌표
geo_coords = {
    '서울': (37.5665, 126.9780), '부산': (35.1796, 129.0756),
    '대구': (35.8714, 128.5903), '인천': (37.2757, 126.6172),
    '광주': (35.1595, 126.8526), '대전': (36.3504, 127.3845),
    '울산': (35.5384, 129.3114), '경기': (37.2756, 127.0093),
    '강원': (37.2503, 128.5347), '충북': (36.6357, 127.4917),
    '충남': (36.8081, 127.1070), '전북': (35.9078, 127.2679),
    '전남': (34.8118, 126.4635), '경북': (36.5760, 128.9054),
    '경남': (35.4437, 128.2680), '제주': (33.4996, 126.5312),
}

def normalize_region(region_name):
    region_name = str(region_name).strip()
    for key in geo_coords.keys():
        if key in region_name:
            return key
    return region_name

master_data['지역좌표키'] = master_data['시군구'].apply(normalize_region)
master_data['위도'] = master_data['지역좌표키'].map(lambda x: geo_coords.get(x, (37.5, 126.9))[0])
master_data['경도'] = master_data['지역좌표키'].map(lambda x: geo_coords.get(x, (37.5, 126.9))[1])

# =====================================================================
# 6. 사이드바
# =====================================================================
st.sidebar.header("⚙️ 시뮬레이션 설정")
years_ahead = st.sidebar.slider("🎯 예측 연도", 1, 10, 3)
growth_rate = st.sidebar.slider("📈 연간 증가율 (%)", -5.0, 10.0, 0.5, 0.5)

st.sidebar.markdown("---")
st.sidebar.info(f"""
**📊 현재 설정**
- 예측: {years_ahead}년 후
- 증가율: {growth_rate}%
- 지역: {len(master_data)}개
- 모델 R²: {r2_score:.3f}
""")

# =====================================================================
# 7. 시뮬레이션 연산
# =====================================================================
master_data['Simulated_Demand'] = master_data['Adaptive_FDI'] * (1 + (years_ahead * growth_rate / 100))
master_data['공급부족도'] = (master_data['Simulated_Demand'] - master_data['특수학교_학생수']).clip(lower=0)
master_data['위험도_점수'] = (
    (master_data['Simulated_Demand'] / (master_data['Simulated_Demand'].max() + 1)) * 50 +
    (master_data['공급부족도'] / (master_data['공급부족도'].max() + 1e-6)) * 50
)

# =====================================================================
# 8. TAB 구성
# =====================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 1. 머신러닝 분석",
    "🗺️ 2. 3D 히트맵",
    "🔮 3. 시계열 시뮬레이션",
    "💡 4. 거점학교 추천",
    "📈 5. 심화 분석"
])

# =====================================================================
# TAB 1
# =====================================================================
with tab1:
    st.subheader("📊 머신러닝 기반 가변 가중치 분석")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🎯 저학년 가중치", f"{w_low:.4f}")
    with col2:
        st.metric("📚 고학년 가중치", f"{w_high:.4f}")
    with col3:
        st.metric("👥 인구 가중치", f"{w_pop:.6f}")
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.write("### 🤖 특성 중요도 분석")
        fig_importance = plt.figure(figsize=(7, 4))
        ax = fig_importance.add_subplot(111)
        colors_imp = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        ax.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors_imp)
        ax.set_xlabel("중요도", fontsize=10)
        ax.set_title("수요 예측 영향 요소", fontsize=11, fontweight='bold')
        st.pyplot(fig_importance, use_container_width=True)
        plt.close(fig_importance)
    
    with col_right:
        st.write("### 🎯 K-Means 위험군 분류")
        fig_scatter = plt.figure(figsize=(7, 5))
        ax = fig_scatter.add_subplot(111)
        
        colors_cluster = {0: '#3498db', 1: '#2ecc71', 2: '#e74c3c'}
        label_map = {0: '안정권', 1: '주의권', 2: '위험권'}
        
        for cluster_id in [0, 1, 2]:
            mask = master_data['Cluster'] == cluster_id
            ax.scatter(
                master_data[mask]['Adaptive_FDI'],
                master_data[mask]['특수학교_학생수'],
                s=80, alpha=0.6,
                label=label_map[cluster_id],
                color=colors_cluster[cluster_id]
            )
        
        ax.set_xlabel("Adaptive FDI", fontsize=10)
        ax.set_ylabel("특수학교 수용인원", fontsize=10)
        ax.set_title("지역별 인프라 위험도", fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig_scatter, use_container_width=True)
        plt.close(fig_scatter)

# =====================================================================
# TAB 2
# =====================================================================
with tab2:
    st.subheader("🗺️ 전국 지역별 3D 히트맵")
    
    try:
        fig_3d = go.Figure()
        
        master_3d = master_data.sort_values('Adaptive_FDI', ascending=False).head(25)
        
        if master_3d['위험도_점수'].max() > master_3d['위험도_점수'].min():
            risk_norm = (master_3d['위험도_점수'] - master_3d['위험도_점수'].min()) / (
                master_3d['위험도_점수'].max() - master_3d['위험도_점수'].min()
            )
        else:
            risk_norm = pd.Series([0.5] * len(master_3d))
        
        fig_3d.add_trace(go.Scatter3d(
            x=master_3d['위도'],
            y=master_3d['경도'],
            z=master_3d['Simulated_Demand'],
            mode='markers',
            marker=dict(
                size=8,
                color=risk_norm,
                colorscale='RdYlBu_r',
                showscale=True,
                colorbar=dict(title="위험도", thickness=12, len=0.6),
                opacity=0.8,
                line=dict(width=0.5, color='white')
            ),
            text=master_3d['시군구'],
            hovertemplate='<b>%{text}</b><br>수요: %{z:.0f}<extra></extra>',
            name='지역'
        ))
        
        fig_3d.update_layout(
            title="<b>3D 특수교육 인프라 수요 지도</b>",
            scene=dict(
                xaxis_title="위도",
                yaxis_title="경도",
                zaxis_title="미래 수요",
                camera=dict(eye=dict(x=1.3, y=1.3, z=1.2))
            ),
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig_3d, use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ 3D 시각화 오류: {str(e)}")

# =====================================================================
# TAB 3
# =====================================================================
with tab3:
    st.subheader("🔮 시계열 수요 전이")
    
    col_sim1, col_sim2 = st.columns(2)
    
    with col_sim1:
        st.write("### 📊 상위 10개 위험 지역")
        danger_regions = master_data.nlargest(10, 'Simulated_Demand')[
            ['시군구', 'Adaptive_FDI', 'Simulated_Demand']
        ].copy()
        
        fig_timeline = plt.figure(figsize=(8, 4))
        ax = fig_timeline.add_subplot(111)
        x_pos = np.arange(len(danger_regions))
        width = 0.35
        
        ax.bar(x_pos - width/2, danger_regions['Adaptive_FDI'], width, label='현재', color='#3498db', alpha=0.8)
        ax.bar(x_pos + width/2, danger_regions['Simulated_Demand'], width, label=f'{years_ahead}년 후', color='#e74c3c', alpha=0.8)
        
        ax.set_xlabel("지역", fontsize=9)
        ax.set_ylabel("수요 (명)", fontsize=9)
        ax.set_title(f"{years_ahead}년 후 수요 변화", fontsize=10, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(danger_regions['시군구'], rotation=45, ha='right', fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig_timeline, use_container_width=True)
        plt.close(fig_timeline)
    
    with col_sim2:
        st.write("### 🎯 공급부족도 TOP 10")
        shortage_rank = master_data.nlargest(10, '공급부족도')[
            ['시군구', '공급부족도', '특수학교_학생수']
        ].reset_index(drop=True)
        shortage_rank.index = range(1, len(shortage_rank) + 1)
        st.dataframe(shortage_rank, use_container_width=True)

# =====================================================================
# TAB 4
# =====================================================================
with tab4:
    st.subheader("💡 거점학교 추천")
    
    selected_region = st.selectbox("🎯 지역 선택", sorted(master_data['시군구'].unique()))
    
    if selected_region:
        region_info = master_data[master_data['시군구'] == selected_region].iloc[0]
        
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            st.metric("현재", int(region_info['중고등_특수_합계']))
        with col_r2:
            st.metric("특수학교 수용", int(region_info['특수학교_학생수']))
        with col_r3:
            st.metric(f"{years_ahead}년 후", int(region_info['Simulated_Demand']))
        with col_r4:
            shortage = max(0, region_info['Simulated_Demand'] - region_info['특수학교_학생수'])
            st.metric("부족도", int(shortage))
        
        st.markdown("---")
        
        region_schools = raw_elem[raw_elem['시군구'] == selected_region].copy()
        
        if not region_schools.empty:
            if '학급당학생수' in region_schools.columns:
                region_schools['학급당학생수'] = pd.to_numeric(
                    region_schools['학급당학생수'], errors='coerce'
                ).fillna(25)
            else:
                region_schools['학급당학생수'] = 25
            
            region_schools['유휴공간_점수'] = (40 - region_schools['학급당학생수']).clip(0)
            region_schools['6학년_특수'] = region_schools.get('6학년_특수', 0)
            region_schools['거점_적합도'] = (
                region_schools['유휴공간_점수'] * 0.6 +
                region_schools['6학년_특수'] * 0.4
            )
            
            top = region_schools.nlargest(3, '거점_적합도')
            
            st.write(f"### 🏆 {selected_region} 상위 3개 학교")
            for rank, (idx, school) in enumerate(top.iterrows(), 1):
                st.write(f"**{rank}. {school.get('학교명', 'N/A')}** (점수: {school['거점_적합도']:.1f})")
        else:
            st.info(f"데이터가 없습니다.")

# =====================================================================
# TAB 5
# =====================================================================
with tab5:
    st.subheader("📈 심화 분석")
    
    col_deep1, col_deep2 = st.columns(2)
    
    with col_deep1:
        st.write("### 위험군 분류")
        cluster_labels = {0: '안정권', 1: '주의권', 2: '위험권'}
        
        for cid in [0, 1, 2]:
            cluster_data = master_data[master_data['Cluster'] == cid]
            if len(cluster_data) > 0:
                st.write(f"**{cluster_labels[cid]}**: {len(cluster_data)}개 지역")
                st.caption(f"평균 수요: {cluster_data['Simulated_Demand'].mean():.1f}명")
    
    with col_deep2:
        st.write("### 정책 제언")
        danger_zones = master_data[master_data['Cluster'] == danger_cluster]
        
        if len(danger_zones) > 0:
            st.write("🔴 **1순위 (위험권)**")
            st.write(f"- 지역: {', '.join(danger_zones['시군구'].head(3))}")
            st.write(f"- 부족도: {danger_zones['공급부족도'].sum():.0f}명")
            st.write("- 액션: 즉시 거점학교 신설")

st.markdown("---")
st.markdown("© 2024 에듀-타임머신 | Streamlit Cloud")
