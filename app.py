import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pymongo import MongoClient
import bcrypt
import time
from functools import lru_cache
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle
import os
from bson import ObjectId

# ------------------ Utility Fix for ObjectId ------------------
def serialize_mongo(doc):
    """Convert ObjectId and other non-serializable types to strings."""
    if isinstance(doc, list):
        return [serialize_mongo(d) for d in doc]
    if isinstance(doc, dict):
        return {k: serialize_mongo(v) for k, v in doc.items()}
    if isinstance(doc, ObjectId):
        return str(doc)
    return doc

# ------------------ Plan Recommendation System ------------------
def train_recommendation_model():
    data = {
        'user_id': range(100),
        'current_plan': np.random.choice(['Starter', 'Pro', 'Ultra'], 100),
        'usage_gb': np.random.randint(10, 500, 100),
        'satisfaction': np.random.randint(1, 6, 100),
        'recommended_plan': np.random.choice(['Starter', 'Pro', 'Ultra'], 100)
    }
    
    df = pd.DataFrame(data)
    plan_mapping = {'Starter': 0, 'Pro': 1, 'Ultra': 2}
    df['current_plan_num'] = df['current_plan'].map(plan_mapping)
    df['recommended_plan_num'] = df['recommended_plan'].map(plan_mapping)
    
    X = df[['current_plan_num', 'usage_gb', 'satisfaction']]
    y = df['recommended_plan_num']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    with open('recommendation_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model

def get_plan_recommendation(user_data):
    if os.path.exists('recommendation_model.pkl'):
        with open('recommendation_model.pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        model = train_recommendation_model()
    
    current_plan = user_data.get('current_plan', 'Starter')
    usage_gb = user_data.get('usage_gb', 100)
    satisfaction = user_data.get('satisfaction', 3)
    
    plan_mapping = {'Starter': 0, 'Pro': 1, 'Ultra': 2}
    current_plan_num = plan_mapping.get(current_plan, 0)
    
    features = np.array([[current_plan_num, usage_gb, satisfaction]])
    prediction = model.predict(features)[0]
    
    reverse_mapping = {0: 'Starter', 1: 'Pro', 2: 'Ultra'}
    return reverse_mapping.get(prediction, 'Pro')

# ------------------ Churn Prediction Model ------------------
def train_churn_model():
    data = {
        'user_id': range(200),
        'tenure_months': np.random.randint(1, 36, 200),
        'usage_gb': np.random.randint(10, 500, 200),
        'support_tickets': np.random.randint(0, 10, 200),
        'payment_delays': np.random.randint(0, 5, 200),
        'plan_changes': np.random.randint(0, 3, 200),
        'churned': np.random.choice([0, 1], 200, p=[0.8, 0.2])
    }
    
    df = pd.DataFrame(data)
    X = df[['tenure_months', 'usage_gb', 'support_tickets', 'payment_delays', 'plan_changes']]
    y = df['churned']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    with open('churn_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return model

def predict_churn(user_data):
    if os.path.exists('churn_model.pkl'):
        with open('churn_model.pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        model = train_churn_model()
    
    tenure = user_data.get('tenure_months', 12)
    usage = user_data.get('usage_gb', 100)
    tickets = user_data.get('support_tickets', 0)
    delays = user_data.get('payment_delays', 0)
    changes = user_data.get('plan_changes', 0)
    
    features = np.array([[tenure, usage, tickets, delays, changes]])
    churn_prob = model.predict_proba(features)[0][1]
    
    if churn_prob < 0.3:
        risk_level = "Low"
        css_class = "low-churn"
    elif churn_prob < 0.7:
        risk_level = "Medium"
        css_class = "medium-churn"
    else:
        risk_level = "High"
        css_class = "high-churn"
    
    return {'probability': churn_prob, 'risk_level': risk_level, 'css_class': css_class}

# ------------------ Usage Forecasting Model ------------------
def forecast_usage(user_id, periods=6, usage_data=None):
    try:
        if usage_data is None:
            dates = pd.date_range(start="2023-01-01", periods=24, freq="M")
            usage = np.random.randint(50, 300, 24)
            usage_data = pd.DataFrame({"date": dates, "usage": usage})

        usage_data["date"] = pd.to_datetime(usage_data["date"])
        usage_data.set_index("date", inplace=True)
        usage_data = usage_data.asfreq("M")
        
        model = ExponentialSmoothing(
            usage_data["usage"], trend="add", seasonal="add", seasonal_periods=12
        )
        fitted_model = model.fit()
        forecast = fitted_model.forecast(periods)
        
        forecast_dates = pd.date_range(
            start=usage_data.index[-1] + pd.DateOffset(months=1),
            periods=periods,
            freq="M",
        )
        forecast_df = pd.DataFrame({"date": forecast_dates, "forecast": forecast})
        return forecast_df.reset_index(drop=True)

    except Exception as e:
        print(f"❌ Forecasting failed: {str(e)}")
        return pd.DataFrame(columns=["date", "forecast"])

# ------------------ Caching & MongoDB ------------------
@st.cache_resource(ttl=3600)
def get_mongo_connection():
    return MongoClient(MONGO_CONN_STRING)

@st.cache_data(ttl=600)
def get_user_data(username):
    user = users_collection.find_one({"username": username})
    subscriptions = list(subscriptions_collection.find({"username": username}))
    return serialize_mongo(user), serialize_mongo(subscriptions)

@st.cache_data(ttl=600)
def get_all_plans():
    return serialize_mongo(list(plans_collection.find()))

def load_data_with_progress():
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Connecting to database...")
    client = get_mongo_connection()
    progress_bar.progress(25)
    
    status_text.text("Loading user data...")
    user, subscriptions = get_user_data(st.session_state.get('username', ''))
    progress_bar.progress(50)
    
    status_text.text("Loading available plans...")
    plans = get_all_plans()
    progress_bar.progress(75)
    
    status_text.text("Loading ML models...")
    progress_bar.progress(100)
    
    status_text.text("Loading complete!")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()
    
    return user, subscriptions, plans


# ---------------- MongoDB Setup ----------------
# Connect to MongoDB with provided URL
MONGO_CONN_STRING ="mongodb+srv://pavan_db_user:Pavan%40db@broadbandcluster.f28qfox.mongodb.net/broadband6"
client = MongoClient(MONGO_CONN_STRING)
db = client['broadband6']  # Updated database name
# Collections
users_collection = db['users']
plans_collection = db['plans']
subscriptions_collection = db['subscriptions']

# ---------------- Helper Functions ----------------
def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def bootstrap_data():
    # Create admin user if not exists
    if users_collection.count_documents({}) == 0:
        admin_password = hash_password("admin123")
        users_collection.insert_one({
            "username": "admin",
            "full_name": "Admin User",
            "email": "admin@broadband.com",
            "phone": "555-0000",
            "password": admin_password,
            "role": "admin",
            "created_at": datetime.utcnow()
        })
    
    # Create default plans if not exists
    if plans_collection.count_documents({}) == 0:
        plans_collection.insert_many([
            {
                "name": "Starter Plan",
                "price": "$10",
                "speed": "50 Mbps",
                "desc": "Perfect for light browsing & emails.",
                "category": "Basic",
                "data": "100 GB/month"
            },
            {
                "name": "Pro Plan",
                "price": "$25",
                "speed": "200 Mbps",
                "desc": "Great for streaming and gaming.",
                "category": "Standard",
                "data": "Unlimited"
            },
            {
                "name": "Ultra Plan",
                "price": "$50",
                "speed": "1 Gbps",
                "desc": "Best for businesses & heavy users.",
                "category": "Premium",
                "data": "Unlimited"
            }
        ])

# Bootstrap data on startup
bootstrap_data()
# ----------------- Global Glassmorphism Styles -----------------
st.markdown("""
<style>
    /* ========== GLOBAL LAYOUT ========== */
    body[data-theme="light"] {
        background: linear-gradient(135deg, #f0f0f0, #dcdcdc);
        color: #222;
    }

    body[data-theme="dark"] {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #f0f0f0;
    }

    .main-header {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 2rem;
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin-bottom: 2rem;
        text-align: center;
    }

    /* Grid for responsive layouts */
    .dashboard-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
    }

    /* ========== GLASS CARDS ========== */
    .glass-card, .enhanced-card, .subscription-card, .plan-card,
    .metric-card, .card, .user-card, .faq-item, .history-card, .form-container, .chart-container {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 15px;
        backdrop-filter: blur(15px) saturate(180%);
        -webkit-backdrop-filter: blur(15px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 8px 32px rgba(0,0,0,0.25);
    }

    .glass-card:hover, .enhanced-card:hover, .subscription-card:hover,
    .plan-card:hover, .metric-card:hover, .card:hover, .user-card:hover {
        transform: translateY(-5px);
         box-shadow: 0 0 25px rgba(255, 105, 180, 0.8),
                0 0 45px rgba(255, 182, 193, 0.6)
            ;
    }

    /* Titles */
    .card-title, .plan-title, .chart-title, .form-title {
        font-size: 1.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #ff6ec4, #7873f5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }

    /* Metric Value */
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #36d1dc, #5b86e5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
            
            .metric-card {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 15px;
    backdrop-filter: blur(15px) saturate(180%);
    -webkit-backdrop-filter: blur(15px) saturate(180%);
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    text-align: center;
    transition: all 0.3s ease-in-out;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

.metric-card:hover {
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 0 25px rgba(102, 126, 234, 0.8), 
                0 0 45px rgba(118, 75, 162, 0.6);
    border-color: rgba(255,255,255,0.5);
}


    /* Badges */
    .badge {
        background: rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-size: 0.8rem;
        font-weight: bold;
        color: white;
    }

    /* Churn indicator */
    .low-churn { background: rgba(40,167,69,0.3); color: #28a745; }
    .medium-churn { background: rgba(255,193,7,0.3); color: #ffc107; }
    .high-churn { background: rgba(220,53,69,0.3); color: #dc3545; }



/* ---------- Login Page Background ---------- */
.login-page {
   background: url("https://th.bing.com/th/id/OIP.iR-L3h8p33r7B8Dn80Lh3AHaHa?w=174&h=180&c=7&r=0&o=5&dpr=1.3&pid=1.7") no-repeat center center fixed;
    background-size: cover;
    height: 60vh;
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 2rem;
}
/* ---------- Glassmorphism Login Form ---------- */
.login-form {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    padding: 2rem;
    width: 100%;
    max-width: 400px;
    backdrop-filter: blur(15px) saturate(180%);
    -webkit-backdrop-filter: blur(15px) saturate(180%);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    transition: all 0.4s ease-in-out;
}

/* ---------- Hover Glow ---------- */
.login-form:hover {
    transform: scale(1.02);
    box-shadow: 0 0 25px rgba(102, 126, 234, 0.8),
                0 0 45px rgba(118, 75, 162, 0.6);
}



            /* Register Page Background */
.register-page {
    background: url("https://static.vecteezy.com/system/resources/previews/003/689/231/original/online-registration-or-sign-up-login-for-account-on-smartphone-app-user-interface-with-secure-password-mobile-application-for-ui-web-banner-access-cartoon-people-illustration-vector.jpg") no-repeat center center fixed;
    background-size: cover;
    height: 60vh;
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 2rem;
}

/* Glass effect form container */
.register-form {
    background: rgba(255, 255, 255, 0.15);
    border-radius: 20px;
    padding: 2rem;
    width: 100%;
    max-width: 450px;
    backdrop-filter: blur(15px) saturate(180%);
    -webkit-backdrop-filter: blur(15px) saturate(180%);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}




    /* ========== BUTTONS ========== */
    .stButton button, .nav-button, .sidebar-button, .subscribe-button, .action-button {
        background: rgba(255, 255, 255, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 12px;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }

            
    /* ---------- Top Navigation ---------- */
.top-nav {
    position: fixed;
    top: 20px;
    left: 20px;
    display: flex;
    gap: 15px;
    z-index: 9998;
}

.nav-link {
    background: rgba(255, 255, 255, 0.2);
    padding: 8px 15px;
    border-radius: 12px;
    text-decoration: none;
    font-weight: bold;
    color: white;
    font-size: 0.95rem;
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    box-shadow: 0 4px 10px rgba(0,0,0,0.25);
}

.nav-link:hover {
    background: rgba(255, 255, 255, 0.4);
    color: #ffde59;
    transform: translateY(-3px) scale(1.05);
    box-shadow: 0 8px 25px rgba(255, 222, 89, 0.6);
}

/* ---------- Home Container ---------- */

.home-container:hover {
    transform: translateY(-3px) scale(1);
    box-shadow: 0 12px 45px rgba(102, 126, 234, 0.6);
}

/* ---------- Home Title & Subtitle ---------- */
.home-title {
    font-size: 2.8rem;
    font-weight: bold;
    background: linear-gradient(90deg, #ff6ec4, #7873f5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 1rem;
}

.home-subtitle {
    font-size: 1.2rem;
    color: rgba(255,255,255,0.85);
}

            


    .stButton button:hover, .nav-button:hover, .sidebar-button:hover,
    .subscribe-button:hover, .action-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 7px 25px rgba(0,0,0,0.4);
    }

    .delete-button {
        background: rgba(255, 0, 0, 0.4) !important;
        color: white !important;
    }

    /* ========== FIXED THEME TOGGLE BUTTON ========== */
    .theme-toggle {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.25);
        border: 1px solid rgba(255, 255, 255, 0.3);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 50%;
        width: 45px;
        height: 45px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
        cursor: pointer;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }

    /* ========== RESPONSIVE DESIGN ========== */
    @media (max-width: 768px) {
        .main-header { padding: 1.5rem; font-size: 1.2rem; }
        .dashboard-grid { grid-template-columns: 1fr; }
        .card-title, .plan-title { font-size: 1.3rem; }
        .metric-value { font-size: 2rem; }
        .theme-toggle { width: 40px; height: 40px; font-size: 1rem; }
    }

    @media (max-width: 480px) {
        .main-header { padding: 1rem; font-size: 1rem; }
        .card, .glass-card { padding: 1rem; }
        .metric-value { font-size: 1.5rem; }
        .stButton button { width: 100%; }
    }
</style>
""", unsafe_allow_html=True)


# ----------------- Notification Toast -----------------
def show_notification(message, icon="✅"):
    st.markdown(f"""
    <div class="toast-container">
        <div class="custom-toast">
            {icon} {message}
        </div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(2)
    st.rerun()

# ----------------- Navigation Bar -----------------
def nav_bar(page_suffix=""):
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🏠 Home", key=f"nav_home_{page_suffix}"):
            st.session_state['page'] = 'home'
    with col2:
        if st.button("🔑 Login", key=f"nav_login_{page_suffix}"):
            st.session_state['page'] = 'login'
    with col3:
        if st.button("📝 Register", key=f"nav_register_{page_suffix}"):
            st.session_state['page'] = 'register'

# ----------------- Landing Page -----------------


def landing_page():
    nav_bar("landing")
    st.markdown('''
    <div class="home-container"  style="position:relative; text-align:center; margin-bottom:50px;">
        <img src="https://img.freepik.com/premium-photo/globe-computers-abstract-background-3d-illustration-elements-this-image-furnished-by-nasa-communication-technology-internet-business-emphasizing-global-world-network-ai-generated_585735-4639.jpg" 
             style="width:100%; height:50vh; object-fit:cover; border-radius:15px; filter: brightness(65%);">
        <div style="position:absolute; top:50%; left:50%; transform:translate(-50%, -50%); color:white; max-width:700px;">
            <h1 style="font-size:2rem; line-height:1.2; text-shadow: 2px 2px 8px black; margin-bottom:20px;">Welcome to Broadband Portal</h1>
            <p style="font-size:1.3rem; text-shadow: 2px 2px 6px black; margin-bottom:25px;">Fast, reliable, and affordable internet for everyone.</p>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Features
    features = [
        ("⚡ High Speed", "Enjoy blazing fast internet with no interruptions."),
        ("📶 Unlimited Data", "Stay connected with unlimited data packs."),
        ("💰 Affordable Plans", "Choose from a wide range of pocket-friendly plans."),
        ("🔒 Secure Connection", "Your data and privacy are our top priority."),
        ("💡 Smart Support", "24/7 expert assistance whenever you need it."),
        ("🌐 Wide Coverage", "Internet available in most urban and rural areas.")
    ]
    
    st.markdown("## Our Features", unsafe_allow_html=True)
    for i in range(0, len(features), 3):
        cols = st.columns(3)
        for j, feature in enumerate(features[i:i+3]):
            with cols[j]:
                st.markdown(f'''
                <div class="metric-card" >
                    <h3 style="margin-bottom:15px;">{feature[0]}</h3>
                    <p style="font-size:1rem;">{feature[1]}</p>
                </div>
                ''', unsafe_allow_html=True)
    
    # Testimonials
 
    
    # Footer
    st.markdown('''
    <div style="background:linear-gradient(135deg, #2c3e50 0%, #34495e 100%); color:white; text-align:center; padding:30px; margin-top:60px; border-radius:10px;">
        <p>© 2025 Broadband Portal. All rights reserved.</p>
    </div>
    ''', unsafe_allow_html=True)

# ----------------- Login Page -----------------
def login_page():
    nav_bar("login")
    st.markdown("""
    <div class="login-page">
        <div class="login-form">
            <h2 style="text-align:center;color:black;">Scroll Down to Login</h2>
    """, unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")
         
        if st.form_submit_button("Login"):
            if not username or not password:
                st.error("Please fill all fields!")
            else:
                # Check user in database
                user = users_collection.find_one({"username": username})
                  
                if user and check_password(password, user["password"]):
                    role = user.get("role", "user")  # Default role = user
                    st.success(f"Welcome back, {username}! Logged in as {role.capitalize()}.")
                        
                    st.session_state['role'] = role
                    st.session_state['page'] = role
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("Invalid username or password!")
        
    if st.button("Don't have an account? Register"):
        st.session_state['page'] = 'register'
        st.rerun()
    
    st.markdown("""
        </div>
    </div>
    """, unsafe_allow_html=True)

# ----------------- Register Page -----------------
def register_page():
    nav_bar("register")
    st.markdown("""
    <div class="register-page">
        <div class="register-form">
            <h2 style="text-align:center; color:black; margin-bottom:20px;">Scroll Down to Register</h2>
    """, unsafe_allow_html=True)
    # st.image("https://static.vecteezy.com/system/resources/previews/003/689/231/original/online-registration-or-sign-up-login-for-account-on-smartphone-app-user-interface-with-secure-password-mobile-application-for-ui-web-banner-access-cartoon-people-illustration-vector.jpg", width=500)
    
    with st.form("register_form"):
        full_name = st.text_input("Full Name", placeholder="Enter your full name")
        username = st.text_input("Username", placeholder="Choose a unique username")
        email = st.text_input("Email", placeholder="your.email@example.com")
        phone = st.text_input("Phone Number", placeholder="+1 (555) 123-4567")
        password = st.text_input("Password", type="password", placeholder="Create a strong password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
        
        if st.form_submit_button("Register"):
            if not full_name or not username or not email or not phone or not password or not confirm_password:
                st.error("Please fill all fields!")
            elif password != confirm_password:
                st.error("Passwords do not match!")
            else:
                # Check if username already exists
                if users_collection.find_one({"username": username}):
                    st.error("Username already exists!")
                else:
                    # Hash password and insert user
                    hashed_password = hash_password(password)
                    users_collection.insert_one({
                        "username": username,
                        "full_name": full_name,
                        "email": email,
                        "phone": phone,
                        "password": hashed_password,
                        "role": "user",
                        "created_at": datetime.utcnow()
                    })
                    st.success(f"🎉 Account created for {full_name}!")
                    st.session_state['page'] = 'login'
                    st.rerun()
    
    if st.button("Already have an account? Login"):
        st.session_state['page'] = 'login'
        st.rerun()

# ----------------- User Dashboard -----------------
def enhanced_user_dashboard():
    # Get current user
    username = st.session_state.get('username')
    user, _ = get_user_data(username)
    plans = get_all_plans()

    # Fetch real-time subscriptions directly from MongoDB
    user_email = user.get("email")
    subscriptions = []
    if user_email:
        subscriptions = list(subscriptions_collection.find({"email": user_email}).sort("start_date", -1))

    # Categorize subscriptions
    active_subs = [s for s in subscriptions if str(s.get('status', '')).lower() == 'active']
    queued_subs = [s for s in subscriptions if str(s.get('status', '')).lower() == 'queued']
    expired_subs = [s for s in subscriptions if str(s.get('status', '')).lower() == 'expired']

    # Dashboard Header
    st.markdown(f"""
    <div class="main-header">
        <h1>👤 Hello, {username} !!</h1>
        <p>Welcome back, {username}! Here's your personalized internet overview.</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## User Navigations")

        nav_options = [
            ("🏠 Dashboard", "dashboard"),
            ("📄 My Subscriptions", "subscriptions"),
            ("📝 Available Plans", "plans"),
            ("👤 My Profile", "profile"),
            ("💬 Support", "support"),
            ("🚪 Logout", "logout")
        ]

        for icon, key in nav_options:
            if st.button(icon, key=f"user_{key}"):
                if key == "logout":
                    st.session_state['page'] = 'home'
                    st.session_state['role'] = None
                    st.session_state['username'] = None
                else:
                    st.session_state['user_nav'] = key
                st.rerun()

    # Initialize navigation state
    if 'user_nav' not in st.session_state:
        st.session_state['user_nav'] = 'dashboard'

    # ---------------- Dashboard View ----------------
    if st.session_state['user_nav'] == 'dashboard':
        col1, col2, col3 = st.columns(3)
        total_subs = len(subscriptions)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(active_subs)}</div>
                <div class="metric-label">Active Subscriptions</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_subs}</div>
                <div class="metric-label">Total Subscriptions</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            days_until_renewal = 0
            if active_subs:
                end_date = active_subs[0].get('end_date')
                if end_date:
                    days_until_renewal = (end_date - datetime.utcnow()).days

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{days_until_renewal}</div>
                <div class="metric-label">Days Until Renewal</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<h2 class="card-title">Current Plan</h2>', unsafe_allow_html=True)

        if active_subs:
            current_plan = active_subs[0]
            plan_name = current_plan.get('plan', 'Unknown Plan')
            plan_details = next((p for p in plans if p.get('name') == plan_name), None)

            if plan_details:
                st.markdown(f"""
                <div class="enhanced-card">
                    <div class="card-header">
                        <h3>{plan_details.get('name', 'Unknown Plan')}</h3>
                        <span class="badge">Active</span>
                    </div>
                    <p><strong>Price:</strong> {plan_details.get('price', '$0')}</p>
                    <p><strong>Speed:</strong> {plan_details.get('speed', 'Unknown')}</p>
                    <p><strong>Data:</strong> {plan_details.get('data', 'Unknown')}</p>
                    <p><strong>Start Date:</strong> {current_plan.get('start_date').strftime('%Y-%m-%d') if current_plan.get('start_date') else 'N/A'}</p>
                    <p><strong>End Date:</strong> {current_plan.get('end_date').strftime('%Y-%m-%d') if current_plan.get('end_date') else 'N/A'}</p>
                </div>
                """, unsafe_allow_html=True)

                # Put interactive buttons below the card
                colA, colB = st.columns(2)
                with colA:
                    if st.button("🔼 Upgrade Plan"):
                        st.session_state['user_nav'] = 'plans'  # Navigate to Available Plans
                        # st.rerun();

                with colB:
                    if st.button("🔄 Renew Now"):
                        # Extend subscription in DB (example)
                        new_end = datetime.utcnow() + timedelta(days=30)
                        subscriptions_collection.update_one(
                            {"_id": current_plan["_id"]},
                            {"$set": {"end_date": new_end}}
                        )
                        st.success("Your plan has been renewed for 30 days!")
                        # st.rerun();
        else:
            st.info("You don't have any active subscriptions.")


        # Usage Analytics
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="chart-title">Usage Analytics</h2>', unsafe_allow_html=True)

        dates = [datetime.now() - timedelta(days=x) for x in range(30, 0, -1)]
        usage = [user.get('usage_gb', 100) + (i % 20) - 10 for i in range(30)]
        df = pd.DataFrame({'Date': dates, 'Usage (GB)': usage})

        fig = px.line(df, x='Date', y='Usage (GB)', title='Daily Data Usage',
                     line_shape='linear', color_discrete_sequence=['#667eea'])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- My Subscriptions View ----------------
    elif st.session_state['user_nav'] == 'subscriptions':
        st.markdown('<h1 class="card-title">My Subscriptions</h1>', unsafe_allow_html=True)

        # ---------------- Active Subscriptions ----------------
        st.markdown('<h3>Active Subscriptions</h3>', unsafe_allow_html=True)
        if active_subs:
            for sub in active_subs:
                plan_name = sub.get('plan', 'Unknown Plan')
                plan_details = next((p for p in plans if p.get('name') == plan_name), None)

                if plan_details:
                    st.markdown(f"""
                    <div class="subscription-card active">
                        <div class="card-header">
                            <h3>{plan_details.get('name')}</h3>
                            <span class="badge">Active</span>
                        </div>
                        <p><strong>Price:</strong> {plan_details.get('price')}</p>
                        <p><strong>Speed:</strong> {plan_details.get('speed')}</p>
                        <p><strong>Data:</strong> {plan_details.get('data')}</p>
                        <p><strong>Start Date:</strong> {sub.get('start_date').strftime('%Y-%m-%d') if sub.get('start_date') else 'N/A'}</p>
                        <p><strong>End Date:</strong> {sub.get('end_date').strftime('%Y-%m-%d') if sub.get('end_date') else 'N/A'}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # --------- Action Buttons for Active Plan ---------
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("Upgrade", key=f"upgrade_{sub['_id']}"):
                            st.session_state['user_nav'] = "plans"  # redirect to available plans
                            st.rerun()
                    with col2:
                        if st.button("Renewal", key=f"renew_{sub['_id']}"):
                            subscriptions_collection.insert_one({
                                "username": sub.get("username"),
                                "email": sub.get("email"),
                                "plan": sub.get("plan"),
                                "price": sub.get("price"),
                                "speed": sub.get("speed"),
                                "data": sub.get("data"),
                                "status": "Queued",
                                "start_date": None,
                                "end_date": None
                            })
                            st.success(f"🔄 {sub.get('plan')} has been added again into your upcoming plans.")
                            st.rerun();
                    with col3:
                        if st.button("Cancel", key=f"cancel_{sub['_id']}"):
                            subscriptions_collection.update_one(
                                {"_id": sub["_id"]},
                                {"$set": {
                                    "status": "Expired",
                                    "end_date": datetime.utcnow()
                                }}
                            )
                            st.warning("⚠️ Canceling a plan does not make you eligible for refunds. The plan has been moved to previous subscriptions.")
                            st.rerun();
        else:
            st.info("You don't have any active subscriptions.")

        # ---------------- Queued Subscriptions ----------------
        queued_subs = [s for s in subscriptions if str(s.get('status', '')).lower() == 'queued']
        if queued_subs:
            st.markdown('<h3>Queued Subscriptions (Upcoming)</h3>', unsafe_allow_html=True)
            queued_subs_sorted = sorted(queued_subs, key=lambda x: x.get('_id'))  # FIFO
            for sub in queued_subs_sorted:
                st.markdown(f"""
                <div class="subscription-card queued">
                    <div class="card-header">
                        <h3>{sub.get('plan', 'Unknown Plan')}</h3>
                        <span class="badge queued">Queued</span>
                    </div>
                    <p><strong>Price:</strong> {sub.get('price', '$0')}</p>
                    <p><strong>Speed:</strong> {sub.get('speed', 'Unknown')}</p>
                    <p><strong>Data:</strong> {sub.get('data', 'Unknown')}</p>
                </div>
                """, unsafe_allow_html=True)

                # --------- Action Button for Queued Plan ---------
                if st.button("Cancel", key=f"cancel_queue_{sub['_id']}"):
                    subscriptions_collection.update_one(
                        {"_id": sub["_id"]},
                        {"$set": {
                            "status": "Expired",
                            "end_date": datetime.utcnow()
                        }}
                    )
                    st.warning("⚠️ Queued plan canceled. Canceling a plan does not make you eligible for refunds.")
                    st.rerun();
        else:
            st.info("You don't have any queued subscriptions.")

        # ---------------- Expired Subscriptions ----------------
        st.markdown('<h3>Previous Subscriptions</h3>', unsafe_allow_html=True)
        expired_subs_list = [s for s in subscriptions if str(s.get('status', '')).lower() == 'expired']
        if expired_subs_list:
            for sub in expired_subs_list:
                plan_name = sub.get('plan', 'Unknown Plan')
                plan_details = next((p for p in plans if p.get('name') == plan_name), None)

                if plan_details:
                    st.markdown(f"""
                    <div class="subscription-card expired">
                        <div class="card-header">
                            <h3>{plan_details.get('name')}</h3>
                            <span class="badge expired">Expired</span>
                        </div>
                        <p><strong>Price:</strong> {plan_details.get('price')}</p>
                        <p><strong>Speed:</strong> {plan_details.get('speed')}</p>
                        <p><strong>Data:</strong> {plan_details.get('data')}</p>
                        <p><strong>Start Date:</strong> {sub.get('start_date').strftime('%Y-%m-%d') if sub.get('start_date') else 'N/A'}</p>
                        <p><strong>End Date:</strong> {sub.get('end_date').strftime('%Y-%m-%d') if sub.get('end_date') else 'N/A'}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("You don't have any previous subscriptions.")



    # ---------------- Available Plans View ----------------
    elif st.session_state['user_nav'] == 'plans':
        st.markdown('<h1 class="card-title">Available Plans</h1>', unsafe_allow_html=True)

        categories = list(set(plan.get('category', 'Standard') for plan in plans))
        selected_category = st.selectbox("Filter by Category", ["All"] + categories)

        filtered_plans = plans if selected_category == "All" else [p for p in plans if p.get('category') == selected_category]

        for i in range(0, len(filtered_plans), 3):
            cols = st.columns(3)
            for j, plan in enumerate(filtered_plans[i:i+3]):
                with cols[j]:
                    st.markdown(f"""
                    <div class="plan-card">
                        <h3 class="plan-title">{plan.get('name', 'Unknown Plan')}</h3>
                        <p>{plan.get('desc', '')}</p>
                        <div class="plan-price">{plan.get('price', '$0')}<span>/month</span></div>
                        <ul class="plan-features">
                            <li>Speed: {plan.get('speed', 'Unknown')}</li>
                            <li>Data: {plan.get('data', 'Unknown')}</li>
                            <li>Category: {plan.get('category', 'Standard')}</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                    # ---------------- Subscribe Now Button ----------------
                    if st.button(f"Subscribe Now - {plan.get('name')}", key=f"sub_{plan['_id']}"):
                        if user_email:
                            # Check for existing active subscription
                            active_plan = subscriptions_collection.find_one({"email": user_email, "status": "Active"})

                            if active_plan:
                                # Add new plan as queued
                                subscriptions_collection.insert_one({
                                    "username": username,
                                    "email": user_email,
                                    "plan": plan.get("name"),
                                    "price": plan.get("price"),
                                    "speed": plan.get("speed"),
                                    "data": plan.get("data"),
                                    "status": "Queued",
                                    "start_date": None,
                                    "end_date": None
                                })
                                st.success(f"🕒 {plan.get('name')} has been added to your queue.")
                            else:
                                # No active plan, add as active
                                subscriptions_collection.insert_one({
                                    "username": username,
                                    "email": user_email,
                                    "plan": plan.get("name"),
                                    "price": plan.get("price"),
                                    "speed": plan.get("speed"),
                                    "data": plan.get("data"),
                                    "status": "Active",
                                    "start_date": datetime.utcnow(),
                                    "end_date": datetime.utcnow() + timedelta(days=30)
                                })
                                st.success(f"🎉 You have successfully subscribed to {plan.get('name')}!")
                            st.rerun();
                        else:
                            st.error("⚠️ Please log in to subscribe.")

    # ---------------- My Profile View ----------------
    elif st.session_state['user_nav'] == 'profile':
        st.markdown('<h1 class="card-title">My Profile</h1>', unsafe_allow_html=True)
        
        st.markdown('<div class="enhanced-card">', unsafe_allow_html=True)
        st.markdown('<h3>Profile Information</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <p><strong>Username:</strong> {user.get('username', 'N/A')}</p>
            <p><strong>Full Name:</strong> {user.get('full_name', 'N/A')}</p>
            <p><strong>Email:</strong> {user.get('email', 'N/A')}</p>
            """, unsafe_allow_html=True)
        
        with col2:
            # Convert created_at safely
            created_at = user.get('created_at')
            if created_at:
                try:
                    created_at_str = created_at.strftime('%Y-%m-%d')
                except AttributeError:
                    created_at_str = str(created_at)  # fallback if it's not datetime
            else:
                created_at_str = 'N/A'

            st.markdown(f"""
            <p><strong>Phone:</strong> {user.get('phone', 'N/A')}</p>
            <p><strong>Member Since:</strong> {created_at_str}</p>
            <p><strong>Role:</strong> {user.get('role', 'User')}</p>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ---------- Subscription Statistics ----------
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="chart-title">Subscription Statistics</h2>', unsafe_allow_html=True)

        # Ensure these are just integers, not ObjectId or custom objects
        active_count = len(active_subs) if active_subs else 0
        expired_count = len(expired_subs) if expired_subs else 0

        status_counts = {"Active": active_count, "Expired": expired_count}

        fig_pie = px.pie(
            values=list(status_counts.values()),
            names=list(status_counts.keys()),
            title="Subscription Status Distribution",
            color_discrete_map={"Active": "#4CAF50", "Expired": "#F44336"}
        )

        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


    # ---------------- Support View ----------------
    elif st.session_state['user_nav'] == 'support':
        st.markdown('<h1 class="card-title">Support</h1>', unsafe_allow_html=True)

        # --- Contact Form ---
        st.markdown('<div class="enhanced-card">', unsafe_allow_html=True)
        st.markdown('<h3>📩 Contact Us</h3>', unsafe_allow_html=True)

        with st.form("contact_form"):
            name = st.text_input("Your Name")
            email = st.text_input("Your Email")
            subject = st.text_input("Subject")
            message = st.text_area("Message", height=150)
            priority = st.selectbox("Priority", ["Low", "Medium", "High"])
            submitted = st.form_submit_button("Send")

            if submitted:
                if name and email and subject and message:
                    st.success("✅ Your message has been sent. We’ll get back to you shortly!")
                    # 👉 Here you can insert into MongoDB or send email
                    # support_collection.insert_one({
                    #     "name": name,
                    #     "email": email,
                    #     "subject": subject,
                    #     "message": message,
                    #     "priority": priority,
                    #     "created_at": datetime.utcnow()
                    # })
                else:
                    st.error("⚠️ Please fill out all fields before submitting.")

        st.markdown('</div>', unsafe_allow_html=True)

        # --- FAQ Section ---
        st.markdown('<div class="enhanced-card">', unsafe_allow_html=True)
        st.markdown('<h3>❓ Frequently Asked Questions</h3>', unsafe_allow_html=True)

        faqs = [
            {
                "q": "How do I reset my password?",
                "a": "Click on **Forgot Password** on the login page and follow the instructions."
            },
            {
                "q": "How do I upgrade my plan?",
                "a": "Go to **Available Plans** section and click **Upgrade** on your desired plan."
            },
            {
                "q": "Can I cancel my subscription?",
                "a": "Yes, go to **My Subscriptions** and click **Cancel**. Note: No refunds for mid-cycle cancellations."
            },
            {
                "q": "What payment methods are accepted?",
                "a": "We accept Credit/Debit Cards, UPI, and Net Banking."
            },
            {
                "q": "Is there a contract?",
                "a": "Most plans are **month-to-month** with no long-term contracts."
            }
        ]

        for item in faqs:
            with st.expander(item["q"]):
                st.markdown(item["a"])

        st.markdown('</div>', unsafe_allow_html=True)


# ----------------- Admin Dashboard -----------------
def enhanced_admin_dashboard():
    st.markdown("""
    <div class="main-header">
        <h1>🛠 Admin Dashboard</h1>
        <p>Manage users, plans, subscriptions, and AI insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## 📌 Admin Navigation")
        
        nav_options = [
            ("📊 Dashboard", "dashboard"),
            ("👥 User Management", "users"),
            ("📝 Plan Management", "plans"),
            ("📈 Analytics", "analytics"),
            ("🤖 Insights", "ai"),
            ("📜 User History", "history"),
            ("🚪 Logout", "logout")
        ]
        
        for icon, key in nav_options:
            if st.button(icon, key=f"admin_{key}"):
                if key == "logout":
                    st.session_state['page'] = 'home'
                    st.session_state['username'] = None
                    st.session_state['role'] = None
                else:
                    st.session_state['admin_nav'] = key
                st.rerun()
    
    # Initialize navigation state
    if 'admin_nav' not in st.session_state:
        st.session_state['admin_nav'] = 'dashboard'
    
    # Dashboard View
    if st.session_state['admin_nav'] == 'dashboard':
        # Key Metrics
        total_users = users_collection.count_documents({})
        total_plans = plans_collection.count_documents({})
        active_subs = subscriptions_collection.count_documents({"status": "Active"})
        total_revenue = sum(int(plan.get('price', '$0').replace('$', '').replace(',', '')) 
                          for plan in plans_collection.find() 
                          for _ in range(subscriptions_collection.count_documents({"plan": plan.get('name'), "status": "Active"})))
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_users}</div>
                <div class="metric-label">Total Users</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_plans}</div>
                <div class="metric-label">Available Plans</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{active_subs}</div>
                <div class="metric-label">Active Subscriptions</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${total_revenue:,}</div>
                <div class="metric-label">Monthly Revenue</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Recent Activity
        st.markdown('<div class="enhanced-card">', unsafe_allow_html=True)
        st.markdown('<h3>Recent Activity</h3>', unsafe_allow_html=True)
        
        recent_subs = list(subscriptions_collection.find().sort("start_date", -1).limit(5))
        for sub in recent_subs:
            user = users_collection.find_one({"username": sub.get('username')})
            user_name = user.get('full_name', 'Unknown') if user else 'Unknown'
            st.markdown(f"""
            <div class="history-item">
                <strong>{user_name}</strong> subscribed to <strong>{sub.get('plan')}</strong> 
                on {sub.get('start_date').strftime('%Y-%m-%d') if sub.get('start_date') else 'N/A'}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # User Management View
    elif st.session_state['admin_nav'] == 'users':
        st.markdown('<h1 class="card-title">User Management</h1>', unsafe_allow_html=True)
        
        # User List
        users = list(users_collection.find({"role": "user"}))
        
        for user in users:
            with st.expander(f"{user.get('full_name', 'Unknown')} ({user.get('username')})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <p><strong>Email:</strong> {user.get('email', 'N/A')}</p>
                    <p><strong>Phone:</strong> {user.get('phone', 'N/A')}</p>
                    <p><strong>Member Since:</strong> {user.get('created_at').strftime('%Y-%m-%d') if user.get('created_at') else 'N/A'}</p>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # User statistics
                    user_subs = list(subscriptions_collection.find({"username": user.get('username')}))
                    active_count = len([s for s in user_subs if s.get('status') == 'Active'])
                    st.markdown(f"""
                    <p><strong>Total Subscriptions:</strong> {len(user_subs)}</p>
                    <p><strong>Active Subscriptions:</strong> {active_count}</p>
                    <p><strong>Usage (GB):</strong> {user.get('usage_gb', 'N/A')}</p>
                    """, unsafe_allow_html=True)
                
                # Action buttons
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Edit User", key=f"edit_{user.get('username')}"):
                        st.session_state['editing_user'] = user.get('username')
                        st.rerun()
                
                with col2:
                    if st.button("View History", key=f"history_{user.get('username')}"):
                        st.session_state['viewing_user'] = user.get('username')
                        st.session_state['admin_nav'] = 'history'
                        st.rerun()
                
                with col3:
                    if st.button("Delete User", key=f"delete_{user.get('username')}"):
                        # Delete user and their subscriptions
                        users_collection.delete_one({"username": user.get('username')})
                        subscriptions_collection.delete_many({"username": user.get('username')})
                        st.success(f"User {user.get('username')} deleted successfully!")
                        st.rerun()
    
    # Plan Management View
    elif st.session_state['admin_nav'] == 'plans':
        st.markdown('<h1 class="card-title">Plan Management</h1>', unsafe_allow_html=True)
        
        # Add New Plan Form
        with st.expander("Add New Plan"):
            with st.form("add_plan_form"):
                plan_name = st.text_input("Plan Name")
                description = st.text_area("Description")
                category = st.selectbox("Category", ["Basic", "Standard", "Premium", "Business"])
                price = st.text_input("Price ($)")
                speed = st.text_input("Speed")
                data = st.text_input("Data Allowance")
                
                if st.form_submit_button("Add Plan"):
                    if all([plan_name, description, price, speed, data]):
                        plans_collection.insert_one({
                            "name": plan_name,
                            "desc": description,
                            "category": category,
                            "price": price,
                            "speed": speed,
                            "data": data
                        })
                        st.success(f"Plan '{plan_name}' added successfully!")
                        st.rerun()
                    else:
                        st.error("Please fill all fields")
        
        # Existing Plans
        plans = list(plans_collection.find())
        
        for plan in plans:
            with st.expander(f"{plan.get('name')} - {plan.get('price')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <p><strong>Description:</strong> {plan.get('desc', 'N/A')}</p>
                    <p><strong>Category:</strong> {plan.get('category', 'N/A')}</p>
                    <p><strong>Speed:</strong> {plan.get('speed', 'N/A')}</p>
                    <p><strong>Data:</strong> {plan.get('data', 'N/A')}</p>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Plan statistics
                    sub_count = subscriptions_collection.count_documents({"plan": plan.get('name')})
                    active_count = subscriptions_collection.count_documents({"plan": plan.get('name'), "status": "Active"})
                    st.markdown(f"""
                    <p><strong>Total Subscriptions:</strong> {sub_count}</p>
                    <p><strong>Active Subscriptions:</strong> {active_count}</p>
                    """, unsafe_allow_html=True)
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Edit Plan", key=f"edit_plan_{plan.get('name')}"):
                        st.session_state['editing_plan'] = plan.get('name')
                        st.rerun()
                
                with col2:
                    if st.button("Delete Plan", key=f"delete_plan_{plan.get('name')}"):
                        plans_collection.delete_one({"name": plan.get('name')})
                        subscriptions_collection.update_many(
                            {"plan": plan.get('name')},
                            {"$set": {"status": "Cancelled"}}
                        )
                        st.success(f"Plan '{plan.get('name')}' deleted successfully!")
                        st.rerun()
    
    # Analytics View
    elif st.session_state['admin_nav'] == 'analytics':
        st.markdown('<h1 class="card-title">Analytics Dashboard</h1>', unsafe_allow_html=True)
        
        # Subscription Trends
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="chart-title">Subscription Trends</h2>', unsafe_allow_html=True)
        
        # Generate sample data for subscription trends
        dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
        new_subs = [10 + i*2 + np.random.randint(0, 5) for i in range(12)]
        cancelled_subs = [2 + i + np.random.randint(0, 3) for i in range(12)]
        
        trend_df = pd.DataFrame({
            'Month': dates,
            'New Subscriptions': new_subs,
            'Cancelled Subscriptions': cancelled_subs
        })
        
        fig_trend = px.line(trend_df, x='Month', y=['New Subscriptions', 'Cancelled Subscriptions'],
                           title='Monthly Subscription Trends',
                           color_discrete_map={
                               'New Subscriptions': '#4CAF50',
                               'Cancelled Subscriptions': '#F44336'
                           })
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Revenue Analysis
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="chart-title">Revenue Analysis</h2>', unsafe_allow_html=True)
        
        # Calculate revenue by plan
        revenue_data = []
        for plan in plans_collection.find():
            active_count = subscriptions_collection.count_documents({"plan": plan.get('name'), "status": "Active"})
            price = int(plan.get('price', '$0').replace('$', '').replace(',', ''))
            revenue_data.append({
                'Plan': plan.get('name'),
                'Revenue': active_count * price,
                'Subscriptions': active_count
            })
        
        revenue_df = pd.DataFrame(revenue_data)
        
        fig_revenue = px.bar(revenue_df, x='Plan', y='Revenue',
                            title='Revenue by Plan',
                            color='Subscriptions',
                            color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig_revenue, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # User Growth
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="chart-title">User Growth</h2>', unsafe_allow_html=True)
        
        # Generate sample user growth data
        user_dates = pd.date_range(start='2023-01-01', periods=12, freq='M')
        user_counts = [50 + i*15 + np.random.randint(0, 10) for i in range(12)]
        
        user_growth_df = pd.DataFrame({
            'Month': user_dates,
            'Users': user_counts
        })
        
        fig_growth = px.line(user_growth_df, x='Month', y='Users',
                            title='User Growth Over Time',
                            line_shape='linear',
                            color_discrete_sequence=['#667eea'])
        st.plotly_chart(fig_growth, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # AI Insights View
    elif st.session_state['admin_nav'] == 'ai':
        st.markdown('<h1 class="card-title">🤖 Business Insights</h1>', unsafe_allow_html=True)
        
        # Churn Analysis
        st.markdown('<div class="ml-section">', unsafe_allow_html=True)
        st.markdown('<h2>Customer Churn Analysis</h2>', unsafe_allow_html=True)
        
        # Get all users and predict churn
        users = list(users_collection.find({"role": "user"}))
        churn_risks = []
        
        for user in users:
            user_data = {
                'tenure_months': user.get('tenure_months', 6),
                'usage_gb': user.get('usage_gb', 100),
                'support_tickets': user.get('support_tickets', 0),
                'payment_delays': user.get('payment_delays', 0),
                'plan_changes': user.get('plan_changes', 0)
            }
            churn_result = predict_churn(user_data)
            churn_risks.append({
                'username': user.get('username'),
                'full_name': user.get('full_name'),
                'risk_level': churn_result['risk_level'],
                'probability': churn_result['probability']
            })
        
        # Create DataFrame
        churn_df = pd.DataFrame(churn_risks)
        
        # Count by risk level
        risk_counts = churn_df['risk_level'].value_counts().reset_index()
        risk_counts.columns = ['Risk Level', 'Count']
        
        # Create bar chart
        fig_churn = px.bar(risk_counts, x='Risk Level', y='Count', 
                          title='Customer Churn Risk Distribution',
                          color='Risk Level',
                          color_discrete_map={
                              'Low': '#28a745',
                              'Medium': '#ffc107',
                              'High': '#dc3545'
                          })
        st.plotly_chart(fig_churn, use_container_width=True)
        
        # Show high-risk customers
        st.subheader("High-Risk Customers")
        high_risk = churn_df[churn_df['risk_level'] == 'High'].sort_values('probability', ascending=False)
        st.dataframe(high_risk)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Plan Performance
        st.markdown('<div class="ml-section">', unsafe_allow_html=True)
        st.markdown('<h2>Plan Performance Analysis</h2>', unsafe_allow_html=True)
        
        # Get all subscriptions
        all_subs = list(subscriptions_collection.find())
        
        # Count subscriptions by plan
        plan_counts = {}
        for sub in all_subs:
            plan_name = sub.get('plan', 'Unknown')
            plan_counts[plan_name] = plan_counts.get(plan_name, 0) + 1
        
        # Create DataFrame
        plan_df = pd.DataFrame(list(plan_counts.items()), columns=['Plan', 'Subscriptions'])
        
        # Create bar chart
        fig_plan = px.bar(plan_df, x='Plan', y='Subscriptions', 
                         title='Plan Popularity',
                         color='Subscriptions',
                         color_continuous_scale=px.colors.sequential.Viridis)
        st.plotly_chart(fig_plan, use_container_width=True)
        
        # Plan profitability
        st.subheader("Plan Profitability")
        profitability_data = []
        
        for plan in plans_collection.find():
            active_count = subscriptions_collection.count_documents({"plan": plan.get('name'), "status": "Active"})
            price = int(plan.get('price', '$0').replace('$', '').replace(',', ''))
            # Assume cost is 60% of price
            cost = price * 0.6
            revenue = active_count * price
            total_cost = active_count * cost
            profit = revenue - total_cost
            
            profitability_data.append({
                'Plan': plan.get('name'),
                'Revenue': revenue,
                'Cost': total_cost,
                'Profit': profit,
                'Profit Margin': (profit / revenue * 100) if revenue > 0 else 0
            })
        
        profit_df = pd.DataFrame(profitability_data)
        st.dataframe(profit_df)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Usage Forecasting
        st.markdown('<div class="ml-section">', unsafe_allow_html=True)
        st.markdown('<h2>Network Usage Forecasting</h2>', unsafe_allow_html=True)
        
        # Generate sample network usage data
        dates = pd.date_range(start='2023-01-01', periods=24, freq='M')
        usage = np.random.randint(5000, 15000, 24)
        
        # Create DataFrame
        df = pd.DataFrame({'date': dates, 'usage': usage})
        df.set_index('date', inplace=True)
        
        # Train model
        model = ExponentialSmoothing(df['usage'], trend='add', seasonal='add', seasonal_periods=12)
        fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.forecast(12)
        
        # Create forecast DataFrame
        forecast_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(months=1), periods=12, freq='M')
        forecast_df = pd.DataFrame({'date': forecast_dates, 'forecast': forecast})
        
        # Create chart
        fig_network = px.line(
            pd.concat([df.reset_index(), forecast_df]),
            x='date',
            y=['usage', 'forecast'],
            title='Network Usage Forecast',
            labels={'value': 'Usage (GB)', 'date': 'Month'},
            color_discrete_map={
                'usage': '#667eea',
                'forecast': '#28a745'
            }
        )
        st.plotly_chart(fig_network, use_container_width=True)
        
        st.info("This forecast helps in network capacity planning and infrastructure investment decisions.")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # User History View
    elif st.session_state['admin_nav'] == 'history':
        st.markdown('<h1 class="card-title">User History</h1>', unsafe_allow_html=True)
        
        # User selection
        users = list(users_collection.find({"role": "user"}))
        user_options = {f"{user.get('full_name')} ({user.get('username')})": user.get('username') for user in users}
        
        selected_user = st.selectbox("Select User", options=list(user_options.keys()))
        username = user_options[selected_user]
        
        # Get user details
        user = users_collection.find_one({"username": username})
        subscriptions = list(subscriptions_collection.find({"username": username}))
        
        # User Information
        st.markdown('<div class="enhanced-card">', unsafe_allow_html=True)
        st.markdown('<h3>User Information</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <p><strong>Username:</strong> {user.get('username', 'N/A')}</p>
            <p><strong>Full Name:</strong> {user.get('full_name', 'N/A')}</p>
            <p><strong>Email:</strong> {user.get('email', 'N/A')}</p>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <p><strong>Phone:</strong> {user.get('phone', 'N/A')}</p>
            <p><strong>Member Since:</strong> {user.get('created_at').strftime('%Y-%m-%d') if user.get('created_at') else 'N/A'}</p>
            <p><strong>Usage (GB):</strong> {user.get('usage_gb', 'N/A')}</p>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Subscription History
        st.markdown('<h3>Subscription History</h3>', unsafe_allow_html=True)
        
        for sub in subscriptions:
            status_class = "active" if sub.get('status') == 'Active' else "expired"
            status_badge = "Active" if sub.get('status') == 'Active' else "Expired"
            
            st.markdown(f"""
            <div class="subscription-card {status_class}">
                <div class="card-header">
                    <h3>{sub.get('plan', 'Unknown Plan')}</h3>
                    <span class="badge {status_class}">{status_badge}</span>
                </div>
                <p><strong>Start Date:</strong> {sub.get('start_date').strftime('%Y-%m-%d') if sub.get('start_date') else 'N/A'}</p>
                <p><strong>End Date:</strong> {sub.get('end_date').strftime('%Y-%m-%d') if sub.get('end_date') else 'N/A'}</p>
            </div>
            """, unsafe_allow_html=True)

def bootstrap_data():
    # Create admin user if not exists
    if users_collection.count_documents({}) == 0:
        admin_password = hash_password("admin123")
        users_collection.insert_one({
            "username": "admin",
            "full_name": "Admin User",
            "email": "admin@broadband.com",
            "phone": "555-0000",
            "password": admin_password,
            "role": "admin",
            "created_at": datetime.utcnow()
        })
    
    # Create default plans if not exists
    if plans_collection.count_documents({}) == 0:
        plans_collection.insert_many([
            {
                "name": "Starter Plan",
                "price": "$10",
                "speed": "50 Mbps",
                "desc": "Perfect for light browsing & emails.",
                "category": "Basic",
                "data": "100 GB/month"
            },
            {
                "name": "Pro Plan",
                "price": "$25",
                "speed": "200 Mbps",
                "desc": "Great for streaming and gaming.",
                "category": "Standard",
                "data": "Unlimited"
            },
            {
                "name": "Ultra Plan",
                "price": "$50",
                "speed": "1 Gbps",
                "desc": "Best for businesses & heavy users.",
                "category": "Premium",
                "data": "Unlimited"
            },
            {
                "name": "Business Basic",
                "price": "$75",
                "speed": "500 Mbps",
                "desc": "Essential for small businesses.",
                "category": "Business",
                "data": "Unlimited",
                "features": ["24/7 Support", "Static IP", "Priority Service"]
            },
            {
                "name": "Business Pro",
                "price": "$150",
                "speed": "1 Gbps",
                "desc": "Complete solution for medium businesses.",
                "category": "Business",
                "data": "Unlimited",
                "features": ["24/7 Support", "Static IP", "SLA Guarantee", "Dedicated Account Manager"]
            }
        ])
    
    # Create default users and subscriptions if not exists
    if users_collection.count_documents({"role": "user"}) == 0:
        # Create sample users
        sample_users = [
            {
                "username": "john_doe",
                "full_name": "John Doe",
                "email": "john@example.com",
                "phone": "555-1234",
                "password": hash_password("password123"),
                "role": "user",
                "created_at": datetime.utcnow() - timedelta(days=180),
                "usage_gb": 120,
                "satisfaction": 4,
                "tenure_months": 6,
                "support_tickets": 2,
                "payment_delays": 0,
                "plan_changes": 1
            },
            {
                "username": "jane_smith",
                "full_name": "Jane Smith",
                "email": "jane@example.com",
                "phone": "555-5678",
                "password": hash_password("password123"),
                "role": "user",
                "created_at": datetime.utcnow() - timedelta(days=90),
                "usage_gb": 250,
                "satisfaction": 3,
                "tenure_months": 3,
                "support_tickets": 4,
                "payment_delays": 1,
                "plan_changes": 2
            },
            {
                "username": "bob_wilson",
                "full_name": "Bob Wilson",
                "email": "bob@example.com",
                "phone": "555-9012",
                "password": hash_password("password123"),
                "role": "user",
                "created_at": datetime.utcnow() - timedelta(days=30),
                "usage_gb": 80,
                "satisfaction": 5,
                "tenure_months": 1,
                "support_tickets": 0,
                "payment_delays": 0,
                "plan_changes": 0
            }
        ]
        
        users_collection.insert_many(sample_users)
        
        # Create sample subscriptions
        sample_subscriptions = [
            # John Doe's subscriptions
            {
                "username": "john_doe",
                "plan": "Starter Plan",
                "start_date": datetime.utcnow() - timedelta(days=180),
                "end_date": datetime.utcnow() - timedelta(days=150),
                "status": "Expired"
            },
            {
                "username": "john_doe",
                "plan": "Pro Plan",
                "start_date": datetime.utcnow() - timedelta(days=150),
                "end_date": datetime.utcnow() + timedelta(days=30),
                "status": "Active"
            },
            # Jane Smith's subscriptions
            {
                "username": "jane_smith",
                "plan": "Pro Plan",
                "start_date": datetime.utcnow() - timedelta(days=90),
                "end_date": datetime.utcnow() - timedelta(days=60),
                "status": "Expired"
            },
            {
                "username": "jane_smith",
                "plan": "Ultra Plan",
                "start_date": datetime.utcnow() - timedelta(days=60),
                "end_date": datetime.utcnow() + timedelta(days=30),
                "status": "Active"
            },
            # Bob Wilson's subscriptions
            {
                "username": "bob_wilson",
                "plan": "Starter Plan",
                "start_date": datetime.utcnow() - timedelta(days=30),
                "end_date": datetime.utcnow() + timedelta(days=30),
                "status": "Active"
            }
        ]
        
        subscriptions_collection.insert_many(sample_subscriptions)

# ----------------- Session State -----------------
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'
if 'role' not in st.session_state:
    st.session_state['role'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None

# ----------------- Routing -----------------
page = st.session_state['page']
role = st.session_state['role']

# Replace your main app flow with this
def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state['page'] = 'home'
    
    # Display appropriate page
    if st.session_state['page'] == 'home':
        landing_page()
    elif st.session_state['page'] == 'login':
        login_page()
    elif st.session_state['page'] == 'register':
        register_page()
    elif st.session_state['page'] == 'user':
        enhanced_user_dashboard()
    elif st.session_state['page'] == 'admin':
        enhanced_admin_dashboard()

if __name__ == '__main__':
    main()