
import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import json
import os
import ast
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Job Recommendation System",
    page_icon="💼",
    layout="wide"
)


# ==========================================
# GLOBAL APPLICATION STYLING
# ==========================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #F5F7FB;
    color: #172033;
    font-family: "Inter", "Segoe UI", sans-serif;
}

/* Main content area */
.block-container {
    max-width: 1450px;
    padding-top: 2.5rem !important;
    padding-bottom: 2rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
}

/* Top Streamlit header */
header[data-testid="stHeader"] {
    background-color: #F5F7FB;
    height: 3.5rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 1px solid #E5EAF2;
    min-width: 245px !important;
    max-width: 245px !important;
    width: 245px !important;
    transform: none !important;
}

/* Hide sidebar collapse/minimize button */
[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebar"] button[kind="header"] {
    display: none !important;
}

/* Sidebar content spacing */
section[data-testid="stSidebar"] > div {
    padding-top: 2rem;
}

/* Sidebar text */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label {
    color: #334155;
    font-size: 0.9rem;
}

/* Main headings */
h1 {
    color: #172554 !important;
    font-size: 1.85rem !important;
    font-weight: 750 !important;
    letter-spacing: -0.5px;
}

h2 {
    color: #1E293B !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
}

h3 {
    color: #334155 !important;
    font-size: 1.05rem !important;
    font-weight: 650 !important;
}

/* Normal text */
p, li, label {
    color: #475569;
    font-size: 0.9rem;
    line-height: 1.55;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E5EAF2;
    border-radius: 12px;
    padding: 15px 18px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}

[data-testid="stMetricLabel"] {
    color: #64748B !important;
    font-size: 0.8rem !important;
}

[data-testid="stMetricValue"] {
    color: #1D4ED8 !important;
    font-size: 1.7rem !important;
    font-weight: 750 !important;
}

/* Buttons */
.stButton > button {
    background-color: #2563EB;
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.2rem;
    font-weight: 600;
    font-size: 0.88rem;
    transition: background-color 0.2s ease;
}

.stButton > button:hover {
    background-color: #1D4ED8;
    color: #FFFFFF;
    border: none;
}

/* Input fields and select boxes */
.stSelectbox div[data-baseweb="select"] > div,
.stTextInput input,
.stNumberInput input {
    background-color: #FFFFFF;
    border-color: #CBD5E1;
    color: #172033;
    border-radius: 8px;
}

/* Sliders */
.stSlider [data-baseweb="slider"] {
    color: #2563EB;
}

/* Expanders */
[data-testid="stExpander"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
}

/* Job recommendation cards / bordered containers */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.035);
}

/* Dividers */
hr {
    border-color: #E2E8F0;
}

/* Progress bars */
.stProgress > div > div > div > div {
    background-color: #2563EB;
}

/* Captions */
.stCaption, [data-testid="stCaptionContainer"] {
    color: #64748B !important;
    font-size: 0.78rem !important;
}

/* Responsive spacing on smaller screens */
@media (max-width: 900px) {
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1.5rem !important;
    }
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# LOAD DATA AND TRAINED MODELS
# ==========================================

@st.cache_data
def load_jobs():
    return pd.read_csv("data/processed/jobs_cleaned.csv")


@st.cache_data
def load_user_profiles():
    return pd.read_csv("data/processed/user_profiles_v2.csv")


@st.cache_data
def load_interactions():
    return pd.read_csv("data/processed/user_interactions_v2.csv")


@st.cache_resource
def load_models():

    combined_similarity = np.load(
        "models/combined_similarity.npy"
    )

    description_similarity = np.load(
        "models/description_similarity.npy"
    )

    skill_similarity = np.load(
        "models/skill_similarity.npy"
    )

    with open("models/job_graph.pkl", "rb") as file:
        job_graph = pickle.load(file)

    return (
        combined_similarity,
        description_similarity,
        skill_similarity,
        job_graph
    )


# Load datasets
jobs = load_jobs()
user_profiles = load_user_profiles()
interactions = load_interactions()

# Load trained models
(
    combined_similarity,
    description_similarity,
    skill_similarity,
    job_graph
) = load_models()


# ==========================================
# BUILD USER-JOB INTERACTION MATRIX
# ==========================================

@st.cache_data
def build_user_job_matrix(interactions):

    user_job_matrix = interactions.pivot_table(
        index="user_id",
        columns="job_id",
        values="rating",
        fill_value=0
    )

    return user_job_matrix


@st.cache_data
def build_user_similarity(user_job_matrix):

    similarity = cosine_similarity(user_job_matrix)

    user_similarity_df = pd.DataFrame(
        similarity,
        index=user_job_matrix.index,
        columns=user_job_matrix.index
    )

    return user_similarity_df


# Build recommendation data structures
user_job_matrix = build_user_job_matrix(interactions)

user_similarity_df = build_user_similarity(user_job_matrix)



# ==========================================
# CONTENT-BASED RECOMMENDATION
# ==========================================

# Map job IDs to their positions in the similarity matrix
job_id_to_index = {
    job_id: index
    for index, job_id in enumerate(jobs["id"])
}


def get_content_scores(user_id):

    user_history = interactions[
        interactions["user_id"] == user_id
    ]

    if user_history.empty:
        return {}

    job_indices = []
    ratings = []

    for _, row in user_history.iterrows():

        job_id = row["job_id"]

        if job_id in job_id_to_index:
            job_indices.append(job_id_to_index[job_id])
            ratings.append(row["rating"])

    if not job_indices:
        return {}

    # Calculate weighted job similarity scores
    scores = np.average(
        combined_similarity[:, job_indices],
        axis=1,
        weights=ratings
    )

    # Exclude jobs the user has already interacted with
    seen_jobs = set(user_history["job_id"])

    content_scores = {
        job_id: scores[index]
        for index, job_id in enumerate(jobs["id"])
        if job_id not in seen_jobs
    }

    return content_scores


# ==========================================
# COLLABORATIVE FILTERING
# ==========================================

def get_collaborative_scores(user_id, top_k=10):

    if user_id not in user_job_matrix.index:
        return {}

    # Find similar users
    similarities = user_similarity_df.loc[user_id].drop(user_id)

    neighbors = similarities.nlargest(top_k)
    neighbors = neighbors[neighbors > 0]

    if neighbors.empty:
        return {}

    # Get ratings from similar users
    neighbor_ratings = user_job_matrix.loc[neighbors.index]

    weighted_sum = neighbor_ratings.mul(
        neighbors,
        axis=0
    ).sum(axis=0)

    weighted_denominator = (
        neighbor_ratings.ne(0).mul(
            neighbors,
            axis=0
        ).sum(axis=0)
    )

    collaborative_scores = (
        weighted_sum /
        weighted_denominator.replace(0, np.nan)
    ).dropna()

    # Exclude jobs already interacted with
    seen_jobs = set(
        interactions[
            interactions["user_id"] == user_id
        ]["job_id"]
    )

    collaborative_scores = collaborative_scores[
        ~collaborative_scores.index.isin(seen_jobs)
    ]

    return collaborative_scores.to_dict()



# ==========================================
# GRAPH-BASED RECOMMENDATION
# ==========================================

def get_graph_scores(user_id, alpha=0.85):

    # Use the exact user ID stored in the graph
    user_node = user_id

    # Check whether the user exists
    if user_node not in job_graph:
        return {}

    # Personalized PageRank
    personalization = {
        node: 1.0 if node == user_node else 0.0
        for node in job_graph.nodes()
    }

    pagerank_scores = nx.pagerank(
        job_graph,
        alpha=alpha,
        personalization=personalization
    )

    # Identify jobs already interacted with
    seen_jobs = set(
        interactions[
            interactions["user_id"] == user_id
        ]["job_id"]
    )

    # Extract scores for unseen jobs
    graph_scores = {}

    for job_id in jobs["id"]:

        job_node = f"JOB_{job_id}"

        if job_node in pagerank_scores and job_id not in seen_jobs:
            graph_scores[job_id] = pagerank_scores[job_node]

    return graph_scores


# ==========================================
# SCORE NORMALIZATION
# ==========================================

def normalize_score_dict(scores):

    if not scores:
        return {}

    values = np.array(list(scores.values()), dtype=float)

    min_score = values.min()
    max_score = values.max()

    if max_score == min_score:
        return {
            job_id: 0.0
            for job_id in scores
        }

    normalized = (values - min_score) / (max_score - min_score)

    return dict(zip(scores.keys(), normalized))


# ==========================================
# HYBRID RECOMMENDATION
# ==========================================

def get_hybrid_scores(user_id):

    # Generate individual model scores
    content_scores = get_content_scores(user_id)

    cf_scores = get_collaborative_scores(
        user_id,
        top_k=10
    )

    graph_scores = get_graph_scores(user_id)

    # Normalize individual scores
    content_normalized = normalize_score_dict(content_scores)
    cf_normalized = normalize_score_dict(cf_scores)
    graph_normalized = normalize_score_dict(graph_scores)

    # Combine candidate jobs
    candidate_jobs = (
        set(content_normalized) |
        set(cf_normalized) |
        set(graph_normalized)
    )

    # Calculate weighted hybrid scores
    hybrid_scores = {}

    for job_id in candidate_jobs:

        hybrid_scores[job_id] = (
            0.50 * content_normalized.get(job_id, 0) +
            0.30 * cf_normalized.get(job_id, 0) +
            0.20 * graph_normalized.get(job_id, 0)
        )

    return hybrid_scores


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Explore Jobs",
        "Job Recommendations",
        "About"
    ]
)

# ==========================================
# HOME PAGE
# ==========================================

if page == "Home":
    
    st.title("💼 Job Recommendation System")

    st.markdown(
        """
        ### Discover jobs that match your skills and interests.

        Our recommendation system uses machine learning to help
        you find relevant job opportunities.
        """
    )

    st.divider()

    # Hero section
    st.markdown("""
    <div style="
        background: linear-gradient(120deg, #172554, #1D4ED8);
        padding: 25px 28px;
        border-radius: 14px;
        margin-bottom: 22px;
    ">
        <p style="color:#BFDBFE; font-size:12px; font-weight:600;">
            INTELLIGENT JOB DISCOVERY
        </p>
        <h1 style="color:white; font-size:28px;">
            SmartJob Recommender
        </h1>
        <p style="color:#DBEAFE; font-size:14px;">
            Discover job opportunities that match your skills,
            interests and career preferences through machine learning.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Platform statistics
    st.markdown("### Platform Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Available Jobs", f"{len(jobs):,}")

    with col2:
        st.metric("Companies", f"{jobs['company'].nunique():,}")

    with col3:
        st.metric("Job Locations", f"{jobs['location'].nunique():,}")

    with col4:
        st.metric("Recommendation Models", "4")

    st.markdown("---")

    # Recommendation models
    st.markdown("### Recommendation Technology")
    st.caption("Four machine learning approaches power job discovery.")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("**01 · Content-Based Filtering**")
            st.caption(
                "Matches jobs using similarities in descriptions "
                "and required technical skills."
            )

        with st.container(border=True):
            st.markdown("**02 · Collaborative Filtering**")
            st.caption(
                "Recommends opportunities based on interactions "
                "from users with similar preferences."
            )

    with col2:
        with st.container(border=True):
            st.markdown("**03 · Graph-Based Recommendation**")
            st.caption(
                "Uses user-job relationships and Personalized "
                "PageRank to identify relevant opportunities."
            )

        with st.container(border=True):
            st.markdown("**04 · Hybrid Recommendation**")
            st.caption(
                "Combines content, collaborative and graph-based "
                "scores into one recommendation ranking."
            )

    st.markdown("---")

    # How the platform works
    st.markdown("### How It Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("**1. Explore Jobs**")
            st.caption("Browse available opportunities and job details.")

    with col2:
        with st.container(border=True):
            st.markdown("**2. Get Matched**")
            st.caption("Choose a user profile and recommendation model.")

    with col3:
        with st.container(border=True):
            st.markdown("**3. Discover Opportunities**")
            st.caption("Explore your ranked and personalized job matches.")

    st.markdown("""
    <div style="
        text-align: center;
        padding: 12px 0 0 0;
        color: #94A3B8;
        font-size: 0.75rem;
    ">
        SmartJob Recommender · Machine Learning Powered
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# EXPLORE JOBS PAGE
# ==========================================


elif page == "Explore Jobs":

    st.title("Explore Available Jobs")
    st.write("Discover opportunities from our job database.")

    # Search jobs
    search = st.text_input(
        "Search by job title or company",
        placeholder="e.g. Data Analyst, Google..."
    )

    filtered_jobs = jobs.copy()

    if search:
        filtered_jobs = filtered_jobs[
            filtered_jobs["title"].str.contains(
                search, case=False, na=False
            ) |
            filtered_jobs["company"].str.contains(
                search, case=False, na=False
            )
        ]

    st.markdown("---")

    st.markdown(
        f"**{len(filtered_jobs):,} jobs found**"
    )

    # Display job cards
    
    # Pagination settings
    page_size = 20
    total_jobs = len(filtered_jobs)
    total_pages = max(1, (total_jobs + page_size - 1) // page_size)

    selected_page = st.selectbox(
        "Select Page",
        options=list(range(1, total_pages + 1)),
        format_func=lambda p: f"Page {p} of {total_pages}"
    )

    # Get jobs for the selected page
    start_index = (selected_page - 1) * page_size
    end_index = start_index + page_size

    display_jobs = filtered_jobs.iloc[start_index:end_index]

    st.caption(
        f"Showing {start_index + 1 if total_jobs else 0}–"
        f"{min(end_index, total_jobs)} of {total_jobs} jobs"
    )

    # Display job cards
    for _, job in display_jobs.iterrows():

        with st.container(border=True):

            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(f"### {job['title']}")
                st.write(f"**{job['company']}**")

                st.caption(
                    f"📍 {job['location']}  |  "
                    f"💼 {job['employment_type']}  |  "
                    f"🏢 {job['workplace_type']}"
                )

            with col2:
                st.markdown("**Salary Range**")
                st.write(job["salary_range"])

            with st.expander("View Job Details"):

                st.markdown("**Job Description**")
                st.write(job["description"])

                st.markdown("**Required Skills**")

                skills = job["technical_skills"]

                if isinstance(skills, str):
                    try:
                        skills = ast.literal_eval(skills)
                    except (ValueError, SyntaxError):
                        skills = [skills]

                if isinstance(skills, list):
                    st.write(", ".join(map(str, skills)))
                else:
                    st.write("Skills information unavailable")


# ==========================================
# JOB RECOMMENDATIONS PAGE
# ==========================================


elif page == "Job Recommendations":

    st.title("Job Recommendations")

    st.write(
        "Discover opportunities selected using intelligent "
        "machine learning recommendation models."
    )

    # Recommendation selection panel
    with st.container(border=True):

        st.subheader("Personalize Your Recommendations")

        col1, col2 = st.columns(2)

        with col1:
            selected_user = st.selectbox(
                "Select User Profile",
                user_profiles["user_id"].tolist()
            )

        with col2:
            selected_model = st.selectbox(
                "Recommendation Model",
                [
                    "Hybrid",
                    "Content-Based",
                    "Collaborative",
                    "Graph-Based"
                ]
            )

        top_n = st.slider(
            "Number of recommendations",
            min_value=5,
            max_value=50,
            value=5
        )

        generate = st.button(
            "Generate Recommendations",
            use_container_width=True
        )

    # Generate recommendations
    if generate:

        with st.spinner("Finding suitable job opportunities..."):

            if selected_model == "Hybrid":
                scores = get_hybrid_scores(selected_user)

            elif selected_model == "Content-Based":
                scores = get_content_scores(selected_user)

            elif selected_model == "Collaborative":
                scores = get_collaborative_scores(selected_user)

            else:
                scores = get_graph_scores(selected_user)

            if scores:

                all_jobs = sorted(
                    scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                recommendations = pd.DataFrame(
                    all_jobs,
                    columns=["id", "recommendation_score"]
                )

                recommendations = recommendations.merge(
                    jobs,
                    on="id",
                    how="left"
                )

                st.session_state["recommendations"] = recommendations
                st.session_state["recommendations_user"] = selected_user
                st.session_state["recommendations_model"] = selected_model

            else:
                st.session_state["recommendations"] = pd.DataFrame()

    # Display recommendations
    if (
        "recommendations" in st.session_state
        and not st.session_state["recommendations"].empty
    ):

        recommendations = st.session_state["recommendations"]

        st.success(
            f"Recommendations generated for "
            f"{st.session_state['recommendations_user']} "
            f"using {st.session_state['recommendations_model']}."
        )

        # Summary metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Available Matches",
                len(recommendations)
            )

        with col2:
            st.metric(
                "Selected Model",
                st.session_state["recommendations_model"]
            )

        with col3:
            st.metric(
                "Selected User",
                st.session_state["recommendations_user"]
            )

        st.divider()

        # Workplace selection
        workplace_options = ["All"] + sorted(
            recommendations["workplace_type"]
            .dropna()
            .unique()
            .tolist()
        )

        selected_workplace = st.selectbox(
            "Filter by Workplace Type",
            workplace_options
        )

        filtered_recommendations = recommendations.copy()

        if selected_workplace != "All":
            filtered_recommendations = filtered_recommendations[
                filtered_recommendations["workplace_type"]
                == selected_workplace
            ]

        displayed_recommendations = filtered_recommendations.head(top_n)

        st.subheader("Recommended Opportunities")

        st.caption(
            f"Showing {len(displayed_recommendations)} "
            f"of {len(filtered_recommendations)} recommendations"
        )

        if displayed_recommendations.empty:

            st.warning("No jobs match the selected workplace type.")

        else:

            for rank, (_, job) in enumerate(
                displayed_recommendations.iterrows(),
                start=1
            ):

                with st.container(border=True):

                    col1, col2 = st.columns([4, 1])

                    with col1:

                        st.markdown(
                            f"### {rank}. {job['title']}"
                        )

                        st.markdown(
                            f"**{job['company']}**"
                        )

                    with col2:

                        st.markdown("**Match Score**")

                        score = float(job["recommendation_score"])

                        st.markdown(
                            f"<h3 style='color:#2563EB;'>"
                            f"{score:.4f}</h3>",
                            unsafe_allow_html=True
                        )

                    st.caption(
                        f"📍 {job['location']}  |  "
                        f"💼 {job['employment_type']}  |  "
                        f"🏢 {job['workplace_type']}"
                    )

                    st.write(
                        f"**Salary:** {job['salary_range']}"
                    )

                    st.progress(
                        max(0.0, min(score, 1.0))
                    )

                    with st.expander("View Job Details"):

                        st.markdown("**Technical Skills**")

                        technical_skills = job["technical_skills"]

                        if pd.notna(technical_skills) and str(technical_skills).strip():

                            try:
                                skills = (
                                    ast.literal_eval(technical_skills)
                                    if isinstance(technical_skills, str)
                                    else technical_skills
                                )

                                if isinstance(skills, list):
                                    st.write(
                                        " · ".join(map(str, skills))
                                        if skills else "Not specified"
                                    )
                                else:
                                    st.write(skills)

                            except (ValueError, SyntaxError):
                                st.write(technical_skills)

                        else:
                            st.write("Not specified")

                        st.markdown("**Professional Skills**")

                        professional_skills = job["professional_skills"]

                        if pd.notna(professional_skills) and str(professional_skills).strip():

                            try:
                                skills = (
                                    ast.literal_eval(professional_skills)
                                    if isinstance(professional_skills, str)
                                    else professional_skills
                                )

                                if isinstance(skills, list):
                                    st.write(
                                        " · ".join(map(str, skills))
                                        if skills else "Not specified"
                                    )
                                else:
                                    st.write(skills)

                            except (ValueError, SyntaxError):
                                st.write(professional_skills)

                        else:
                            st.write("Not specified")

                        st.divider()

                        st.markdown("**Job Description**")
                        st.write(job["description"])

    elif "recommendations" in st.session_state:

        st.warning(
            "No recommendations found. Try another user or model."
        )



# ==========================================
# ABOUT PAGE
# ==========================================


elif page == "About":

    st.title("About SmartJob Recommender")

    st.write(
        "An intelligent job recommendation platform powered by "
        "machine learning techniques."
    )

    # Project overview
    st.markdown("## Project Overview")

    st.markdown("""
    SmartJob Recommender is designed to help job seekers discover
    relevant employment opportunities based on their interests,
    preferences and interaction history.

    The system combines multiple recommendation techniques to
    identify and rank suitable job opportunities.
    """)

    # Project statistics
    st.markdown("## Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Job Postings", "1,020")
    col2.metric("User Profiles", "4,000")
    col3.metric("Interactions", "80,224")
    col4.metric("Recommendation Models", "4")

    st.caption(
        "Job postings are sourced from a real job dataset. "
        "User profiles and interactions are synthetically generated "
        "for system development and evaluation."
    )

    st.divider()

    # Recommendation techniques
    st.markdown("## Recommendation Techniques")

    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("### Content-Based Filtering")
            st.write(
                "Recommends jobs based on similarities in job "
                "descriptions and required technical skills."
            )

        with st.container(border=True):
            st.markdown("### Graph-Based Recommendation")
            st.write(
                "Uses relationships between users and jobs in a "
                "graph structure to identify relevant opportunities "
                "through Personalized PageRank."
            )

    with col2:
        with st.container(border=True):
            st.markdown("### Collaborative Filtering")
            st.write(
                "Identifies users with similar interaction patterns "
                "and recommends jobs they have found relevant."
            )

        with st.container(border=True):
            st.markdown("### Hybrid Recommendation")
            st.write(
                "Combines content, collaborative and graph-based "
                "scores using a weighted recommendation approach."
            )

    st.divider()

    # Hybrid model weights
    st.markdown("## Hybrid Model Configuration")

    st.write(
        "The hybrid model combines the recommendation techniques "
        "using the following weights:"
    )

    weights = {
        "Content-Based": 50,
        "Collaborative": 30,
        "Graph-Based": 20
    }

    for model, weight in weights.items():
        st.write(f"**{model} — {weight}%**")
        st.progress(weight / 100)

    st.divider()

    # Evaluation
    st.markdown("## Model Evaluation")

    st.write(
        "The recommendation models were evaluated using "
        "Precision@5, Recall@5 and F1@5."
    )

    st.info(
        "Evaluation was performed on a sample of 1,000 users "
        "using a held-out test set."
    )

    evaluation_data = pd.DataFrame({
        "Model": [
            "Graph-Based",
            "Hybrid",
            "Content-Based",
            "Collaborative"
        ],
        "Precision@5": [0.2280, 0.2046, 0.1236, 0.0788],
        "Recall@5": [0.4396, 0.3934, 0.2348, 0.1485],
        "F1@5": [0.2817, 0.2522, 0.1525, 0.0965]
    })

    st.dataframe(
        evaluation_data,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("### Key Finding")

    st.success(
        "The Graph-Based model achieved the strongest overall "
        "evaluation results, followed by the Hybrid model."
    )

    st.divider()

    # Technology
    st.markdown("## Technologies Used")

    st.write(
        "Python · Pandas · NumPy · Scikit-learn · "
        "NetworkX · Streamlit"
    )

    st.caption(
        "SmartJob Recommender | Machine Learning Project"
    )

