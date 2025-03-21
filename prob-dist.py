import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Custom CSS for styling
st.markdown("""
<style>
    /* Light mode styles */
    body {
        background-color: #f0f2f6;
        color: #333333; /* Dark text for readability */
    }
    .stApp { background-color: #f0f2f6; }
    .stSidebar { background-color: #ffffff; }
    h1, h2, h3, p, label, .stTextInput label, .stSelectbox label, .stSlider label, .stRadio label, .stCheckbox label {
        color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)

# Dark mode toggle
dark_mode = st.sidebar.checkbox("Dark Mode", value=False)
if dark_mode:
    st.markdown("""
    <style>
        /* Dark mode styles */
        body {
            background-color: #1e1e1e;
            color: #ffffff; /* Light text for dark mode */
        }
        .stApp { background-color: #1e1e1e; }
        .stSidebar { background-color: #2e2e2e; }
        h1, h2, h3, p, label, .stTextInput label, .stSelectbox label, .stSlider label, .stRadio label, .stCheckbox label {
            color: #ffffff !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown("<h1 style='white-space: nowrap;'>📊 Probability Distribution Explorer</h1>", unsafe_allow_html=True)
st.write("Explore different probability distributions by adjusting their parameters.")

# Sidebar for distribution selection
dist_options = ["Normal", "Binomial","Bernoulli", "Poisson","Uniform", "Exponential", "Beta", "Gamma", "Log-Normal", "Laplacian"]
dist_choice = st.sidebar.selectbox("Choose a Distribution", dist_options)

# Sidebar for selecting plot type
plot_type = st.sidebar.radio("Select Plot Type", ["PDF", "CDF"])

# Sidebar mean
show_stats = st.sidebar.checkbox("Show Mean", value=False)


# colors for plots
colors = {
    "Normal": "blue", "Binomial": "green", "Bernoulli":"yellow","Poisson": "red","Uniform":"Magenta", "Exponential": "purple",
    "Beta": "orange", "Gamma": "brown", "Log-Normal": "pink", "Laplacian": "cyan"
}

# Parameter selection based on distribution
if dist_choice == "Normal":
    mean = st.sidebar.slider("Mean (μ)", -10.0, 10.0, 0.0)
    std = st.sidebar.slider("Standard Deviation (σ)", 0.1, 5.0, 1.0)
    x = np.linspace(mean - 4*std, mean + 4*std, 1000)
    pdf = stats.norm.pdf(x, mean, std)
    cdf = stats.norm.cdf(x, mean, std)

elif dist_choice == "Bernoulli":
    p = st.sidebar.slider("Probability of Success (p)", 0.0, 1.0, 0.5)
    x = np.array([0, 1])
    pdf = stats.bernoulli.pmf(x, p)
    cdf = stats.bernoulli.cdf(x, p)
    mean = p

elif dist_choice == "Binomial":
    n = st.sidebar.slider("Number of Trials (n)", 1, 100, 10)
    p = st.sidebar.slider("Probability of Success (p)", 0.0, 1.0, 0.5)
    x = np.arange(0, n+1)
    pdf = stats.binom.pmf(x, n, p)
    cdf = stats.binom.cdf(x, n, p)
    mean = n * p

elif dist_choice == "Poisson":
    lam = st.sidebar.slider("Lambda (λ)", 0.1, 10.0, 2.0)
    x = np.arange(0, 20)
    pdf = stats.poisson.pmf(x, lam)
    cdf = stats.poisson.cdf(x, lam)
    mean = lam

elif dist_choice == "Uniform":
    a = st.sidebar.slider("Lower Bound (a)", -10.0, 0.0, -5.0)
    b = st.sidebar.slider("Upper Bound (b)", 0.0, 10.0, 5.0)
    x = np.linspace(a, b, 1000)
    pdf = stats.uniform.pdf(x, loc=a, scale=b-a)
    cdf = stats.uniform.cdf(x, loc=a, scale=b-a)
    mean = (a + b) / 2


elif dist_choice == "Exponential":
    lam = st.sidebar.slider("Lambda (λ)", 0.1, 5.0, 1.0)
    x = np.linspace(0, 10, 1000)
    pdf = stats.expon.pdf(x, scale=1/lam)
    cdf = stats.expon.cdf(x, scale=1/lam)
    mean = 1/lam

elif dist_choice == "Beta":
    a = st.sidebar.slider("Alpha (α)", 0.1, 5.0, 2.0)
    b = st.sidebar.slider("Beta (β)", 0.1, 5.0, 2.0)
    x = np.linspace(0, 1, 1000)
    pdf = stats.beta.pdf(x, a, b)
    cdf = stats.beta.cdf(x, a, b)
    mean = a / (a + b)

elif dist_choice == "Gamma":
    shape = st.sidebar.slider("Shape (k)", 0.1, 10.0, 2.0)
    scale = st.sidebar.slider("Scale (θ)", 0.1, 5.0, 1.0)
    x = np.linspace(0, 20, 1000)
    pdf = stats.gamma.pdf(x, shape, scale=scale)
    cdf = stats.gamma.cdf(x, shape, scale=scale)
    mean = shape * scale

elif dist_choice == "Log-Normal":
    mean_log = st.sidebar.slider("Mean (μ)", 0.1, 5.0, 1.0)
    sigma = st.sidebar.slider("Sigma (σ)", 0.1, 2.0, 0.5)
    x = np.linspace(0, 10, 1000)
    pdf = stats.lognorm.pdf(x, sigma, scale=np.exp(mean_log))
    cdf = stats.lognorm.cdf(x, sigma, scale=np.exp(mean_log))
    mean = np.exp(mean_log + (sigma**2) / 2)

elif dist_choice == "Laplacian":
    mu = st.sidebar.slider("Mean (μ)", -10.0, 10.0, 0.0)
    b = st.sidebar.slider("Scale (b)", 0.1, 5.0, 1.0)
    x = np.linspace(mu - 10*b, mu + 10*b, 1000)
    pdf = stats.laplace.pdf(x, loc=mu, scale=b)
    cdf = stats.laplace.cdf(x, loc=mu, scale=b)
    mean = mu

# Plot the selected type
fig, ax = plt.subplots(figsize=(6,4))
if plot_type == "PDF":
    ax.plot(x, pdf, label=f"{dist_choice} PDF", color=colors[dist_choice])
    ax.fill_between(x, pdf, alpha=0.3, color=colors[dist_choice])
    ax.set_ylabel("Probability Density")
elif plot_type == "CDF":
    ax.plot(x, cdf, label=f"{dist_choice} CDF", color=colors[dist_choice])
    ax.set_ylabel("Cumulative Probability")

if show_stats:
    ax.axvline(mean, color='red', linestyle='--', label=f"Mean: {mean:.2f}")

ax.set_title(f"{dist_choice} {plot_type}")
ax.set_xlabel("x")
ax.legend()
st.pyplot(fig)


