# ✈️ UDAN 2024 - Comprehensive Exploratory Data Analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup
sns.set(style="whitegrid")
%matplotlib inline

# Paths to cleaned data (adjust if needed)
routes_path = "../data/processed/cleaned_routes.csv"
projects_path = "../data/processed/cleaned_projects.csv"
funding_path = "../data/processed/cleaned_funding.csv"

# --- Load datasets ---
routes_df = pd.read_csv(routes_path)
projects_df = pd.read_csv(projects_path)
funding_df = pd.read_csv(funding_path)

# --- Inspect datasets ---
print("Routes Data Sample:")
display(routes_df.head())

print("Projects Data Sample:")
display(projects_df.head())

print("Funding Data Sample:")
display(funding_df.head())

# ===========================
# SECTION 1: Top States by Valid Routes
# ===========================
print("\n--- Top States by Valid UDAN Routes ---")
top_states = (
    routes_df.groupby("state_ut")["valid_routes"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
display(top_states)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_states.values, y=top_states.index, palette="Blues_r")
plt.title("Top 10 States/UTs by Valid UDAN Routes")
plt.xlabel("Number of Valid Routes")
plt.ylabel("State/UT")
plt.tight_layout()
plt.show()

# ===========================
# SECTION 2: Budget Allocation vs Expenditure by Airport
# ===========================
print("\n--- Budget Allocation vs Expenditure (Top 10 Airports) ---")
top_funding = funding_df.sort_values("amount_allocated_in_crore", ascending=False).head(10)
display(top_funding[["airport", "amount_allocated_in_crore", "amount_spent_in_crore"]])

plt.figure(figsize=(12, 6))
sns.barplot(data=top_funding, x="airport", y="amount_allocated_in_crore", color="skyblue", label="Allocated")
sns.barplot(data=top_funding, x="airport", y="amount_spent_in_crore", color="navy", label="Spent")
plt.title("Top 10 Airports by Allocation and Expenditure (₹ Crore)")
plt.xticks(rotation=45)
plt.ylabel("Amount (₹ Crore)")
plt.legend()
plt.tight_layout()
plt.show()

# ===========================
# SECTION 3: Top Airports by Budget Utilization (%)
# ===========================
print("\n--- Top Airports by Budget Utilization (%) ---")
top_utilized = funding_df.sort_values("utilization_pct", ascending=False).dropna(subset=["utilization_pct"]).head(10)
display(top_utilized[["airport", "utilization_pct"]])

plt.figure(figsize=(10, 6))
sns.barplot(data=top_utilized, x="utilization_pct", y="airport", palette="viridis")
plt.title("Top 10 Airports by Budget Utilization (%)")
plt.xlabel("Utilization %")
plt.ylabel("Airport")
plt.tight_layout()
plt.show()

# ===========================
# SECTION 4: Project Status Distribution
# ===========================
print("\n--- Project Status Distribution ---")
status_counts = projects_df["project_status"].value_counts()
display(status_counts)

plt.figure(figsize=(6, 6))
status_counts.plot.pie(
    autopct="%1.1f%%",
    startangle=140,
    colors=["#66b3ff", "#ff9999"]
)
plt.title("Distribution of Project Status under UDAN")
plt.ylabel("")
plt.tight_layout()
plt.show()

# ===========================
# SECTION 5: Correlation between Routes and Projects by State
# ===========================
print("\n--- Correlation: Routes vs Projects by State ---")
routes_summary = routes_df.groupby("state_ut")["valid_routes"].sum().reset_index()
projects_summary = projects_df.groupby("state_ut")["project_status"].count().reset_index().rename(columns={"project_status": "project_count"})

merged_df = pd.merge(routes_summary, projects_summary, on="state_ut", how="left").fillna(0)
display(merged_df.head())

plt.figure(figsize=(10, 6))
sns.scatterplot(data=merged_df, x="valid_routes", y="project_count", hue="state_ut", legend=False, s=100)
plt.title("Correlation between Valid Routes and Number of Projects by State")
plt.xlabel("Number of Valid Routes")
plt.ylabel("Number of Projects")
plt.tight_layout()
plt.show()
