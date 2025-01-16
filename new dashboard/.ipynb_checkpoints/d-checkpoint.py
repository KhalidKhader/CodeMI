import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from radon.raw import analyze
from radon.metrics import h_visit  # For Halstead Metrics
from radon.cli.harvest import CCHarvester  # For advanced metrics like cohesion

# Streamlit App Title and Description
st.title("Comprehensive Code Metrics Dashboard")
st.write("""
This dashboard provides a detailed analysis of uploaded Python code files using the Radon library and additional metrics. 
Metrics include:
- Cyclomatic Complexity
- Maintainability Index
- Halstead Metrics
- Raw Metrics (LOC, SLOC, Comments, Multi-line Comments, Blank Lines)
- Cohesion and Advanced Metrics
Use this analysis to assess code quality and identify areas for improvement.
""")

# File Upload Section
uploaded_files = st.file_uploader(
    "Upload Python files for analysis", accept_multiple_files=True, type=["py"]
)

# Function to calculate Cohesion using Radon
def calculate_cohesion(file_path):
    """Calculate cohesion using Radon's CCHarvester."""
    try:
        harvester = CCHarvester([file_path])
        results = harvester.results
        cohesion = len(results)
        return cohesion
    except Exception as e:
        return None

# Process the uploaded files
if uploaded_files:
    data = []

    for uploaded_file in uploaded_files:
        # Read the file content
        code = uploaded_file.read().decode("utf-8")
        file_name = uploaded_file.name

        # Save the code to a temporary file for cohesion analysis
        temp_file_path = f"temp_{file_name}"
        with open(temp_file_path, "w") as f:
            f.write(code)

        # Cyclomatic Complexity
        complexity = cc_visit(code)
        total_cyclomatic_complexity = sum([c.complexity for c in complexity])
        avg_cyclomatic_complexity = (
            total_cyclomatic_complexity / len(complexity) if complexity else 0
        )

        # Raw Metrics
        raw_metrics = analyze(code)
        loc = raw_metrics.loc
        sloc = raw_metrics.sloc
        comments = raw_metrics.comments
        multi = raw_metrics.multi
        blank = raw_metrics.blank

        # Maintainability Index
        maintainability_index = mi_visit(code, multi=False)

        # Halstead Metrics
        halstead_metrics = h_visit(code)
        halstead_volume = halstead_metrics.total.volume
        halstead_effort = halstead_metrics.total.effort
        halstead_difficulty = halstead_metrics.total.difficulty

        # Cohesion
        cohesion = calculate_cohesion(temp_file_path)

        # Delete the temporary file
        os.remove(temp_file_path)

        # Store results
        data.append({
            "File": file_name,
            "Total Cyclomatic Complexity": total_cyclomatic_complexity,
            "Avg Cyclomatic Complexity": avg_cyclomatic_complexity,
            "LOC (Lines of Code)": loc,
            "SLOC (Source Lines of Code)": sloc,
            "Comments": comments,
            "Multi-line Comments": multi,
            "Blank Lines": blank,
            "Maintainability Index": maintainability_index,
            "Halstead Volume": halstead_volume,
            "Halstead Effort": halstead_effort,
            "Halstead Difficulty": halstead_difficulty,
            "Cohesion (Experimental)": cohesion,
        })

    # Create a DataFrame for the analysis
    df = pd.DataFrame(data)

    # Display the DataFrame
    st.write("### Code Analysis Results")
    st.dataframe(df)

    # Add options for downloading the data
    if st.button("Save Results as CSV"):
        df.to_csv("code_metrics.csv", index=False)
        st.success("Results saved successfully as `code_metrics.csv`")

    # Visualization Section
    st.write("### Visualizations")
    st.write("Explore the code metrics using the charts below.")

    # Visualization 1: Cyclomatic Complexity vs. LOC
    st.write("#### Cyclomatic Complexity vs. LOC")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="LOC (Lines of Code)", y="Total Cyclomatic Complexity", hue="File", s=100, ax=ax)
    plt.title("Cyclomatic Complexity vs. LOC")
    plt.xlabel("Lines of Code")
    plt.ylabel("Cyclomatic Complexity")
    st.pyplot(fig)

    # Visualization 2: Maintainability Index per File
    st.write("#### Maintainability Index for Each File")
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="File", y="Maintainability Index", palette="viridis", ax=ax)
    plt.title("Maintainability Index per File")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Maintainability Index")
    st.pyplot(fig)

    # Visualization 3: Halstead Volume vs. Difficulty
    st.write("#### Halstead Volume vs. Difficulty")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Halstead Difficulty", y="Halstead Volume", hue="File", s=100, ax=ax)
    plt.title("Halstead Volume vs. Difficulty")
    plt.xlabel("Halstead Difficulty")
    plt.ylabel("Halstead Volume")
    st.pyplot(fig)

    # Additional Insights
    st.write("### Insights")
    if df["Maintainability Index"].min() < 65:
        st.warning("⚠️ Some files have a low Maintainability Index (below 65). Consider refactoring the code.")
    else:
        st.success("✅ All files have a good Maintainability Index.")
    
    st.info(
        f"""
        **Summary**:
        - Total Files Analyzed: {len(df)}
        - Average Cyclomatic Complexity (Total): {df['Total Cyclomatic Complexity'].mean():.2f}
        - Average Maintainability Index: {df['Maintainability Index'].mean():.2f}
        - Average Halstead Volume: {df['Halstead Volume'].mean():.2f}
        """
    )

else:
    st.info("Please upload one or more Python files to begin the analysis.")
