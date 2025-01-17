import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from radon.complexity import cc_visit
from radon.metrics import mi_visit
from radon.raw import analyze
from radon.visitors import ComplexityVisitor

# Streamlit App Title and Description
st.set_page_config(page_title="Code Metrics Dashboard", layout="wide")
st.title("📊 Advanced Code Metrics Dashboard")
st.markdown("""
This interactive dashboard provides a **comprehensive analysis of code quality** across multiple programming languages.

### Key Features:
- Cyclomatic Complexity
- Maintainability Index
- Halstead Metrics
- Raw Metrics (Lines of Code, Comments, Blank Lines)
- Advanced Maintainability Index (Maintainability Index)
- Cohesion and Magic Numbers detection

📂 **Upload Python files** or **CSV/Excel files** containing code metrics to begin your analysis.
""")

# Column RenMaintainability Indexng Dictionary
dict = {
    'std.code.complexity:cyclomatic': 'Avg Cyclomatic Complexity',
    'std.code.lines:comments': 'Comments',
    'std.code.lines:code': 'LOC (Lines of Code)',
    'std.code.magic:numbers': 'Magic Numbers'
}

# File Upload Section
uploaded_files = st.file_uploader(
    "📥 Upload Python, CSV, or Excel files for analysis", 
    accept_multiple_files=True, 
    type=["py", "csv", "xlsx"]
)

# Helper Function: Calculate Magic Numbers
def calculate_magic_numbers(code):
    visitor = ComplexityVisitor.from_code(code)
    magic_numbers = sum(len(node.magic_numbers) for node in visitor.functions)
    return magic_numbers

# Helper Function: Calculate Cohesion (Experimental)
def calculate_cohesion(file_path):
    try:
        from radon.cli.harvest import CCHarvester
        harvester = CCHarvester([file_path])
        results = harvester.results
        cohesion = len(results)
        return cohesion
    except Exception:
        return None

# Process Uploaded Files
if uploaded_files:
    data = []
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name

        if file_name.endswith(".py"):
            # Process Python files
            code = uploaded_file.read().decode("utf-8")

            # Save code temporarily for cohesion analysis
            temp_file_path = f"temp_{file_name}"
            with open(temp_file_path, "w") as f:
                f.write(code)

            # Cyclomatic Complexity
            complexity = cc_visit(code)
            total_cc = sum(c.complexity for c in complexity)
            avg_cc = total_cc / len(complexity) if complexity else 0

            # Raw Metrics
            raw_metrics = analyze(code)
            loc = raw_metrics.loc
            comments = raw_metrics.comments
            blank_lines = raw_metrics.blank

            # Maintainability Index
            mi = mi_visit(code, multi=False)

            # Magic Numbers
            magic_numbers = calculate_magic_numbers(code)

            # Cohesion
            cohesion = calculate_cohesion(temp_file_path)

            # Clean up temporary file
            os.remove(temp_file_path)

            # Append results
            data.append({
                "file": file_name,
                "Avg Cyclomatic Complexity": avg_cc,
                "LOC (Lines of Code)": loc,
                "Comments": comments,
                "Blank Lines": blank_lines,
                "Maintainability Index": mi,
                "Magic Numbers": magic_numbers,
                "Cohesion (Experimental)": cohesion
            })

        elif file_name.endswith((".csv", ".xlsx")):
            # Process CSV/Excel files
            if file_name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            data.extend(df.to_dict(orient="records"))

    # Create DataFrame
    df = pd.DataFrame(data)
    df = df.fillna(0)

    # Normalize Data for Maintainability Index Calculation
    scaler = MinMaxScaler()
    if file_name.endswith(".py"):
        normalized_df = df.drop(columns=["Cohesion (Experimental)"])
    else:
        df = df.rename(columns=dict)
        normalized_df = df.drop(columns=["file", "region", "type", "modified"])
        normalized_df = pd.DataFrame(scaler.fit_transform(normalized_df), columns=normalized_df.columns)

    # Maintainability Index Calculation
    alpha, beta, gamma = 4/7, 3/7, 1/7
    normalized_df["Maintainability Index"] = (
        alpha * np.log(1 + normalized_df["Avg Cyclomatic Complexity"]) +
        beta * np.log(1 + normalized_df["Comments"]) +
        beta * (1 - np.exp(-normalized_df["LOC (Lines of Code)"])) +
        gamma / (1 + normalized_df["Magic Numbers"])
    )
    df["Maintainability Index"] = normalized_df["Maintainability Index"]

    # Display DataFrame
    st.write("### 📝 Code Analysis Results")
    st.dataframe(df)

    # Download Results
    if st.button("📥 Save Results as CSV"):
        df.to_csv("code_metrics_results.csv", index=False)
        st.success("Results saved successfully as `code_metrics_results.csv`!")

    # Statistics and Insights
    st.write("### 📊 Statistics and Insights")
    st.write("#### Summary Statistics")
    st.dataframe(df.describe())

    st.write("#### Key Insights")
    if df["Maintainability Index"].min() < 65:
        st.warning("⚠️ Some files have a low Maintainability Index (below 65). Consider refactoring.")
    else:
        st.success("✅ All files have a good Maintainability Index.")

    if df["Maintainability Index"].mean() > 0.7:
        st.success("🎉 The average Maintainability Index indicates excellent maintainability!")
    else:
        st.warning("🔍 The average Maintainability Index suggests room for improvement in maintainability.")

    # Visualizations
    st.write("### 📈 Visualizations")

    # Visualization 1: Cyclomatic Complexity vs. LOC
    st.write("#### Cyclomatic Complexity vs. LOC")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="LOC (Lines of Code)", y="Avg Cyclomatic Complexity", hue="file", s=100, ax=ax)
    plt.title("Cyclomatic Complexity vs. LOC")
    st.pyplot(fig)

    # Visualization 2: Maintainability Index per File
    st.write("#### Maintainability Index per File")
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="file", y="Maintainability Index", palette="viridis", ax=ax)
    plt.title("Maintainability Index per File")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    # Visualization 3: Maintainability Index Distribution
    st.write("#### Maintainability Index Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["Maintainability Index"], kde=True, ax=ax, bins=20, color="blue")
    plt.title("Distribution of Advanced Maintainability Index (Maintainability Index)")
    st.pyplot(fig)

else:
    st.info("Please upload Python files or data files to begin the analysis.")
