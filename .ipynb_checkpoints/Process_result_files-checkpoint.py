#!/usr/bin/env python
# coding: utf-8

# In[93]:


import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer


# In[40]:


directory = "data/results/"  
csv_files = [file for file in os.listdir(directory) if file.endswith(".csv")]
dfs = []
for file_name in file_names:
    df = pd.read_csv(f"data/results/{file_name}", skiprows=1)
    project_name = os.path.basename(file_name).split("_result")[0]
    df.columns = df.columns.str.strip()  
    df = df[~df.isin(["Grand Total"]).any(axis=1)]  # Remove rows with "Grand Total"
    if "Grand Total" in df.columns:
        df = df.drop(columns=["Grand Total"])
    if "Rule set" in df.columns:
        df = df.rename(columns={"Count of Rule set": "Metric"})
        df["project"] = project_name
    else:
        df = df.rename(columns={df.columns[0]: "Metric"})
        df["project"] = project_name
    dfs.append(df)
combined_df = pd.concat(dfs, ignore_index=True)
combined_df.head()


# In[41]:


combined_df.info()


# In[42]:


combined_df.head(50)


# In[27]:


combined_df.to_csv('results_data.csv')


# In[153]:


combined_df.groupby('Rule set')['1'].mean()


# In[158]:


mean_values = combined_df.groupby('Rule set')['1'].mean()
# Plotting the results
plt.figure(figsize=(10, 6))
mean_values.plot(kind='bar', color='skyblue')
plt.title('Mean of Column "1" by Rule set')
plt.xlabel('Rule set')
plt.ylabel('Mean of "1"')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[154]:


combined_df.groupby('Rule set')['2'].mean()


# In[160]:


mean_values = combined_df.groupby('Rule set')['2'].mean()
# Plotting the results
plt.figure(figsize=(10, 6))
mean_values.plot(kind='bar', color='skyblue')
plt.title('Mean of Column "2" by Rule set')
plt.xlabel('Rule set')
plt.ylabel('Mean of "2"')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[155]:


combined_df.groupby('Rule set')['3'].mean()


# In[161]:


mean_values = combined_df.groupby('Rule set')['3'].mean()
# Plotting the results
plt.figure(figsize=(10, 6))
mean_values.plot(kind='bar', color='skyblue')
plt.title('Mean of Column "3" by Rule set')
plt.xlabel('Rule set')
plt.ylabel('Mean of "3"')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[156]:


combined_df.groupby('Rule set')['4'].mean()


# In[162]:


mean_values = combined_df.groupby('Rule set')['4'].mean()
# Plotting the results
plt.figure(figsize=(10, 6))
mean_values.plot(kind='bar', color='skyblue')
plt.title('Mean of Column "4" by Rule set')
plt.xlabel('Rule set')
plt.ylabel('Mean of "4"')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ### Thats mean that the highest weghits for:
# - Code Style 
# - Best Practices
# - Design
# - Error Prone
# - Documentation
# - Performance
# - Multithreading
# 
# 1. **Code Style:**
#    - Refers to the format, structure, and style of the code. This could be linked to:
#      - `Max Indentation`
#      - `Code Lines`
#      - `Comment Lines`
#      - `Preprocessor Lines`
#      - `Comment Ratio`
# 
# 2. **Best Practices:**
#    - This category typically involves standards that enhance readability and maintainability. These columns could be relevant:
#      - `Cyclomatic Complexity`
#      - `Max Indentation`
#      - `Comment Ratio`
#    
# 3. **Design:**
#    - Refers to the architecture of the code and its components. These columns may be involved in design aspects:
#      - `Cyclomatic Complexity` (measuring how complex the design is)
#      - `Total Code Length`
#      - `Code Lines`
# 
# 4. **Error Prone:**
#    - Relates to code that is likely to have defects or bugs. This could be determined by:
#      - `Cyclomatic Complexity` (higher complexity generally means higher error probability)
#      - `Magic Numbers`
#    
# 5. **Documentation:**
#    - Refers to the presence and quality of documentation. Relevant columns:
#      - `Comment Lines`
#      - `Comment Ratio`
# 
# 6. **Performance:**
#    - Performance optimization involves aspects like code efficiency, file size, etc. Columns related to performance:
#      - `File Size`
#      - `Total Code Length`
#      - `Code Lines`
#    
# 7. **Multithreading:**
#    - Multithreading involves concurrent execution, so this may not have a direct relationship with the provided columns but could relate to:
#      - `Cyclomatic Complexity` (to identify complex, multi-threaded code paths)

# 
# ### Let's assign the weight due to that!:
# - Region: 0/7
# - Type: 0/7
# - Start Line: 0/7 
# - End Line: 0/7
# - Cyclomatic Complexity: Multithreading, Error Prone, Design, Best Practices = 4/7
# - Max Indentation: Best Practices, Code Style = 2/7
# - Total Code Length: Performance, Design = 2/7 
# - Code Lines: Performance, Design, Code Style = 3/7
# - Comment Lines: Documentation, Code Style = 2/7
# - Preprocessor Lines: Code Style = 2/7
# - Magic Numbers: Error Prone = 1/7
# - Comment Ratio: Code Style, Best Practices, Documentation = 3/7
# - File Size: Performance 1/7
# 
# #### Due to that: 
# - Cyclomatic Complexity is the most important factor with 4/7
# - Code Lines, and Comment Ratio with 3/7
# - Total Code Length, Max Indentation, Comment Lines, and Preprocessor Lines with 2/7
# - Magic Numbers, and File Size with 1/7
# ## Let's create the index

# In[174]:


directory = "data/output"  
csv_files = [file for file in os.listdir(directory) if file.endswith(".csv")]
output_combined_df = pd.DataFrame()
for file_name in csv_files:
    df = pd.read_csv(f"data/output/{file_name}")
    output_combined_df = pd.concat([output_combined_df, df], ignore_index=True)
output_combined_df.head()


# In[175]:


output_combined_df.isna().sum()/len(output_combined_df) *100


# In[176]:


output_combined_df = output_combined_df.drop(columns=['modified', 'file'])


# In[177]:


new_names = {
    'file': 'File Path',
    'region': 'Region',
    'type': 'Type',
    'line start': 'Start Line',
    'line end': 'End Line',
    'std.code.complexity:cyclomatic': 'Cyclomatic Complexity',
    'std.code.complexity:maxindent': 'Max Indentation',
    'std.code.length:total': 'Total Code Length',
    'std.code.lines:code': 'Code Lines',
    'std.code.lines:comments': 'Comment Lines',
    'std.code.lines:preprocessor': 'Preprocessor Lines',
    'std.code.magic:numbers': 'Magic Numbers',
    'std.code.ratio:comments': 'Comment Ratio',
    'std.general:size': 'File Size',
}
output_combined_df = output_combined_df.rename(columns=new_names)
output_combined_df.info()


# In[178]:


output_combined_df['File Size'].fillna(-1)


# In[179]:


for col in output_combined_df.select_dtypes(include=['object']).columns:
    if output_combined_df[col].isnull().sum() > 0:
        output_combined_df[col].fillna(output_combined_df[col].mode()[0], inplace=True)


# In[180]:


for col in output_combined_df.select_dtypes(include=['float64', 'int64']).columns:
    if output_combined_df[col].isnull().sum() > 0:
        output_combined_df[col].fillna(output_combined_df[col].mean(), inplace=True)


# In[181]:


output_combined_df.isna().sum()


# In[182]:


output_combined_df.to_csv('output_data.csv')


# In[197]:


# data = output_combined_df
# os.makedirs("plots", exist_ok=True)
# for col in data.columns:
#     plt.figure(figsize=(10, 6))
#     if data[col].dtype == 'object' and len(data[col].unique()) < 50:  
#         sns.countplot(y=data[col], palette="coolwarm")
#         plt.title(f"Distribution of {col}")
#         plt.savefig(f"plots/{col}_seaborn_barplot.png")
#         plt.close()

#         fig = px.bar(data[col].value_counts().reset_index(), 
#                      x="index", y=col, 
#                      labels={"index": col, col: "Count"},
#                      title=f"Distribution of {col} (Interactive)")
#         fig.write_html(f"plots/{col}_plotly_bar.html")

#     elif data[col].dtype in ['int64', 'float64']:
#         sns.histplot(data[col], kde=True, bins=30, color="blue")
#         plt.title(f"Distribution of {col}")
#         plt.savefig(f"plots/{col}_seaborn_histogram.png")
#         plt.close()

#         fig = px.histogram(data, x=col, nbins=30, title=f"Histogram of {col} (Interactive)")
#         fig.write_html(f"plots/{col}_plotly_histogram.html")

#         sns.boxplot(x=data[col], color="orange")
#         plt.title(f"Boxplot of {col}")
#         plt.savefig(f"plots/{col}_seaborn_boxplot.png")
#         plt.close()

#         if 'Cyclomatic Complexity' in data.columns and col != 'Cyclomatic Complexity':
#             fig = px.scatter(data, x="Cyclomatic Complexity", y=col, 
#                              title=f"Scatter Plot: Cyclomatic Complexity vs {col}",
#                              labels={"Cyclomatic Complexity": "Cyclomatic Complexity", col: col})
#             fig.write_html(f"plots/Cyclomatic_Complexity_vs_{col}_scatter.html")

# plt.figure(figsize=(12, 10))
# sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
# plt.title("Correlation Heatmap")
# plt.savefig("plots/correlation_heatmap.png")
# plt.close()

# fig = px.imshow(data.corr(), text_auto=True, color_continuous_scale="Viridis",
#                 title="Correlation Heatmap (Interactive)")
# fig.write_html("plots/correlation_heatmap_interactive.html")


# In[ ]:


output_combined_df


# In[ ]:


data = output_combined_df
os.makedirs("plots", exist_ok=True)
for col in data.columns:
    plt.figure(figsize=(10, 6))
    if data[col].dtype == 'object' and len(data[col].unique()) < 50:  # Set a reasonable threshold for categorical data
        # Bar plot for categorical columns
        sns.countplot(y=data[col], palette="coolwarm")
        plt.title(f"Distribution of {col}")
        plt.savefig(f"plots/{col}_seaborn_barplot.png")
        plt.close()
        value_counts = data[col].value_counts().reset_index()
        value_counts.columns = ['index', 'count']
        fig = px.bar(value_counts, x="index", y="count", 
                     labels={"index": col, "count": "Count"},
                     title=f"Distribution of {col} (Interactive)")
        fig.write_html(f"plots/{col}_plotly_bar.html")

    elif data[col].dtype in ['int64', 'float64']:
        sns.histplot(data[col], kde=True, bins=30, color="blue")
        plt.title(f"Distribution of {col}")
        plt.savefig(f"plots/{col}_seaborn_histogram.png")
        plt.close()

        fig = px.histogram(data, x=col, nbins=30, title=f"Histogram of {col} (Interactive)")
        fig.write_html(f"plots/{col}_plotly_histogram.html")

        sns.boxplot(x=data[col], color="orange")
        plt.title(f"Boxplot of {col}")
        plt.savefig(f"plots/{col}_seaborn_boxplot.png")
        plt.close()

        if 'Cyclomatic Complexity' in data.columns and col != 'Cyclomatic Complexity':
            fig = px.scatter(data, x="Cyclomatic Complexity", y=col, 
                             title=f"Scatter Plot: Cyclomatic Complexity vs {col}",
                             labels={"Cyclomatic Complexity": "Cyclomatic Complexity", col: col})
            fig.write_html(f"plots/Cyclomatic_Complexity_vs_{col}_scatter.html")

plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("plots/correlation_heatmap.png")
plt.close()

fig = px.imshow(data.corr(numeric_only=True), text_auto=True, color_continuous_scale="Viridis",
                title="Correlation Heatmap (Interactive)")
fig.write_html("plots/correlation_heatmap_interactive.html")


# In[184]:


for col in data.columns:
    plt.figure(figsize=(10, 6))
    if data[col].dtype == 'object' and len(data[col].unique()) < 50:  # Set a reasonable threshold for categorical data
        # Bar plot for categorical columns
        sns.countplot(y=data[col], palette="coolwarm")
        plt.title(f"Distribution of {col}")
        value_counts = data[col].value_counts().reset_index()
        value_counts.columns = ['index', 'count']
        fig = px.bar(value_counts, x="index", y="count", 
                     labels={"index": col, "count": "Count"},
                     title=f"Distribution of {col} (Interactive)")
        fig.show()

    elif data[col].dtype in ['int64', 'float64']:
        sns.histplot(data[col], kde=True, bins=30, color="blue")
        plt.title(f"Distribution of {col}")
        plt.show()

        fig = px.histogram(data, x=col, nbins=30, title=f"Histogram of {col} (Interactive)")
        fig.show()

        sns.boxplot(x=data[col], color="orange")
        plt.title(f"Boxplot of {col}")
        plt.show()

        if 'Cyclomatic Complexity' in data.columns and col != 'Cyclomatic Complexity':
            fig = px.scatter(data, x="Cyclomatic Complexity", y=col, 
                             title=f"Scatter Plot: Cyclomatic Complexity vs {col}",
                             labels={"Cyclomatic Complexity": "Cyclomatic Complexity", col: col})
            fig.show()

plt.figure(figsize=(12, 10))
sns.heatmap(data.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

fig = px.imshow(data.corr(numeric_only=True), text_auto=True, color_continuous_scale="Viridis",
                title="Correlation Heatmap (Interactive)")
fig.show()


# In[185]:


output_combined_df.info()


# In[186]:


output_combined_df.describe()


# In[210]:


df = output_combined_df.copy() 
scaler = MinMaxScaler()
normalized_df = df[['Cyclomatic Complexity', 'Code Lines', 'Comment Ratio', 
                    'Total Code Length', 'Max Indentation', 'Comment Lines', 
                    'Preprocessor Lines', 'Magic Numbers', 'File Size']]

normalized_df = pd.DataFrame(scaler.fit_transform(normalized_df), columns=normalized_df.columns)
alpha, beta, gamma, = 4/7, 3/7, 1/7

normalized_df["AMI"] = (
    alpha * np.log(1 + normalized_df["Cyclomatic Complexity"]) +
    beta * np.log(1 + normalized_df["Comment Ratio"]) +
    beta * (1 - np.exp(-normalized_df["Code Lines"])) +
    gamma / (1 + normalized_df["Magic Numbers"])
)

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
metrics = ["Cyclomatic Complexity", "Code Lines", "Comment Ratio", "Magic Numbers"]
for ax, metric in zip(axes.flatten(), metrics):
    sns.scatterplot(x=normalized_df[metric], y=normalized_df["AMI"], ax=ax, alpha=0.7, color='blue')
    ax.set_title(f"AMI vs {metric}")
    ax.set_xlabel(metric)
    ax.set_ylabel("AMI")

fig.delaxes(axes[2][1])
plt.tight_layout()
plt.show()


# In[211]:


plt.figure(figsize=(10, 6))
plt.hist(normalized_df['AMI'], bins=30, color='skyblue', edgecolor='black')
plt.title("Distribution of Management Index (MI)")
plt.xlabel("MI")
plt.ylabel("Frequency")
plt.show()

mi_reshaped = normalized_df[['AMI']].values.reshape(-1, 1)
imputer = SimpleImputer(strategy='mean')
mi_reshaped = imputer.fit_transform(mi_reshaped)
kmeans = KMeans(n_clusters=3, random_state=42)
normalized_df['Cluster'] = kmeans.fit_predict(mi_reshaped)

plt.figure(figsize=(10, 6))
plt.scatter(normalized_df['AMI'], normalized_df['Cluster'], c=normalized_df['Cluster'], cmap='viridis', marker='o', alpha=0.5)
plt.title("Clusters Based on Maintaibility Index (MI)")
plt.xlabel("MI")
plt.ylabel("Cluster")
plt.show()

cluster_centers = kmeans.cluster_centers_
print(f"Cluster Centers: {cluster_centers}")

cluster_min = normalized_df.groupby('Cluster')['AMI'].min()
cluster_max = normalized_df.groupby('Cluster')['AMI'].max()
print(f"Cluster Min MI: {cluster_min}")
print(f"Cluster Max MI: {cluster_max}")


# ### **AMI Formula**
# ```python
# normalized_df["AMI"] = (
#     alpha * np.log(1 + normalized_df["Cyclomatic Complexity"]) +
#     beta * np.log(1 + normalized_df["Comment Ratio"]) +
#     beta * (1 - np.exp(-normalized_df["Code Lines"])) +
#     gamma / (1 + normalized_df["Magic Numbers"])
# )
# ```
# ### **Components Analysis**
# 
# 1. **Cyclomatic Complexity:**
#    - **Term:** `np.log(1 + normalized_df["Cyclomatic Complexity"])`
#    - **Weight (alpha):** Controls the importance of Cyclomatic Complexity in the overall index.
#    - **Analysis:**
#      - Cyclomatic Complexity grows logarithmically, which is reasonable since a very high complexity doesn't affect maintainability linearly (e.g., doubling complexity doesn't double difficulty).
#      - Adding `1` ensures that values of zero complexity don't create undefined behavior.
#      - **Impact:** Higher complexity reduces maintainability, as this term increases with higher complexity values.
# 
# 2. **Comment Ratio:**
#    - **Term:** `beta * np.sqrt(normalized_df["Comment Ratio"])`
#    - **Weight (beta):** Amplifies the importance of well-documented code.
#    - **Analysis:**
#      - Using the square root gives diminishing returns for higher comment ratios. For example, increasing comments from 5% to 10% has a larger impact than increasing from 50% to 55%.
#      - **Impact:** A higher comment ratio improves maintainability.
# 
# 3. **Code Lines:**
#    - **Term:** `beta * (1 - np.exp(-normalized_df["Code Lines"]))`
#    - **Weight (beta):** Same as for Comment Ratio.
#    - **Analysis:**
#      - The exponential function ensures that maintainability improves initially with more code lines but eventually saturates (plateaus).
#      - This makes sense because overly long codebases may start to hurt maintainability despite a structured design.
#      - **Impact:** Moderate-length codebases are preferred; both too little and too much code may lower maintainability.
# 
# 4. **Magic Numbers:**
#    - **Term:** `gamma / (1 + normalized_df["Magic Numbers"])`
#    - **Weight (gamma):** Determines how heavily the presence of magic numbers influences the score.
#    - **Analysis:**
#      - The reciprocal ensures that more magic numbers decrease maintainability significantly.
#      - Adding `1` avoids division by zero.
#      - **Impact:** Fewer magic numbers improve maintainability.

# ### **Overall Equation Analysis**
# - The equation is well-designed to capture **non-linear relationships** between software quality factors and maintainability.
# - **Strengths:**
#   - Each component addresses an essential aspect of maintainability: complexity, documentation, structure, and readability.
#   - The use of logarithmic, square root, exponential, and reciprocal transformations shows an understanding of how these factors affect maintainability in diminishing or compounding ways.
# - **Weaknesses:**
#   - **Weight Assignment (alpha, beta, gamma):** These weights need to be calibrated or justified (e.g., via expert input, sensitivity analysis, or experimental validation).
#   - **Normalization:** It assumes all input variables are normalized. Errors in normalization could skew the index.
#   - **Bias Toward Comment Ratio and Code Lines:** With `beta` appearing in two terms, Comment Ratio and Code Lines collectively have more weight than Cyclomatic Complexity or Magic Numbers unless `alpha` or `gamma` compensates.
# 

# - **Higher MI values** indicate better maintainability.
#   - The code is easier to read, modify, test, and extend.
#   - It suggests well-written, simple, and well-documented code with fewer complexities and bad practices (e.g., excessive magic numbers or long functions).
# 
# - **Lower MI values** indicate worse maintainability.
#   - The code is more complex, harder to understand, and likely to contain technical debt.
#   - It may require refactoring, better documentation, or simplifying overly complicated structures.
# 
# ---
# 
# ### ** Good MI **
# 1. **Normalized Range:**
#    - Ensure that your MI is normalized to a meaningful range (e.g., 0–100) for easier interpretation.
#    - A **higher MI** should clearly and intuitively mean better maintainability.
# 
# 2. **Thresholds:**
#    - Define thresholds to help interpret the values:
#      - **80–100:** Excellent maintainability.
#      - **50–80:** Moderate maintainability; acceptable but could be improved.
#      - **Below 50:** Poor maintainability; high technical debt.
# 
# 3. **Validation:**
#    - Validate your MI metric on real-world examples of codebases (e.g., simple scripts vs. large complex projects).
#    - Check if higher MI values align with what developers intuitively consider "well-maintained" code.
# 
# ---
# 
# **higher MI values** will naturally correspond to:
# - **Lower cyclomatic complexity:** Simpler and more modular code.
# - **Higher comment ratio:** Better documentation.
# - **Reasonable code length:** Neither too short nor excessively long.
# - **Fewer magic numbers:** More readable and maintainable.

# # Applying our MI to check our previous work repo!

# In[219]:


our_python_codes = pd.read_csv('new dashboard/code_metrics.csv')
our_python_codes.head()


# In[220]:


df = our_python_codes.copy() 
scaler = MinMaxScaler()
normalized_df = df.drop(columns=['Unnamed: 0','File', 'Cohesion (Experimental)'])

normalized_df = pd.DataFrame(scaler.fit_transform(normalized_df), columns=normalized_df.columns)
alpha, beta, gamma, = 4/7, 3/7, 1/7

normalized_df["AMI"] = (
    alpha * np.log(1 + normalized_df["Avg Cyclomatic Complexity"]) +
    beta * np.log(1 + normalized_df["Comments"]) +
    beta * (1 - np.exp(-normalized_df["LOC (Lines of Code)"])) +
    gamma / (1 + normalized_df["Magic Numbers"])
)

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
metrics = ["Avg Cyclomatic Complexity", "Comments", "LOC (Lines of Code)", "Magic Numbers"]
for ax, metric in zip(axes.flatten(), metrics):
    sns.scatterplot(x=normalized_df[metric], y=normalized_df["AMI"], ax=ax, alpha=0.7, color='blue')
    ax.set_title(f"AMI vs {metric}")
    ax.set_xlabel(metric)
    ax.set_ylabel("AMI")

fig.delaxes(axes[2][1])
plt.tight_layout()
plt.show()


# In[223]:


normalized_df.describe()


# In[ ]:


## 

