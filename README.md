# Code Maintainability Index (AMI)

## Overview
The Code Maintainability Index (AMI) project aims to provide a robust method to evaluate and enhance the maintainability of source code. This project introduces a custom formula to calculate the AMI using multiple key metrics, enabling developers and teams to gain insights into the complexity, documentation quality, and overall maintainability of their code.

## Key Features
- **Cyclomatic Complexity Analysis**: Measures the complexity of code logic.
- **Comment Ratio Evaluation**: Assesses the ratio of comments to lines of code for better documentation.
- **Code Line Impact**: Evaluates how code length affects maintainability.
- **Magic Numbers**: Analyzes the presence of magic numbers (hardcoded values) and their influence on code clarity.

## Formula
The AMI is calculated using the following formula:

```python
normalized_df["AMI"] = (
    alpha * np.exp(1 + normalized_df["Cyclomatic Complexity"]) +
    beta * np.log(normalized_df["Comment Ratio"] + 1) +
    beta * (1 - np.exp(-normalized_df["Code Lines"])) +
    gamma / (1 + normalized_df["Magic Numbers"])
)
```
- **alpha, beta, gamma**: Tunable weights that adjust the influence of each metric.

## Getting Started

### Prerequisites
- Python 3.8+
- Required Libraries:
  - numpy
  - pandas
  - matplotlib (optional, for visualization)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-repo>/CodeMI.git
   cd CodeMI
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Normalize your dataset containing Cyclomatic Complexity, Comment Ratio, Code Lines, and Magic Numbers.
   ```python
   import numpy as np
   import pandas as pd

   # Example normalized DataFrame
   data = {
       "Cyclomatic Complexity": [5, 3, 8],
       "Comment Ratio": [0.4, 0.6, 0.2],
       "Code Lines": [200, 150, 300],
       "Magic Numbers": [10, 5, 0]
   }
   normalized_df = pd.DataFrame(data)
   ```

2. Calculate the AMI:
   ```python
   alpha = 0.5
   beta = 0.3
   gamma = 0.2

   normalized_df["AMI"] = (
       alpha * np.exp(1 + normalized_df["Cyclomatic Complexity"]) +
       beta * np.log(normalized_df["Comment Ratio"] + 1) +
       beta * (1 - np.exp(-normalized_df["Code Lines"])) +
       gamma / (1 + normalized_df["Magic Numbers"])
   )

   print(normalized_df)
   ```

3. Visualize results (optional):
   ```python
   import matplotlib.pyplot as plt

   plt.bar(normalized_df.index, normalized_df["AMI"], color='skyblue')
   plt.xlabel('Code Snippet Index')
   plt.ylabel('AMI')
   plt.title('Code Maintainability Index')
   plt.show()
   ```

### Example Output
| Cyclomatic Complexity | Comment Ratio | Code Lines | Magic Numbers | AMI   |
|------------------------|---------------|------------|---------------|-------|
| 5                      | 0.4           | 200        | 10            | 2.34  |
| 3                      | 0.6           | 150        | 5             | 2.76  |
| 8                      | 0.2           | 300        | 0             | 3.89  |

## Repository Structure
```
CodeMI/
├── data/               # Example datasets
├── src/                # Source code
├── notebooks/          # Jupyter notebooks for experimentation
├── README.md           # Project overview (this file)
├── requirements.txt    # Dependencies
└── LICENSE             # License information
```

## Contributing
We welcome contributions from the community! If you'd like to contribute:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed explanation of your changes.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact
For questions or feedback, feel free to reach out:
- **Abdullah Afifi** 
- **Khalid Khader**: [LinkedIn](https://www.linkedin.com/in/khalidkhader/)

