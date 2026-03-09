# Customer Segmentation with Clustering

Unsupervised learning project applying K-Means, Agglomerative Clustering, and dimensionality reduction techniques (PCA, t-SNE) to transactional retail data, identifying distinct behavioural customer groups to support targeted marketing strategies.

## Project Overview

Understanding customer behaviour is critical for retail businesses seeking to improve marketing efficiency, product development, and customer retention. This project segments customers based on transactional patterns and demographics using multiple clustering approaches, then validates and interprets the resulting segments.

## Approach

**Part I - Data Exploration and Feature Engineering**
- Data preprocessing: missing value handling, duplicate removal, outlier detection
- Feature engineering: Frequency, Recency, Customer Lifetime Value (CLV), and Average Unit Cost
- Aggregation to one-customer-per-row format for clustering
- Exploratory data analysis of feature distributions and relationships

**Part II - Clustering with ML Models**
- K-Means clustering with Elbow Method and Silhouette Score optimisation
- Agglomerative (Hierarchical) Clustering with dendrogram analysis
- Systematic comparison of k=2 and k=5 cluster solutions based on Silhouette Scores
- Multi-dimensional cluster profiling across Recency, CLV, Age, and Frequency

**Part III - Customer Segment Profiling**
- Detailed analysis of cluster characteristics across all engineered features
- Identification of high-value loyal customers, recent low-value customers, and at-risk segments
- Actionable segment descriptions for marketing strategy

**Part IV - Dimensionality Reduction and Visualisation**
- PCA (60.99% explained variance across two components)
- t-SNE for non-linear structure visualisation
- PCA loadings analysis to interpret what drives segment separation

## Key Findings

- Silhouette analysis identified k=2 and k=5 as optimal cluster counts
- Five-cluster solution revealed distinct segments: high-value loyal (older), moderate-value engaged, low-value new/young customers, at-risk inactive, and transitional groups
- PCA Component 1 driven by frequency and CLV; Component 2 driven by average unit cost and inversely by age
- Older customers correlate with higher CLV; younger customers tend toward lower engagement

## Tech Stack

- **Python** - Pandas, NumPy, Matplotlib, Seaborn
- **Clustering** - Scikit-learn (K-Means, Agglomerative Clustering)
- **Dimensionality Reduction** - PCA, t-SNE
- **Anomaly Detection** - Isolation Forest (outlier removal)
- **Preprocessing** - StandardScaler, SimpleImputer

## Repository Structure

```
customer-segmentation/
|-- customer_segmentation.ipynb   # Main analysis notebook
|-- requirements.txt              # Python dependencies
|-- README.md                     # This file
```

## Getting Started

```bash
pip install -r requirements.txt
jupyter notebook customer_segmentation.ipynb
```

## Dataset

Retail transaction data containing customer purchase records with product details, quantities, unit costs, and customer demographics. Data was sourced from the University of Cambridge Data Science programme.

## Author

**Raquel Jones** - Data Scientist and Analytics Engineer

- Portfolio: [rjdatavoyage.co.uk](https://rjdatavoyage.co.uk)
- LinkedIn: [Raquel Jones](https://linkedin.com/in/664113153)
