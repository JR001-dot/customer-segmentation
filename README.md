# Customer Segmentation with Clustering

K-Means and agglomerative clustering on retail customer data, with PCA and t-SNE for visualisation. Published as a learning exercise with its limitations stated.

**Origin:** University of Cambridge Data Science Career Accelerator coursework (2024-25). **Data:** source anonymised and not included in this repository. Notebook outputs are cleared to protect the source data; the figures below come from the original run.

## What was done
- 951,669 order lines aggregated to 68,300 customers (47 countries, 2012-2016).
- Engineered features: order frequency, recency, total profit (labelled CLV in the notebook), average unit cost and age.
- Isolation Forest used to remove 3,415 outlying customers (5%).
- Elbow and silhouette analysis, K-Means and agglomerative (hierarchical) clustering, PCA and t-SNE plots.

## Results
- Elbow plot suggested k = 5; silhouette score was highest at **k = 2 (0.266)**, with k = 5 at 0.255.
- Final K-Means (k = 5) cluster sizes: 905, 5,503, 43,161, 1,911 and 13,405 customers.
- First two principal components explain 61.0% of the variance.

## Known limitations
- **Weak separation:** silhouette scores of about 0.25 and one cluster holding about two-thirds of customers mean the segments are not well separated.
- **CLV calculation bug:** per-customer profit was merged onto every order line and then summed, inflating "CLV" by the number of lines per customer.
- **Scaling:** the final K-Means ran on unscaled features, so large-valued features dominate the distances.
- Segment names are therefore not assigned; fixing the two issues above and re-running is the next step.

## Tech stack
Python, pandas, NumPy, scikit-learn (K-Means, AgglomerativeClustering, Isolation Forest, PCA, t-SNE), SciPy, Matplotlib, Seaborn

## Repository
- `customer_segmentation.ipynb` - full analysis
- `requirements.txt`

---

**Author:** Raquel J. | [rjdatavoyage.co.uk](https://rjdatavoyage.co.uk)
