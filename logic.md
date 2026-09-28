# Logic of the Strategy

## 1. Data

### 1.1 What data do we download?

We download the daily U.S. Treasury par yield curve from the official Treasury website:

<https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView?field_tdr_date_value=2025&type=daily_treasury_yield_curve>

The dataset contains daily Treasury yields at fixed maturities. Our first version uses 2Y, 5Y, 10Y, and 30Y yields.

### 1.2 What is a Treasury yield?

The underlying securities are U.S. Treasury securities already issued and traded in the secondary market:

- Treasury Bills: maturities of one year or less;
- Treasury Notes: 2Y, 3Y, 5Y, 7Y, and 10Y securities;
- Treasury Bonds: 20Y and 30Y securities.

The Treasury constructs the par yield curve approximately as follows:

```text
Treasury securities quoted in the secondary market
        ↓
Obtain representative bid-side quotations
        ↓
Calculate yields from bond prices
        ↓
Fit the Treasury par yield curve
        ↓
Read fixed maturity points such as 1M, 2Y, 5Y, 10Y, and 30Y
```

### 1.3 How can the yields be checked?

| Maturity | TradingView symbol |
|---|---|
| 2Y yield | `TVC:US02Y` |
| 5Y yield | `TVC:US05Y` |
| 10Y yield | `TVC:US10Y` |
| 30Y yield | `TVC:US30Y` |

10Y yield: <https://www.tradingview.com/symbols/TVC-US10Y/>

### 1.4 How do we use the data and trade the strategy?

Treasury yields and Treasury futures prices normally move in opposite directions:

```text
10Y yield rises
        ↓
10Y Treasury bond price falls
        ↓
ZN futures price normally falls
```

The 10Y Treasury futures continuous contract can be checked with `CBOT:ZN1!`:

<https://www.tradingview.com/symbols/CBOT-ZN1%21/>

The two datasets have different jobs:

```text
Treasury yield data
→ describe and forecast changes in the yield curve

Treasury futures data
→ convert the yield-curve forecast into tradable positions
```

## 2. PCA Validation Notes

### 2.1 Input data and centering

We apply PCA to the **daily yield changes** of the 2Y, 5Y, 10Y, and 30Y maturities, measured in basis points:

$$
X_t=[\Delta y_{2Y,t},\Delta y_{5Y,t},\Delta y_{10Y,t},\Delta y_{30Y,t}]
$$

PCA first calculates the training-period mean $\mu$ of the four columns and centers each daily observation:

$$
X_t^{centered}=X_t-\mu
$$

Centering makes each column fluctuate around zero. We do not divide by its standard deviation, so this is **covariance PCA** and all maturities retain the same bp scale.

### 2.2 How are the loadings obtained?

PCA calculates the covariance matrix of the centered data:

$$
\Sigma=\operatorname{Cov}(X^{centered})
$$

It then performs an eigenvalue decomposition:

$$
\Sigma l_k=\lambda_k l_k
$$

- The eigenvalues $\lambda_k$, ordered from largest to smallest, correspond to PC1, PC2, PC3, and PC4.
- The eigenvector $l_k$ is the loading of component $k$ and defines its yield-curve shape.
- The four numbers in each loading are the weights of the 2Y, 5Y, 10Y, and 30Y changes.

| Component | Loading shape | Interpretation |
|---|---|---|
| PC1 | `[+,+,+,+]` | Level: the whole curve moves together |
| PC2 | `[+,+,-,-]` | Slope: the short and long ends move relative to each other |
| PC3 | `[-,+,+,-]` | Curvature: the middle moves relative to the two wings |

Multiplying an entire loading by `-1` gives the same PCA axis. The code therefore applies consistent sign conventions: positive PC1 means a broad rise in yields, positive PC2 means the short end rises relative to the long end, and positive PC3 means the middle rises relative to the wings.

![PCA loadings](results/pca/pca_loadings.png)

### 2.3 How is a score calculated?

A loading defines the direction of a PCA axis. A score is one day's projection onto that axis:

$$
score_{t,k}=(X_t-\mu)\cdot l_k
$$

For example, suppose the PC1 loading is approximately:

$$
l_1=[0.39,0.56,0.55,0.49]
$$

If one day's centered yield changes are `[5,6,6,5]` bp, then:

$$
PC1\ score\approx5(0.39)+6(0.56)+6(0.55)+5(0.49)=11.06
$$

This means that the curve moved mainly in the positive PC1 direction—a broad increase in yields—with an intensity of about 11.06. For the full dataset, the matrix form is:

$$
F=(X-\mu)L^\top
$$

`pca.transform(X)` performs the centering and projection onto the loadings. It does not refit the PCA model.

### 2.4 `fit()`, `transform()`, and `inverse_transform()`

```python
pca.fit(X_train)
scores = pca.transform(X)
reconstructed = pca.inverse_transform(scores)
```

- `fit()` learns the training mean `mean_`, the loadings `components_`, and the explained variances.
- `transform()` uses the fitted mean and loadings to convert maturity-level yield changes into PC scores.
- `inverse_transform()` uses the scores and loadings to reconstruct the maturity-level yield changes.

The reconstruction formula is:

$$
\widehat X=FL+\mu
$$

The mean $\mu$ must be added back because it was removed before the projection. To test a reduced representation with the first $k$ PCs, set the remaining scores to zero before applying `inverse_transform()`.

### 2.5 Validation results

- PC1, PC2, and PC3 explain `87.13%`, `10.32%`, and `1.94%`, respectively.
- The first three PCs explain `99.39%` cumulatively.
- The three-PC reconstruction has `MAE = 0.2661 bp` and `RMSE = 0.4030 bp`.
- The loading shapes are broadly stable across historical periods.
- PCA validates an efficient representation of curve changes; it does not prove that the PCs are predictable or tradable.

![Explained variance](results/pca/explained_variance.png)

![Reconstruction error](results/pca/reconstruction_error.png)

![Loading stability](results/pca/loading_stability.png)

### 2.6 Related files

- PCA implementation: [`src/factors/pca_factors.py`](src/factors/pca_factors.py)
- PCA execution script: [`scripts/run_pca.py`](scripts/run_pca.py)
- Results notebook: [`notebook/01_pca_results.ipynb`](notebook/01_pca_results.ipynb)
- Numerical outputs: `data/processed/pca/`

The PCA validation notebook uses a fixed coordinate system fitted on 2006–2019 data. The production forecasting pipeline uses monthly expanding PCA fits, with each fit restricted to information available at that time.
