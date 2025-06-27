#  Credit Risk Modeling for BNPL (Buy Now Pay Later)

This repository contains an end-to-end machine learning project to develop a **Credit Risk Scoring Model** for Bati Bank in partnership with an eCommerce platform. The model leverages customer transactional behavior to predict the likelihood of default for BNPL services. The solution incorporates modern ML practices, regulatory compliance (Basel II), and model deployment through FastAPI.

---

##  Project Structure

```
credit-risk-model/
├── .github/workflows/ci.yml         # CI/CD pipeline
├── data/
│   ├── raw/                         # Raw data
│   └── processed/                   # Cleaned/engineered data
├── notebooks/
│   └── 1.0-eda.ipynb                # Exploratory Data Analysis
├── src/
│   ├── data_processing.py           # Feature engineering pipeline
│   ├── train.py                     # Model training script
│   ├── predict.py                   # Inference script
│   └── api/
│       ├── main.py                  # FastAPI app
│       └── pydantic_models.py       # Request/response schemas
├── tests/
│   └── test_data_processing.py      # Unit tests
├── Dockerfile                       # Docker container
├── docker-compose.yml               # Docker setup
├── requirements.txt                 # Project dependencies
├── .gitignore
└── README.md                        # Project documentation
```

---

##  Credit Scoring Business Understanding

###  Introduction

Credit scoring plays a crucial role in financial services by helping institutions like banks assess the creditworthiness of potential borrowers. With the rise of alternative data sources—especially in digital commerce—the need for robust, accurate, and interpretable models has become more important than ever. This project focuses on developing a credit risk model for Bati Bank in collaboration with an eCommerce platform to enable a Buy-Now-Pay-Later (BNPL) service.

Given the regulatory framework (notably the Basel II Accord), the absence of explicit default labels, and the tension between model interpretability and performance, a careful balance of data science and domain expertise is necessary.

---

### 1 Basel II Accord and the Need for Interpretability

The **Basel II Capital Accord** defines how much capital financial institutions must hold to manage risks, including credit risk. Its **Internal Ratings-Based (IRB)** approach permits the use of internal models to estimate risk parameters such as:

* **Probability of Default (PD)**
* **Loss Given Default (LGD)**
* **Exposure at Default (EAD)**

To comply, institutions must:

* Use **data-driven models** built on historical performance.
* Ensure models are **transparent and explainable**.
* Maintain **auditable records** for regulatory review.

Implications:

* **Interpretable models** like Logistic Regression with WoE are ideal for production.
* **Opaque models** may lead to non-compliance if not justified with explainability techniques.
* **Model documentation and version control** are mandatory for governance.

---

### 2 Proxy Target Variable: Why and What Risks?

Since the dataset lacks a direct "default" label, a **proxy variable** is engineered using behavioral patterns:

* RFM (Recency, Frequency, Monetary) metrics
* K-Means clustering to define risk segments
* The least engaged cluster is labeled `is_high_risk = 1`

**Risks** of using proxy labels:

*  **Misclassification**: Low activity ≠ credit default
*  **Bias**: Proxies may reflect socioeconomic/geographic biases
*  **Revenue Loss or Default Risk**: Incorrect labels lead to poor lending decisions
*  **Validation Challenge**: Hard to validate without true default history

Therefore, this approach must include:

* Post-deployment feedback loops
* Fairness and bias checks
* Ongoing label reevaluation

---

### 3 Model Trade-offs: Simplicity vs. Complexity

####  Simple Models (e.g., Logistic Regression with WoE)

** Pros**:

* Interpretability & transparency
* Faster to train and easier to deploy
* Regulatory compliance
* Ideal for credit policy alignment

** Cons**:

* Limited in modeling nonlinear interactions
* May underperform in complex datasets

####  Complex Models (e.g., Gradient Boosting Machines)

** Pros**:

* Superior predictive accuracy
* Captures non-linear patterns and interactions

** Cons**:

* Harder to interpret without SHAP/LIME
* Requires more governance and validation
* Risk of overfitting without careful tuning

 **Best Practice**: Use simple models for deployment, complex models for internal benchmarking or ensemble blending.

---

###  Conclusion

To build a credit risk model that is fair, reliable, and compliant:

* Adhere to **Basel II regulatory guidance**
* Develop and validate **proxy targets** with care
* Balance **model interpretability and performance**

This project provides a foundation for delivering ethical, explainable, and actionable credit scores based on alternative data.

---

##  Project Phases

| Phase                           | Description                                   |
| ------------------------------- | --------------------------------------------- |
| Task 1 - Business Understanding | Read references, build domain understanding   |
| Task 2 - EDA                    | Visualize and summarize data patterns         |
| Task 3 - Feature Engineering    | Create predictive and interpretable features  |
| Task 4 - Target Engineering     | Label high-risk users via RFM clustering      |
| Task 5 - Modeling               | Train, tune, and evaluate ML models           |
| Task 6 - Deployment             | Serve via FastAPI, enable CI/CD, containerize |

---

##  Tools & Libraries

* Python 3.10+
* scikit-learn
* pandas, numpy
* xverse, woe
* matplotlib, seaborn
* FastAPI, uvicorn
* MLFlow for model tracking
* Docker & GitHub Actions

---

##  Testing & Quality Assurance

* Unit tests for preprocessing and model logic
* Linting using flake8
* CI/CD workflow with GitHub Actions


##  References

* [Basel II Accord](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n535.pdf)
* [World Bank Credit Scoring Guidelines](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)
* [HKMA Alternative Credit Scoring](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
* [Toward Data Science Guide](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)
* [CFI on Credit Risk](https://corporatefinanceinstitute.com/resources/commercial-lending/credit-risk/)
* [Risk Officer Guide](https://www.risk-officer.com/Credit_Risk.htm)

---

##  Authors

* Henok Yoseph 


---

##  Contact

For inquiries or suggestions ,please email: henokapril@gmail.com 
github : https://github.com/aprilyab/credit-risk-model 

---
