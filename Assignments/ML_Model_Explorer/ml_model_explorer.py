import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings
warnings.filterwarnings("ignore")

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Set page configuration
st.set_page_config(
    page_title="Machine Learning Model Explorer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title
st.title("🤖 Machine Learning Model Explorer")
st.markdown("---")

# ========================
# SIDEBAR CONTROLS
# ========================
st.sidebar.title("⚙️ Configuration")

# Data source selection
st.sidebar.subheader("📁 Data Source")
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Upload CSV", "Use Sample Dataset"],
    horizontal=True
)

uploaded_file = None
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help="Upload a CSV file for analysis"
    )
else:
    st.sidebar.info("Using built-in sample datasets")

# Sample dataset selection
sample_datasets = {
    "Iris (Classification)": "iris",
    "Wine (Classification)": "wine",
    "Diabetes (Regression)": "diabetes",
    "Breast Cancer": "breast_cancer",
    "Digits (Classification)": "digits"
}

if data_source == "Use Sample Dataset":
    selected_sample = st.sidebar.selectbox(
        "Select sample dataset:",
        list(sample_datasets.keys())
    )

# Learning type selection
st.sidebar.subheader("🎯 Learning Type")
learning_type = st.sidebar.radio(
    "Select learning type:",
    ["Supervised", "Unsupervised"],
    horizontal=True
)

# ========================
# DATA LOADING AND PREVIEW
# ========================
@st.cache_data
def load_sample_dataset(dataset_name):
    """Load sample datasets from scikit-learn"""
    from sklearn import datasets
    
    dataset_map = {
        "iris": datasets.load_iris(),
        "wine": datasets.load_wine(),
        "diabetes": datasets.load_diabetes(),
        "breast_cancer": datasets.load_breast_cancer(),
        "digits": datasets.load_digits()
    }
    
    data = dataset_map[dataset_name]
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

# Load data
df = None
if data_source == "Upload CSV" and uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ CSV loaded successfully! Shape: {df.shape}")
    except Exception as e:
        st.error(f"❌ Error reading CSV: {e}")
        st.stop()
elif data_source == "Use Sample Dataset":
    dataset_key = sample_datasets[selected_sample]
    df = load_sample_dataset(dataset_key)
    st.info(f"📊 Using {selected_sample} dataset. Shape: {df.shape}")
else:
    st.warning("⚠️ Please upload a CSV file or select a sample dataset.")
    st.stop()

# Display dataset info
st.header("📊 Dataset Overview")
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Rows", df.shape[0])
    st.metric("Columns", df.shape[1])

with col2:
    st.metric("Numeric Columns", len(df.select_dtypes(include=[np.number]).columns))
    st.metric("Categorical Columns", len(df.select_dtypes(include=['object']).columns))

with col3:
    missing_values = df.isnull().sum().sum()
    st.metric("Missing Values", missing_values)
    if missing_values > 0:
        st.warning(f"Found {missing_values} missing values")

# Dataset preview
st.subheader("Dataset Preview")
with st.expander("View Data (First 10 rows)"):
    st.dataframe(df.head(10), use_container_width=True)

# Show column information
with st.expander("Column Information"):
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Unique Values': [df[col].nunique() for col in df.columns],
        'Missing Values': df.isnull().sum().values
    })
    st.dataframe(col_info, use_container_width=True)

# ========================
# DATA PREPROCESSING
# ========================
st.header("🔧 Data Preprocessing")

# Select target column (for supervised learning)
if learning_type == "Supervised":
    st.subheader("Target Selection")
    all_columns = df.columns.tolist()
    
    # Try to automatically detect target column
    target_candidates = []
    for col in all_columns:
        if col.lower() in ['target', 'label', 'class', 'y', 'result', 'output']:
            target_candidates.append(col)
    
    if target_candidates:
        default_target = target_candidates[0]
    else:
        # For regression, try to find numeric columns with reasonable distribution
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        default_target = numeric_cols[-1] if numeric_cols else all_columns[-1]
    
    target_col = st.selectbox(
        "Select target column:",
        all_columns,
        index=all_columns.index(default_target) if default_target in all_columns else 0
    )
    
    # Feature selection
    feature_cols = [col for col in all_columns if col != target_col]
    selected_features = st.multiselect(
        "Select features for modeling:",
        feature_cols,
        default=feature_cols[:min(10, len(feature_cols))]
    )
    
    if not selected_features:
        st.error("⚠️ Please select at least one feature column.")
        st.stop()
    
    X = df[selected_features].copy()
    y = df[target_col].copy()
    
    # Display target distribution
    st.subheader("Target Distribution")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    if y.dtype == 'object' or y.nunique() < 10:
        # Classification problem
        value_counts = y.value_counts()
        ax1.bar(value_counts.index.astype(str), value_counts.values)
        ax1.set_title("Class Distribution")
        ax1.set_xlabel("Class")
        ax1.set_ylabel("Count")
        
        ax2.pie(value_counts.values, labels=value_counts.index.astype(str), autopct='%1.1f%%')
        ax2.set_title("Class Proportions")
        
        problem_type = "classification"
    else:
        # Regression problem
        ax1.hist(y, bins=30, edgecolor='black')
        ax1.set_title("Target Value Distribution")
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Frequency")
        
        ax2.boxplot(y)
        ax2.set_title("Target Value Box Plot")
        
        problem_type = "regression"
    
    st.pyplot(fig)
    
else:
    # Unsupervised learning - select all features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.multiselect(
        "Select features for clustering:",
        numeric_cols,
        default=numeric_cols[:min(10, len(numeric_cols))]
    )
    
    if not selected_features:
        st.error("⚠️ Please select at least one numeric feature column.")
        st.stop()
    
    X = df[selected_features].copy()
    y = None
    problem_type = "clustering"

# Preprocessing configuration
st.subheader("Preprocessing Configuration")
col1, col2, col3 = st.columns(3)

with col1:
    missing_strategy = st.selectbox(
        "Missing value strategy:",
        ["mean", "median", "most_frequent", "constant"]
    )

with col2:
    categorical_encode = st.selectbox(
        "Categorical encoding:",
        ["One-Hot Encoding", "Label Encoding"]
    )

with col3:
    scale_features = st.checkbox("Scale features", value=True)
    scale_method = st.selectbox(
        "Scaling method:",
        ["StandardScaler", "MinMaxScaler"]
    )

# Identify column types
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

st.info(f"📊 Identified {len(numeric_features)} numeric and {len(categorical_features)} categorical features")

# Create preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy=missing_strategy)),
    ('scaler', StandardScaler() if scale_method == "StandardScaler" else MinMaxScaler()) if scale_features else ('scaler', 'passthrough')
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore') if categorical_encode == "One-Hot Encoding" else LabelEncoder())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

st.success(f"✅ Preprocessing complete! Transformed shape: {X_processed.shape}")

# ========================
# SUPERVISED LEARNING
# ========================
if learning_type == "Supervised":
    st.header("🎯 Supervised Learning")
    
    # Train-test split
    test_size = st.slider("Test set size (%)", 10, 40, 20, 5)
    random_state = st.number_input("Random seed", value=42, min_value=0, max_value=1000)
    
    # Model selection
    st.subheader("Model Selection")
    
    if problem_type == "classification":
        model_choice = st.selectbox(
            "Select classifier:",
            ["Random Forest", "Decision Tree", "Support Vector Machine", "Logistic Regression"]
        )
        
        # Model hyperparameters
        if model_choice == "Random Forest":
            n_estimators = st.slider("Number of trees", 10, 200, 100, 10)
            max_depth = st.slider("Max depth", 1, 20, 10)
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
            
        elif model_choice == "Decision Tree":
            max_depth = st.slider("Max depth", 1, 20, 10)
            min_samples_split = st.slider("Min samples split", 2, 20, 2)
            model = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state
            )
            
        elif model_choice == "Support Vector Machine":
            C = st.slider("C (Regularization)", 0.01, 10.0, 1.0, 0.1)
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
            model = SVC(C=C, kernel=kernel, random_state=random_state, probability=True)
            
        else:  # Logistic Regression
            C = st.slider("C (Inverse regularization)", 0.01, 10.0, 1.0, 0.1)
            model = LogisticRegression(C=C, random_state=random_state, max_iter=1000)
            
    else:  # Regression
        model_choice = st.selectbox(
            "Select regressor:",
            ["Random Forest", "Decision Tree", "Support Vector Machine", "Linear Regression"]
        )
        
        # Model hyperparameters for regression
        if model_choice == "Random Forest":
            n_estimators = st.slider("Number of trees", 10, 200, 100, 10)
            max_depth = st.slider("Max depth", 1, 20, 10)
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )
            
        elif model_choice == "Decision Tree":
            max_depth = st.slider("Max depth", 1, 20, 10)
            min_samples_split = st.slider("Min samples split", 2, 20, 2)
            model = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=random_state
            )
            
        elif model_choice == "Support Vector Machine":
            C = st.slider("C (Regularization)", 0.01, 10.0, 1.0, 0.1)
            kernel = st.selectbox("Kernel", ["linear", "rbf", "poly"])
            model = SVR(C=C, kernel=kernel)
            
        else:  # Linear Regression
            model = LinearRegression()
    
    # Train the model
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, 
                test_size=test_size/100, 
                random_state=random_state,
                stratify=y if problem_type == "classification" and y.nunique() < 10 else None
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Display results
            st.subheader("📈 Model Performance")
            
            col1, col2, col3 = st.columns(3)
            
            if problem_type == "classification":
                with col1:
                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
                with col2:
                    st.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
                with col3:
                    st.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted', zero_division=0):.3f}")
                
                # Classification report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
                
                # Confusion matrix
                st.subheader("Confusion Matrix")
                fig, ax = plt.subplots(figsize=(8, 6))
                cm = confusion_matrix(y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)
                
                # ROC Curve (for binary classification)
                if len(np.unique(y)) == 2:
                    try:
                        y_proba = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        roc_auc = roc_auc_score(y_test, y_proba)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
                        ax.plot([0, 1], [0, 1], 'k--', label='Random')
                        ax.set_xlabel('False Positive Rate')
                        ax.set_ylabel('True Positive Rate')
                        ax.set_title('ROC Curve')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                    except:
                        pass
                
            else:  # Regression metrics
                with col1:
                    st.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
                with col2:
                    st.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.3f}")
                with col3:
                    st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
                
                # Scatter plot of predictions vs actual
                st.subheader("Predictions vs Actual")
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test, y_pred, alpha=0.6)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax.set_xlabel('Actual Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title('Predictions vs Actual')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Residual plot
                st.subheader("Residual Plot")
                residuals = y_test - y_pred
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_pred, residuals, alpha=0.6)
                ax.axhline(y=0, color='r', linestyle='--')
                ax.set_xlabel('Predicted Values')
                ax.set_ylabel('Residuals')
                ax.set_title('Residual Plot')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance")
                try:
                    feature_names = []
                    if hasattr(preprocessor, 'get_feature_names_out'):
                        feature_names = preprocessor.get_feature_names_out()
                    else:
                        feature_names = [f'Feature {i}' for i in range(X_processed.shape[1])]
                    
                    importances = model.feature_importances_
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False).head(20)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(importance_df['Feature'], importance_df['Importance'])
                    ax.set_xlabel('Importance')
                    ax.set_title('Top 20 Feature Importances')
                    st.pyplot(fig)
                except:
                    pass

# ========================
# UNSUPERVISED LEARNING
# ========================
else:
    st.header("🔍 Unsupervised Learning")
    
    # Clustering algorithm selection
    st.subheader("Clustering Algorithm")
    cluster_algo = st.selectbox(
        "Select clustering algorithm:",
        ["K-Means", "Agglomerative Clustering", "DBSCAN"]
    )
    
    # Algorithm-specific parameters
    if cluster_algo == "K-Means":
        n_clusters = st.slider("Number of clusters", 2, 10, 3)
        random_state = st.number_input("Random seed", value=42, min_value=0, max_value=1000)
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        
    elif cluster_algo == "Agglomerative Clustering":
        n_clusters = st.slider("Number of clusters", 2, 10, 3)
        linkage = st.selectbox("Linkage method", ["ward", "complete", "average", "single"])
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        
    else:  # DBSCAN
        eps = st.slider("Epsilon (eps)", 0.1, 5.0, 0.5, 0.1)
        min_samples = st.slider("Min samples", 1, 20, 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
    
    # Dimensionality reduction for visualization
    st.subheader("Visualization Settings")
    dr_method = st.radio(
        "Dimensionality reduction method:",
        ["PCA", "t-SNE"],
        horizontal=True
    )
    
    if st.button("🚀 Run Clustering", type="primary"):
        with st.spinner("Running clustering algorithm..."):
            # Apply clustering
            labels = model.fit_predict(X_processed)
            
            # Calculate metrics
            st.subheader("📊 Clustering Results")
            
            col1, col2, col3 = st.columns(3)
            
            n_clusters_found = len(np.unique(labels[labels != -1])) if -1 in labels else len(np.unique(labels))
            
            with col1:
                st.metric("Number of Clusters Found", n_clusters_found)
            
            if n_clusters_found > 1:
                try:
                    silhouette = silhouette_score(X_processed, labels)
                    with col2:
                        st.metric("Silhouette Score", f"{silhouette:.3f}")
                    
                    davies_bouldin = davies_bouldin_score(X_processed, labels)
                    with col3:
                        st.metric("Davies-Bouldin Index", f"{davies_bouldin:.3f}")
                    
                    # Interpretation
                    st.info(f"""
                    **Silhouette Score Interpretation:**
                    - > 0.7: Strong structure found
                    - 0.5 - 0.7: Reasonable structure
                    - 0.25 - 0.5: Weak structure
                    - < 0.25: No substantial structure
                    
                    **Current score: {silhouette:.3f}**
                    """)
                except:
                    pass
            
            # Dimensionality reduction for visualization
            if dr_method == "PCA":
                reducer = PCA(n_components=2, random_state=42)
            else:
                reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            
            X_reduced = reducer.fit_transform(X_processed)
            
            # Visualization
            st.subheader("Cluster Visualization")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Scatter plot
            scatter = ax1.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                 c=labels, cmap='tab20', alpha=0.7, s=50)
            ax1.set_xlabel(f'{dr_method} Component 1')
            ax1.set_ylabel(f'{dr_method} Component 2')
            ax1.set_title(f'{cluster_algo} Clustering ({dr_method} Projection)')
            ax1.grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax1, label='Cluster')
            
            # Cluster size distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            ax2.bar(unique_labels.astype(str), counts)
            ax2.set_xlabel('Cluster')
            ax2.set_ylabel('Number of Points')
            ax2.set_title('Cluster Size Distribution')
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add count labels on bars
            for i, count in enumerate(counts):
                ax2.text(i, count + max(counts)*0.01, str(count), 
                        ha='center', va='bottom')
            
            st.pyplot(fig)
            
            # Cluster statistics
            st.subheader("Cluster Statistics")
            
            # Create a DataFrame with original features and cluster labels
            cluster_df = pd.DataFrame(X_processed, columns=[f'Feature_{i}' for i in range(X_processed.shape[1])])
            cluster_df['Cluster'] = labels
            
            # Calculate mean values for each cluster
            cluster_stats = cluster_df.groupby('Cluster').mean()
            st.dataframe(cluster_stats.style.background_gradient(cmap='Blues'), use_container_width=True)
            
            # Download clusters
            st.subheader("Export Results")
            result_df = df.copy()
            result_df['Cluster'] = labels
            
            csv = result_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="clustering_results.csv">📥 Download Clustering Results</a>'
            st.markdown(href, unsafe_allow_html=True)

# ========================
# FOOTER
# ========================
st.markdown("---")
st.markdown("### 📚 About This App")
st.info("""
This Machine Learning Model Explorer allows you to:
1. **Upload any dataset** in CSV format or use built-in sample datasets
2. **Automatically preprocess** data (handle missing values, encode categorical features, scale numeric features)
3. **Train supervised models** (Classification: Random Forest, Decision Tree, SVM, Logistic Regression | Regression: Random Forest, Decision Tree, SVM, Linear Regression)
4. **Apply unsupervised clustering** (K-Means, Agglomerative Clustering, DBSCAN)
5. **Visualize results** with interactive charts and metrics

**Tips:**
- For classification, ensure your target column has discrete values
- For regression, ensure your target column has continuous values
- Select relevant features for better model performance
- Experiment with different preprocessing options
""")

st.success("✅ Analysis complete! Adjust parameters in the sidebar and rerun for different results.")