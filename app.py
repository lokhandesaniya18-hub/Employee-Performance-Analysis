
import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="INX Future - Employee Performance",
    page_icon="📊",
    layout="wide"
)


# --------------------------------------------------
# 2. FILE PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "final_employee_performance_model.pkl"
)


# --------------------------------------------------
# 3. LOAD DATA AND MODEL
# --------------------------------------------------

@st.cache_data
def load_data(path):
    return pd.read_excel(path, engine="xlrd")


@st.cache_resource
def load_model(path):
    return joblib.load(path)


st.title("📊 INX Future Inc.")
st.subheader("Employee Performance Analysis")

if not os.path.exists(DATA_PATH):
    st.error("Dataset file not found. Check the Excel filename.")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Check the .pkl filename.")
    st.stop()

try:
    df = load_data(DATA_PATH)
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading data or model: {e}")
    st.stop()


# --------------------------------------------------
# 4. DEFINE FEATURES
# --------------------------------------------------

TARGET = "PerformanceRating"

if TARGET not in df.columns:
    st.error(f"Target column '{TARGET}' was not found.")
    st.stop()

if hasattr(model, "feature_names_in_"):
    feature_columns = list(model.feature_names_in_)
else:
    feature_columns = [
        col for col in df.columns
        if col not in [TARGET, "EmpNumber"]
    ]

missing_columns = [
    col for col in feature_columns
    if col not in df.columns
]

if missing_columns:
    st.error(
        "The model's input columns do not match the dataset. "
        f"Missing columns: {missing_columns}"
    )
    st.stop()


# --------------------------------------------------
# 5. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Dashboard",
        "Employee Data",
        "Performance Prediction",
        "About Project"
    ]
)


# --------------------------------------------------
# 6. DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.header("Employee Performance Dashboard")

    total_employees = len(df)
    average_rating = df[TARGET].mean()
    departments = (
        df["EmpDepartment"].nunique()
        if "EmpDepartment" in df.columns
        else 0
    )
    rating_count = df[TARGET].nunique()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Employees", f"{total_employees:,}")
    col2.metric("Average Performance Rating", f"{average_rating:.2f}")
    col3.metric("Departments", departments)
    col4.metric("Rating Categories", rating_count)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Performance Rating Distribution")

        rating_data = (
            df[TARGET]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        rating_data.columns = ["Performance Rating", "Employees"]

        fig = px.bar(
            rating_data,
            x="Performance Rating",
            y="Employees",
            text="Employees",
            title="Employees by Performance Rating"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Department-wise Average Rating")

        if "EmpDepartment" in df.columns:
            dept_data = (
                df.groupby("EmpDepartment", as_index=False)[TARGET]
                .mean()
                .sort_values(TARGET, ascending=False)
            )

            fig = px.bar(
                dept_data,
                x="EmpDepartment",
                y=TARGET,
                text_auto=".2f",
                title="Average Rating by Department"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Department information is unavailable.")

    st.subheader("Dataset Overview")

    st.write(f"Rows: {df.shape[0]}")
    st.write(f"Columns: {df.shape[1]}")
    st.write(f"Missing values: {int(df.isnull().sum().sum())}")

    with st.expander("View dataset preview"):
        st.dataframe(df.head(10), use_container_width=True)


# --------------------------------------------------
# 7. EMPLOYEE DATA
# --------------------------------------------------

elif page == "Employee Data":

    st.header("Employee Data Explorer")

    search_text = st.text_input(
        "Search employees",
        placeholder="Enter employee number or department"
    )

    filtered_df = df.copy()

    if search_text:
        mask = filtered_df.astype(str).apply(
            lambda col: col.str.contains(
                search_text,
                case=False,
                na=False
            )
        ).any(axis=1)

        filtered_df = filtered_df[mask]

    st.write(f"Records found: {len(filtered_df)}")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        label="Download displayed data as CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="employee_performance_data.csv",
        mime="text/csv"
    )


# --------------------------------------------------
# 8. PERFORMANCE PREDICTION
# --------------------------------------------------

elif page == "Performance Prediction":

    st.header("Predict Employee Performance")

    st.write(
        "Enter employee details below to estimate the "
        "employee's performance rating."
    )

    input_values = {}

    with st.form("employee_prediction_form"):

        st.subheader("Employee Details")

        for feature in feature_columns:

            series = df[feature]
            non_null = series.dropna()

            # Categorical and text columns
            if (
                pd.api.types.is_object_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(series)
                or pd.api.types.is_string_dtype(series)
            ):

                options = sorted(
                    non_null.astype(str).unique().tolist()
                )

                if options:
                    input_values[feature] = st.selectbox(
                        feature,
                        options=options,
                        key=f"input_{feature}"
                    )
                else:
                    st.warning(
                        f"No available values for {feature}."
                    )
                    input_values[feature] = None

            # Numeric columns
            elif pd.api.types.is_numeric_dtype(series):

                if non_null.empty:
                    input_values[feature] = 0

                else:
                    minimum = float(non_null.min())
                    maximum = float(non_null.max())
                    default = float(non_null.median())

                    if pd.api.types.is_integer_dtype(series):

                        input_values[feature] = st.number_input(
                            feature,
                            min_value=int(minimum),
                            max_value=int(maximum),
                            value=int(default),
                            step=1,
                            key=f"input_{feature}"
                        )

                    else:
                        input_values[feature] = st.number_input(
                            feature,
                            min_value=minimum,
                            max_value=maximum,
                            value=default,
                            key=f"input_{feature}"
                        )

            # Fallback for other data types
            else:

                options = sorted(
                    non_null.astype(str).unique().tolist()
                )

                if options:
                    input_values[feature] = st.selectbox(
                        feature,
                        options=options,
                        key=f"input_{feature}"
                    )
                else:
                    input_values[feature] = None

        submitted = st.form_submit_button(
            "Predict Performance Rating",
            use_container_width=True
        )

    if submitted:

        if any(
            value is None
            for value in input_values.values()
        ):
            st.error(
                "Some inputs are unavailable. "
                "Please check the dataset columns."
            )

        else:
            try:
                input_df = pd.DataFrame(
                    [input_values],
                    columns=feature_columns
                )

                # Match input column types to the training data.
                for feature in feature_columns:

                    if (
                        pd.api.types.is_numeric_dtype(df[feature])
                        and not pd.api.types.is_bool_dtype(df[feature])
                    ):
                        input_df[feature] = pd.to_numeric(
                            input_df[feature],
                            errors="raise"
                        )

                    elif pd.api.types.is_bool_dtype(df[feature]):
                        input_df[feature] = input_df[feature].map(
                            lambda x: (
                                str(x).strip().lower() == "true"
                                if isinstance(x, str)
                                else bool(x)
                            )
                        )

                    elif (
                        pd.api.types.is_object_dtype(df[feature])
                        or pd.api.types.is_string_dtype(df[feature])
                        or isinstance(
                            df[feature].dtype,
                            pd.CategoricalDtype
                        )
                    ):
                        input_df[feature] = input_df[feature].astype(
                            df[feature].dtype
                        )

                prediction = model.predict(input_df)[0]

                st.success("Prediction completed!")

                st.metric(
                    "Predicted Performance Rating",
                    str(prediction)
                )

                st.info(
                    "This prediction is an analytical estimate, "
                    "not a definitive judgment of an employee. "
                    "Use it alongside human review."
                )

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(input_df)[0]

                    probability_df = pd.DataFrame({
                        "Performance Rating": model.classes_.astype(str),
                        "Probability (%)": probabilities * 100
                    }).sort_values(
                        "Probability (%)",
                        ascending=False
                    )

                    st.subheader("Prediction Probabilities")

                    fig = px.bar(
                        probability_df,
                        x="Performance Rating",
                        y="Probability (%)",
                        text=probability_df["Probability (%)"].round(2),
                        title="Estimated Probability by Rating"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

            except Exception as e:
                st.error(
                    "Prediction could not be completed. "
                    "The model may expect a different input format."
                )
                st.exception(e)


# --------------------------------------------------
# 9. ABOUT PROJECT
# --------------------------------------------------

elif page == "About Project":

    st.header("About This Project")

    st.markdown("""
    **Project:** INX Future Inc. Employee Performance Analysis

    **Objective:** Analyze employee data and estimate employee
    performance ratings using machine learning.

    **Dataset:** Employee performance dataset.

    **Target variable:** `PerformanceRating`

    **Selected model:** Tuned Decision Tree Classifier.
    """)

    st.subheader("Model Evaluation")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric("Test Accuracy", "93.75%")
    metric2.metric("Macro F1 Score", "0.8996")
    metric3.metric("Macro Recall", "0.8661")

    st.subheader("Model Hyperparameters")

    st.write("- Maximum depth: 5")
    st.write("- Minimum samples per leaf: 2")
    st.write("- Minimum samples to split: 10")

    st.subheader("Important Features")

    importance = getattr(model, "feature_importances_", None)

    if importance is not None and len(importance) == len(feature_columns):

        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": importance
        }).sort_values(
            "Importance",
            ascending=False
        ).head(10)

        fig = px.bar(
            importance_df.sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 10 Important Features"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.write(
            "Feature importance is not directly available "
            "from this saved model object."
        )

    st.caption(
        "Machine learning predictions support analysis; "
        "they should not replace human judgment."
    )


# --------------------------------------------------
# 10. FOOTER
# --------------------------------------------------

st.sidebar.divider()
st.sidebar.caption("Employee Performance Analysis | ML Project")
