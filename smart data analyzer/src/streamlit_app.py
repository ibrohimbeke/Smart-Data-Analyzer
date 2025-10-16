import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import openai
import os
import io

# --- START: Streamlit Configuration Overrides ---
# These global settings must be set via config.toml or command line arguments in the Dockerfile.
# They cannot be set programmatically using st.set_option in app.py.
# All problematic st.set_option calls have been removed from here.
# --- END: Streamlit Configuration Overrides ---


# Set OpenAI API key
# Ensure you have your OpenAI API key set as an environment variable (OPENAI_API_KEY)
# For local development, you might set it directly like: openai.api_key = "YOUR_API_KEY"
# But for deployment, environment variables are recommended.
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Smart Data Analyzer", layout="wide")

# Load data from uploaded CSV file
@st.cache_data # Cache the data loading to improve performance
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            st.info(f"Attempting to load file: {uploaded_file.name}")
            st.info(f"File type: {type(uploaded_file)}")
            st.info(f"File size (bytes): {uploaded_file.getbuffer().nbytes if hasattr(uploaded_file, 'getbuffer') else 'N/A'}")

            # Let pandas directly read the uploaded file object.
            # Streamlit's uploaded_file object is a BytesIO-like object that pandas can often read directly.
            df = pd.read_csv(uploaded_file)
            st.success("File loaded successfully into DataFrame.")

            # Explicitly convert all 'object' type columns to string immediately after loading
            # This prevents pyarrow from attempting to infer numeric types for string columns.
            for col in df.select_dtypes(include='object').columns:
                df[col] = df[col].astype(str)
            return df
        except UnicodeDecodeError as ude:
            st.error(f"Encoding Error: {ude}. "
                     f"The CSV file might not be in UTF-8 format. "
                     f"Try saving your CSV with 'UTF-8' encoding or specify the 'encoding' parameter in pd.read_csv.")
            return None
        except Exception as e:
            # Provide a more detailed error message to help diagnose the issue
            st.error(f"Error loading file: {e}. "
                     f"Please ensure it's a valid CSV, correctly formatted, and check its encoding. "
                     f"If the issue persists, inspect Hugging Face Space logs for a full traceback.")
            return None
    return None

# Synchronous AI insight function using OpenAI API
# This function sends a prompt to the GPT-4 model and returns its response.
def get_ai_insights_sync(prompt):
    try:
        client = openai.OpenAI() # Initialize OpenAI client
        response = client.chat.completions.create(
            model="gpt-4",  # Using GPT-4 for more comprehensive insights
            messages=[
                {"role": "system", "content": "You are a helpful data analyst assistant. Provide concise and actionable insights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7, # Controls randomness of the response
            max_tokens=500 # Increased max_tokens for potentially longer insights
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error getting AI insights: {e}. Please check your OpenAI API key and internet connection."

# Suggest visualizations based on column types and combinations
def suggest_visualizations(df):
    suggestions = []
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    # Single column visualizations
    for col in numeric_cols:
        suggestions.append({"type": "Histogram", "x": col, "y": "count", "description": f"Distribution of '{col}'"})
        suggestions.append({"type": "Box Plot", "x": None, "y": col, "description": f"Box plot of '{col}' (outliers & spread)"})
        suggestions.append({"type": "Violin Plot", "x": None, "y": col, "description": f"Violin plot of '{col}' (distribution shape)"})

    for col in categorical_cols:
        suggestions.append({"type": "Bar Chart", "x": col, "y": "count", "description": f"Frequency of '{col}'"})
        suggestions.append({"type": "Pie Chart", "names": col, "values": "count", "description": f"Proportion of categories in '{col}'"})

    # Two-column visualizations
    # Numeric vs Numeric
    for i in range(len(numeric_cols)):
        for j in range(i + 1, len(numeric_cols)):
            suggestions.append({"type": "Scatter Plot", "x": numeric_cols[i], "y": numeric_cols[j],
                                "description": f"Relationship between '{numeric_cols[i]}' and '{numeric_cols[j]}'"})
            suggestions.append({"type": "Density Heatmap", "x": numeric_cols[i], "y": numeric_cols[j],
                                "description": f"Density heatmap of '{numeric_cols[i]}' vs '{numeric_cols[j]}'"})

    # Categorical vs Numeric
    for cat_col in categorical_cols:
        for num_col in numeric_cols:
            suggestions.append({"type": "Box Plot", "x": cat_col, "y": num_col,
                                "description": f"Distribution of '{num_col}' across '{cat_col}' categories"})
            suggestions.append({"type": "Violin Plot", "x": cat_col, "y": num_col,
                                "description": f"Distribution shape of '{num_col}' across '{cat_col}' categories"})
            suggestions.append({"type": "Bar Chart", "x": cat_col, "y": num_col,
                                "description": f"Average of '{num_col}' by '{cat_col}' (use mean)"}) # This will require aggregation

    # Time series (simple detection)
    # Check if any column contains 'date' or 'time' and is of object/datetime type
    date_cols = [col for col in all_cols if 'date' in col.lower() or 'time' in col.lower() or df[col].dtype == 'datetime64[ns]']
    if date_cols:
        for date_col in date_cols:
            for num_col in numeric_cols:
                suggestions.append({"type": "Line Chart", "x": date_col, "y": num_col,
                                    "description": f"Trend of '{num_col}' over '{date_col}'"})

    return suggestions

# Function to generate Plotly chart based on user selections
def generate_chart(df, chart_type, x_col, y_col, color_col=None, size_col=None, names_col=None, values_col=None):
    fig = None
    try:
        if chart_type == "Bar Chart":
            if y_col == "count": # For frequency bar charts
                vc_df = df[x_col].value_counts().reset_index()
                vc_df.columns = [x_col, 'count']
                fig = px.bar(vc_df, x=x_col, y='count', color=color_col)
            else: # For aggregated bar charts (e.g., mean of y_col by x_col)
                # Group by x_col and calculate mean of y_col
                agg_df = df.groupby(x_col)[y_col].mean().reset_index()
                fig = px.bar(agg_df, x=x_col, y=y_col, color=color_col)
        elif chart_type == "Histogram":
            fig = px.histogram(df, x=x_col, color=color_col)
        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col)
        elif chart_type == "Line Chart":
            # Ensure x_col is suitable for line chart (e.g., sort by it)
            df_sorted = df.sort_values(by=x_col)
            fig = px.line(df_sorted, x=x_col, y=y_col, color=color_col)
        elif chart_type == "Box Plot":
            fig = px.box(df, x=x_col, y=y_col, color=color_col)
        elif chart_type == "Violin Plot":
            fig = px.violin(df, x=x_col, y=y_col, color=color_col)
        elif chart_type == "Pie Chart":
            if names_col and values_col:
                if values_col == "count":
                    pie_df = df[names_col].value_counts().reset_index()
                    pie_df.columns = [names_col, 'count']
                    fig = px.pie(pie_df, names=names_col, values='count')
                else:
                    fig = px.pie(df, names=names_col, values=values_col)
        elif chart_type == "Density Heatmap":
            fig = px.density_heatmap(df, x=x_col, y=y_col, color_continuous_scale="Viridis")
        else:
            st.warning("Unsupported chart type selected for manual visualization.")
            return None
        return fig
    except Exception as e:
        st.error(f"Error generating chart: {e}. Please check your column selections for the chosen chart type.")
        return None

# Main Streamlit app
st.title("📊 Smart Data Analyzer")

# Sidebar for file upload and page navigation
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
df = load_data(uploaded_file) # The explicit string conversion is now handled inside load_data

if df is not None:
    # Convert potential date columns to datetime objects for better plotting
    for col in df.columns:
        # Attempt to convert columns that look like dates
        try:
            if 'date' in col.lower() or 'time' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
                # Drop rows where date conversion failed
                df.dropna(subset=[col], inplace=True)
        except Exception:
            pass # Ignore if conversion fails, column is not a date

    page = st.sidebar.radio("Choose a page", ["Dataset", "Visualization", "Model", "AI Insights"])

    if page == "Dataset":
        st.subheader("🧾 Raw Dataset")
        st.dataframe(df)
        st.markdown(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

        st.markdown("### 📌 Dataset Information")
        st.markdown("#### Column Types")
        st.write(df.dtypes)

        st.markdown("#### Missing Values")
        st.write(df.isnull().sum())

        st.markdown("#### Descriptive Statistics")
        st.write(df.describe(include='all'))

    elif page == "Visualization":
        st.subheader("📊 Data Visualization")

        # --- Manual Visualization Section ---
        st.markdown("### 🛠️ Manual Chart Builder")
        st.write("Select columns and a chart type to build your own visualization.")

        col1, col2, col3 = st.columns(3)

        with col1:
            all_columns = df.columns.tolist()
            # Add "None" option for Y-axis if not needed (e.g., for histograms)
            y_axis_options = ["None"] + all_columns
            x_axis_col = st.selectbox("Select X-axis Column", all_columns, key="x_axis_manual")

        with col2:
            y_axis_col = st.selectbox("Select Y-axis Column", y_axis_options, key="y_axis_manual")

        with col3:
            chart_types = [
                "Bar Chart", "Histogram", "Scatter Plot", "Line Chart",
                "Box Plot", "Violin Plot", "Pie Chart", "Density Heatmap"
            ]
            chart_type_manual = st.selectbox("Select Chart Type", chart_types, key="chart_type_manual")

        # Optional aesthetic columns (color, size)
        st.markdown("##### Optional Aesthetic Mappings")
        col_color, col_size = st.columns(2)
        with col_color:
            color_col_options = ["None"] + all_columns
            color_col_manual = st.selectbox("Color by (Categorical)", color_col_options, key="color_manual")
            if color_col_manual == "None": color_col_manual = None
        with col_size:
            size_col_options = ["None"] + df.select_dtypes(include=np.number).columns.tolist()
            size_col_manual = st.selectbox("Size by (Numerical for Scatter)", size_col_options, key="size_manual")
            if size_col_manual == "None": size_col_manual = None

        if st.button("Generate Custom Chart"):
            # Logic to handle 'count' for Y-axis in bar/pie charts
            y_col_for_chart = y_axis_col if y_axis_col != "None" else "count"
            names_col_for_pie = x_axis_col if chart_type_manual == "Pie Chart" else None
            values_col_for_pie = y_col_for_chart if chart_type_manual == "Pie Chart" else None

            # Pass appropriate arguments to generate_chart
            fig_manual = generate_chart(
                df, chart_type_manual, x_axis_col, y_col_for_chart,
                color_col=color_col_manual, size_col=size_col_manual,
                names_col=names_col_for_pie, values_col=values_col_for_pie
            )
            if fig_manual:
                st.plotly_chart(fig_manual, use_container_width=True)

                # AI Insights for manual chart
                # Fix: Reconstruct the f-string to avoid backslashes in expression parts
                y_axis_insight_part = ""
                if y_axis_col != 'None':
                    y_axis_insight_part = f"and Y-axis as '{y_axis_col}'"

                color_insight_part = ""
                if color_col_manual:
                    color_insight_part = f"Colored by '{color_col_manual}'."

                # Constructing the prompt parts with proper spacing and punctuation
                chart_description_parts = []
                chart_description_parts.append(f"We plotted a {chart_type_manual.lower()} with X-axis as '{x_axis_col}'")
                if y_axis_insight_part:
                    chart_description_parts.append(y_axis_insight_part)
                if color_insight_part:
                    chart_description_parts.append(color_insight_part)

                # Join parts with a space, then add a period at the end of the description
                chart_description = " ".join(chart_description_parts) + "."

                ai_prompt_manual = (
                    f"{chart_description} "
                    f"Here is a sample of the dataset:\n{df.head(3).to_string(index=False)}\n"
                    "What insights, patterns, or trends can we derive from this visualization?"
                )
                with st.spinner("Getting AI insights for your custom chart..."):
                    ai_response_manual = get_ai_insights_sync(ai_prompt_manual)
                    st.markdown("### 🤖 AI Insights for Custom Chart")
                    st.write(ai_response_manual)

        st.markdown("---") # Separator

        # --- Smart Visualization Suggestions Section ---
        st.markdown("### 💡 Smart Visualization Suggestions")
        st.write("Based on your dataset, here are some suggested visualizations. Pick one to see the chart and AI insights.")

        suggestions = suggest_visualizations(df)
        # Format suggestions for display in selectbox
        formatted_suggestions = []
        for s in suggestions:
            cols_involved = []
            if s.get("x"): cols_involved.append(s["x"])
            if s.get("y") and s["y"] != "count": cols_involved.append(s["y"])
            if s.get("names"): cols_involved.append(s["names"])
            if s.get("values") and s["values"] != "count": cols_involved.append(s["values"])
            
            # Remove duplicates from cols_involved and join
            cols_str = ', '.join(sorted(list(set(cols_involved))))
            if not cols_str: # If no specific columns are listed (e.g., for a simple histogram with 'count')
                cols_str = "various columns"

            formatted_suggestions.append(f"{s['type']} - {cols_str}: {s['description']}")

        selected_suggestion_str = st.selectbox(
            "Pick a suggested visualization",
            options=formatted_suggestions,
            key="suggested_viz"
        )

        if selected_suggestion_str:
            # Find the original suggestion dictionary
            selected_index = formatted_suggestions.index(selected_suggestion_str)
            suggestion = suggestions[selected_index]

            chart_type = suggestion["type"]
            x_col = suggestion.get("x")
            y_col = suggestion.get("y")
            names_col = suggestion.get("names")
            values_col = suggestion.get("values")

            fig_suggested = generate_chart(
                df, chart_type, x_col, y_col,
                names_col=names_col, values_col=values_col
            )

            if fig_suggested:
                st.plotly_chart(fig_suggested, use_container_width=True)

                # AI Insights for suggested chart
                cols_used_for_ai = []
                if x_col: cols_used_for_ai.append(x_col)
                if y_col and y_col != "count": cols_used_for_ai.append(y_col)
                if names_col: cols_used_for_ai.append(names_col)
                if values_col and values_col != "count": cols_used_for_ai.append(values_col)
                
                ai_prompt_suggested = (
                    f"We plotted a {chart_type.lower()} using the columns {', '.join(cols_used_for_ai) if cols_used_for_ai else 'implied columns'}. "
                    f"Here is a sample of the dataset:\n{df.head(3).to_string(index=False)}\n"
                    "What insights, patterns, or trends can we derive from this visualization?"
                )
                with st.spinner("Getting AI insights for suggested chart..."):
                    ai_response_suggested = get_ai_insights_sync(ai_prompt_suggested)
                    st.markdown("### 🤖 AI Insights")
                    st.write(ai_response_suggested)

    elif page == "Model":
        st.subheader("🤖 Machine Learning Model")
        
        # Filter out non-numeric columns for target selection, unless there's a good reason to include them
        numeric_and_binary_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) or df[col].nunique() <= 2]
        
        if not numeric_and_binary_cols:
            st.warning("No suitable numeric or binary columns found for model training. Please upload a dataset with appropriate columns.")
        else:
            target_column = st.selectbox("Select the target column", numeric_and_binary_cols)
            if target_column:
                # Drop rows with NaN values in target or feature columns to avoid errors
                df_cleaned = df.dropna(subset=[target_column] + [col for col in df.columns if col != target_column])
                
                if df_cleaned.empty:
                    st.warning("Dataset became empty after dropping rows with missing values. Cannot train model.")
                else:
                    X = df_cleaned.drop(columns=[target_column])
                    y = df_cleaned[target_column]

                    # Convert categorical features to numerical using one-hot encoding
                    X = pd.get_dummies(X, drop_first=True)
                    
                    # FIX: Convert only numeric and boolean-like columns to float,
                    # excluding datetime columns which cannot be directly cast to float.
                    # This ensures that all features passed to the model are numerical.
                    # This also handles potential BoolDType issues.
                    for col in X.columns:
                        if pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col]):
                            X[col] = X[col].astype(float)


                    y_is_numeric = pd.api.types.is_numeric_dtype(y)
                    model_class = RandomForestRegressor if y_is_numeric else RandomForestClassifier

                    try:
                        # Ensure there are enough samples after one-hot encoding
                        if len(X) < 2 or len(y) < 2:
                            st.warning("Not enough data points to train the model after preprocessing. Need at least 2 samples.")
                        else:
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                            
                            # Check if training data is empty
                            if X_train.empty or y_train.empty:
                                st.warning("Training data is empty after split. Adjust test_size or provide more data.")
                            else:
                                model = model_class(random_state=42) # Add random_state for reproducibility
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)

                                if y_is_numeric:
                                    score = mean_squared_error(y_test, y_pred)
                                    st.metric("Mean Squared Error", round(score, 4))
                                else:
                                    score = accuracy_score(y_test, y_pred)
                                    st.metric("Accuracy", round(score, 4))

                                st.markdown("### 🔍 Feature Importances")
                                if not X.empty: # Ensure X is not empty before accessing columns
                                    importances = model.feature_importances_
                                    indices = np.argsort(importances)[::-1]
                                    feature_names = X.columns
                                    top_features = pd.DataFrame({
                                        'Feature': feature_names[indices],
                                        'Importance': importances[indices]
                                    }).head(10)
                                    st.dataframe(top_features)

                                    st.markdown("### 📈 Predicted vs Actual")
                                    if y_is_numeric:
                                        fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual', 'y': 'Predicted'},
                                                         title="Actual vs Predicted Values")
                                        fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(),
                                                      x1=y_test.max(), y1=y_test.max(),
                                                      line=dict(color="Red", width=2, dash="dash"),
                                                      name="Perfect Prediction")
                                    else:
                                        pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
                                        fig = px.histogram(pred_df, x="Actual", color="Predicted", barmode="group",
                                                           title="Actual vs Predicted Categories")
                                    st.plotly_chart(fig, use_container_width=True)

                                    ai_prompt = (
                                        f"We trained a {'regressor' if y_is_numeric else 'classifier'} on a dataset with shape {df_cleaned.shape}. "
                                        f"The target column is '{target_column}', and the top features were: "
                                        f"{', '.join(top_features['Feature'].tolist())}. What does this tell us about the factors influencing '{target_column}'?"
                                    )
                                    with st.spinner("Getting AI interpretation of feature importances..."):
                                        ai_response = get_ai_insights_sync(ai_prompt)
                                        st.markdown("### 🤖 AI Interpretation of Feature Importances")
                                        st.write(ai_response)
                                else:
                                    st.warning("No features available for importance calculation after one-hot encoding.")

                    except Exception as e:
                        st.error(f"Model training failed: {e}. Please check your data and selections.")

    elif page == "AI Insights":
        st.subheader("💡 General AI Insights")
        st.write("Ask a general question about your dataset and get insights from the AI.")
        question = st.text_area("Your question about the data:")
        if st.button("Get Answer") and question:
            preview = df.head(5).to_string(index=False)
            ai_prompt = f"Given the following data preview:\n{preview}\n\nAnswer the following question about the dataset:\n{question}"
            with st.spinner("Generating AI insights..."):
                ai_response = get_ai_insights_sync(ai_prompt)
                st.markdown("### 🤖 AI Answer")
                st.write(ai_response)
else:
    st.info("Please upload a CSV file to begin. Once uploaded, you can explore the dataset, visualize data, build machine learning models, and get AI insights.")
