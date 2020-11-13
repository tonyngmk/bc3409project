import streamlit as st
# import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.ensemble import GradientBoostingRegressor
# import xgboost
import shap
# from sklearn.model_selection import KFold
# import SessionState

@st.cache(persist = True)
def load_data():
    X = pd.read_csv("X.csv")
    X_scaled = pd.read_csv("X_scaled.csv")
    y = pd.read_csv("y.csv")
    X_train = X_scaled[X_scaled["reviews_per_month"] > 0]
    X_test = X_scaled[X_scaled["reviews_per_month"] <= 0]
    X_topup = X_test.sample(n=2928)
    X_train = pd.concat([X_scaled[X_scaled["reviews_per_month"] > 0], X_topup])
    X_test = X_test.drop(X_topup.index)
    y_train = y.loc[X_train.index, :]
    y_test = y.loc[X_test.index, :]
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    df = pd.concat([y, X], axis=1)
    df_test = pd.concat([y_test, X_test], axis=1)
    del X_topup
    return X, y, X_train, y_train, X_test, y_test, df, df_test
    
# @st.cache
# def load_gbm():
    # model_gb =  GradientBoostingRegressor(n_estimators = 220).fit(X_train, y_train.values.ravel())
    # explainer = shap.TreeExplainer(model_gb)
    # shap_values = explainer.shap_values(X_test)
    # return explainer, shap_values
# @st.cache(hash_funcs={'xgboost.XGBRegressor': id})
# def load_xgboost():
    # lr = 0.032
    # n_folds = 10
    # numerical_features =  X_train.select_dtypes(exclude=['object'])
    # kf = KFold(n_folds, shuffle=True, random_state = 2020).get_n_splits(numerical_features)
    # xgb_model = xgboost.XGBRegressor(objective ='reg:squarederror',\
                                            # subsample = 0.6, \
                                            # reg_lambda = 2.5, n_estimators  = 75, max_depth = 13, \
                                            # learning_rate = lr,\
                                            # gamma = 0, colsample_bytree = .9,\
                                            # early_stopping=5)
    # xgb_model.fit(X_train, y_train)
    # explainer = shap.TreeExplainer(xgb_model)
    # shap_values = explainer.shap_values(X_test)
    # return xgb_model

# def st_shap(plot, height=None):
    # shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    # components.html(shap_html, height=height)

st.image('https://raw.githubusercontent.com/tonyngmk/free_storage/master/Images/NTU%20Logo.png', width = 750)
st.write('''<h1 align=center><font color='Blue'>BC3409</font> - 
<font color='red'>AI in Accounting & Finance</font>''', unsafe_allow_html=True)

# use_case = st.sidebar.selectbox("Select use cases", 
# ("GBM",
# "XGBoost"
# ))


st.write("<h2 align=center>Airbnb Optimal Price Recommender</h2>", unsafe_allow_html=True)
'''A key part to address the problem of this assignment is to advise property listers of an ‘optimum price’ value given a set of input variables in their pricing decisions. 
This web app provides a peek at the high interpretability and explainability of using SHAP for both GBM and XGBoost. '''
use_case = st.radio("Select model:", 
("GBM",
"XGBoost"
))

# with st.spinner(''):
_, _, X_train, y_train, X_test, y_test, _, df_test = load_data()
# session_state = SessionState.get(n=0)

if use_case == "GBM":
    st.write("<h2 align=center>{}</h2>".format(use_case), unsafe_allow_html=True)
    # with st.spinner('Do allow approximately 5s for GBM to be fitted.'):
    f = open("gbmBase.txt")
    gbmBaseValue = float(f.readlines()[0])
    f.close()
    gbmVals = np.load("gbmValues.npy")
    
    # if st.button('Start Training'):
    # with st.spinner('Waiting for models to be fitted'):
        # model_gb = GradientBoostingRegressor(n_estimators = 220).fit(X_train, y_train.values.ravel())
        # explainer = shap.TreeExplainer(model_gb)
        # shap_values = explainer.shap_values(X_test)
    # features = st.radio(
     # "Which features are you interested in?",
    # ('Price Prediction Breakdown', 
    # 'All Prediction Breakdown', 
    # 'Feature Importance Summary', 
    # 'Feature Dependence Plot'))
    # if features == "Price Prediction Breakdown":
        # if st.button('Start Training'):
            # with st.spinner('Waiting for models to be fitted'):
                # model_gb = GradientBoostingRegressor(n_estimators = 220).fit(X_train, y_train.values.ravel())
                # explainer = shap.TreeExplainer(model_gb)
                # shap_values = explainer.shap_values(X_test)

    '###### *We will be using cached predictions to conserve CPU usage (Heroku dynos) and avoid long train/fitting time.*'
    st.success('Done!')
    "#### Reproducible code:"
    ''' 
    ~~~
    model_gb =  GradientBoostingRegressor(n_estimators = 220).fit(X_train, y_train.values.ravel())
    explainer = shap.TreeExplainer(model_gb)
    shap_values = explainer.shap_values(X_test)
    ~~~
    '''
    # with st.echo():
        # ''' 
            # model_gb =  GradientBoostingRegressor(n_estimators = 220).fit(X_train, y_train.values.ravel())\
            # explainer = shap.TreeExplainer(model_gb)\
            # shap_values = explainer.shap_values(X_test)
        # '''
    "## SHAP Explainer"
    # placeholder1 = st.empty()
    # placeholder2 = st.empty()
    n = 0
    n = st.slider("Select test case:", 0, len(df_test), n)
    n = st.number_input("Alternatively, type desired row:", 0, len(df_test), n)
    # "**Original y-value:** ", y_test.iloc[n].values.tolist()[0]
    # "**Predicted y-hat:** ",  (explainer.expected_value + shap_values[n,:].sum())[0]
    # "**Residual MAE:**", abs(y_test.iloc[n].values.tolist()[0] - (explainer.expected_value + shap_values[n,:].sum())[0])
    # n = session_state.n
    results = pd.DataFrame({"Original y-value": y_test.iloc[n].values.tolist()[0], 
                  "Predicted y-hat": (gbmBaseValue + gbmVals[n,:].sum()),
                  "Residual MAE": abs(y_test.iloc[n].values.tolist()[0] - (gbmBaseValue + gbmVals[n,:].sum()))},
                  index = ["Test Case Result: {}".format(n)])
    results
    # st_shap(shap.force_plot(gbmBaseValue, gbmVals[n,:], X_test.iloc[n,:]))
    # st.pyplot(shap.force_plot(gbmBaseValue, gbmVals[n,:], X_test.iloc[n,:]),bbox_inches='tight',dpi=300,pad_inches=0)
    with st.spinner('Plotting, please wait for approximately 3 seconds...'):
        st.pyplot(shap.force_plot(gbmBaseValue, gbmVals[n,:], X_test.iloc[n,:],matplotlib=True,show=False),bbox_inches='tight',dpi=300,pad_inches=0)
        plt.clf()
    "**Breakdown:**"
    breakdownCols = ["Base value"] + X_test.columns.tolist()
    breakdownVals = [gbmBaseValue] + list(gbmVals[n,:])
    breakdown = pd.DataFrame(breakdownVals, index = breakdownCols, columns = ["Value"])
    breakdown
    "#### Sum of Value (Predicted Y-Hat):", breakdown["Value"].sum()
    # "Base value:", explainer.expected_value[0]
    # for i, v in enumerate(shap_values[n,:]):
        # X_test.columns[i],":", v
    # if st.checkbox("Show Raw Data"):
    "## Raw Data"
    "**Sample dataset** (top 5 rows)"
    df_test.iloc[:6, :]
    "### Inspect Test Set"
    nrows = st.slider("Select number of rows:", 0, len(df_test))
    df_test.iloc[:nrows, :]
    # model_gb = load_gbm(X_train, y_train)
    # explainer = shap.TreeExplainer(model_gb)
    # shap_values = explainer.shap_values(X_test)

elif use_case == "XGBoost":
    st.write("<h2 align=center>{}</h2>".format(use_case), unsafe_allow_html=True)
    f = open("xgboostBase.txt")
    xgbBaseValue = float(f.readlines()[0])
    f.close()
    xgbVals = np.load("xgboostValues.npy")
    '###### *We will be using cached predictions to conserve CPU usage (Heroku dynos) and avoid long train/fitting time.*'
    st.success('Done!')
    # with st.spinner('Do allow approximately 60s for XGBoost model to be fitted.'):
        # lr = 0.032
        # n_folds = 10
        # numerical_features =  X_train.select_dtypes(exclude=['object'])
        # kf = KFold(n_folds, shuffle=True, random_state = 2020).get_n_splits(numerical_features)
        # xgb_model = xgboost.XGBRegressor(objective ='reg:squarederror',\
                                                # subsample = 0.6, \
                                                # reg_lambda = 2.5, n_estimators  = 75, max_depth = 13, \
                                                # learning_rate = lr,\
                                                # gamma = 0, colsample_bytree = .9,\
                                                # early_stopping=5)
        # xgb_model.fit(X_train, y_train)
        # explainer = shap.TreeExplainer(xgb_model)
        # shap_values = explainer.shap_values(X_test)

    "#### Reproducible code:"
    ''' 
    ~~~
    lr = 0.032
    n_folds = 10
    numerical_features =  X_train.select_dtypes(exclude=['object'])
    kf = KFold(n_folds, shuffle=True, random_state = 2020).get_n_splits(numerical_features)
    xgb_model = xgboost.XGBRegressor(objective ='reg:squarederror',\
subsample = 0.6, \
reg_lambda = 2.5, n_estimators  = 75, max_depth = 13, \
learning_rate = lr,\
gamma = 0, colsample_bytree = .9,\
early_stopping=5)
    xgb_model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    ~~~
    '''
    "## SHAP Explainer"
    n = 0
    n = st.slider("Select test case:", 0, len(df_test), n)
    n = st.number_input("Alternatively, type desired row:", 0, len(df_test), n)
    results = pd.DataFrame({"Original y-value": y_test.iloc[n].values.tolist()[0], 
                  "Predicted y-hat": (xgbBaseValue + xgbVals[n,:].sum()),
                  "Residual MAE": abs(y_test.iloc[n].values.tolist()[0] - (xgbBaseValue + xgbVals[n,:].sum()))},
                  index = ["Test Case Result: {}".format(n)])
    results
    # st_shap(shap.force_plot(xgbBaseValue, xgbVals[n,:], X_test.iloc[n,:]))
    with st.spinner('Plotting, please wait for approximately 3 seconds...'):
        st.pyplot(shap.force_plot(xgbBaseValue, xgbVals[n,:], X_test.iloc[n,:],matplotlib=True,show=False),bbox_inches='tight',dpi=300,pad_inches=0)
        plt.clf()
    "**Breakdown:**"
    breakdownCols = ["Base value"] + X_test.columns.tolist()
    breakdownVals = [xgbBaseValue] + list(xgbVals[n,:])
    breakdown = pd.DataFrame(breakdownVals, index = breakdownCols, columns = ["Value"])
    breakdown
    "#### Sum of Value (Predicted Y-Hat):", breakdown["Value"].sum()
    "## Raw Data"
    "**Sample dataset** (top 5 rows)"
    df_test.iloc[:6, :]
    "### Inspect Test Set"
    nrows = st.slider("Select number of rows:", 0, len(df_test))
    df_test.iloc[:nrows, :]

    
hide_streamlit_style = """
            <style>
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

myWatermark = """
            <style>
            footer:after {
            content:'Tony Ng'; 
            visibility: visible;
            display: block;
            position: relative;
            #background-color: red;
            padding: 5px;
            top: 2px;
            }
            </style>
            """
st.markdown(myWatermark, unsafe_allow_html=True)