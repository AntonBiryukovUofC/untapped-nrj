import os
import pickle
from pathlib import Path
import altair as alt
import shap
import streamlit as st
import sys
import matplotlib
matplotlib.use('Agg')

alt.data_transformers.disable_max_rows()
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, '../../')
sys.path.insert(0, project_dir)
from src.visualization.utils import intro_blurb, feature_engineering, modelling_approach, model_blurb, modelconf_blurb, \
    feat_imp_blurb, target_transform_blurb
# from src.models.train_model_with_val_feat_select_boxcox import LogLGBM
import pandas as pd
import numpy as np

tgt_dict = dict(zip(['Oil', 'Gas', 'Water'], ['Oil_norm', 'Gas_norm', 'Water_norm']))


@st.cache(show_spinner=True)
def get_data():
    input_file_path = os.path.join(project_dir, "data", "final")
    train_file_name = os.path.join(input_file_path, "Train_final.pck")
    val_file_name = os.path.join(input_file_path, "Validation_final.pck")
    test_file_name = os.path.join(input_file_path, "Test_final.pck")

    df_train = pd.read_pickle(train_file_name)
    df_val = pd.read_pickle(val_file_name)
    df_test = pd.read_pickle(test_file_name)

    return df_train, df_val, df_test


@st.cache(show_spinner=True)
def get_models():
    model = {}
    for tgt in list(tgt_dict.values()):
        output_file_name = os.path.join(project_dir, 'models', f"models_lgbm_{tgt}.pck")
        with open(output_file_name, 'rb') as f:
            model[tgt] = pickle.load(f)
    return model


@st.cache(show_spinner=True)
def get_shap():
    shap = {}
    for tgt in list(tgt_dict.values()):
        output_file_name = os.path.join(project_dir, 'models', f"shap_{tgt}.pck")
        with open(output_file_name, 'rb') as f:
            shap[tgt] = pickle.load(f)
    return shap


st.title('Model results exploration - regression')
st.sidebar.title('Model explorer - Regression')
st.sidebar.markdown(intro_blurb)

df_train, df_val, df_test = get_data()
shaps = get_shap()

# models = get_models()

st.sidebar.header('Parameters of a model')
tgt = st.sidebar.selectbox('Target of prediction', options=list(tgt_dict.keys()), index=0)
pdp_feature = st.sidebar.selectbox('Partial Dependence plot Feature', options=shaps[tgt_dict[tgt]]['feature_names'],
                                   index=len(shaps[tgt_dict[tgt]]['feature_names'])-1)
st.header('Modelling approach')
st.write(modelling_approach)
# TODO Altair with test-train colored lat-long
st.header('Data split, visualized')
st.write('Notice an evidently randomized split between the train/test/validation datasets. Notice a clearly '
         'non-stationary HZ length distribution in the figure below. Try selecting different windows along the time '
         'axis to see the distribution of selected wells around the basin (*very non-random*)')
latlong_cols = ['Surf_Longitude', 'Surf_Latitude', 'LengthDrill', 'haversine_Length', '_Max`Prod`(BOE)', 'SpudDate_dt']
vmin = [df_train[x].min() * 0.999 for x in latlong_cols[0:len(latlong_cols) - 1]]
vmax = [df_train[x].max() * 1.001 for x in latlong_cols[0:len(latlong_cols) - 1]]
df_latlong: pd.DataFrame = pd.concat([df_train[latlong_cols], df_val[latlong_cols], df_test[latlong_cols]],
                                     keys=['Train', 'Validation', 'Test'], axis=0).reset_index().rename(
    columns={'level_0': 'Split'}).sort_values('SpudDate_dt')
df_latlong['rolling_mean'] = df_latlong.rolling(window=100)['LengthDrill'].median()
# Add selector:
selector = alt.selection_interval(encodings=['x'], empty='all')

ch_base = alt.Chart(df_latlong, width=300)

ch_map = ch_base.encode(
    x=alt.Longitude(latlong_cols[0], scale=alt.Scale(domain=(vmin[0], vmax[0]))),
    y=alt.Latitude(latlong_cols[1], scale=alt.Scale(domain=(vmin[1], vmax[1]))),
    column='Split',
    color=alt.condition(selector, 'Split', alt.value('lightgray'))).mark_point(filled=True)

ch_time_component = ch_base.add_selection(selector).encode(
    x=alt.X('yearmonth(SpudDate_dt)', scale=alt.Scale(zero=False)),
    y=alt.Y('LengthDrill', scale=alt.Scale(domain=(1100, 3000))),
    color=alt.condition(selector, 'Split', alt.value('lightgray'))).mark_point(filled=True, clip=True).properties(
    width=700)

ch_rm = ch_base.encode(
    x=alt.X('SpudDate_dt', scale=alt.Scale(zero=False), title=None),
    y='rolling_mean', color=alt.value('black')).mark_point(filled=True)

st.write(ch_map & (ch_time_component + ch_rm))
# _Max`Prod`(BOE)
st.header('The evolution of maximum production over vintage ')
st.write(
    f'The effect of *bigger fracs* over time is more evident if we plot Max BOE vs time - there is clearly a drift '
    f'towards higher values. Wells production distribution has long tails - '
    f'therefore logarithm of Max BOE is shown below. Compare that with a trend in target IP {tgt} we need to predict.')
# TODO Altair with SpudDate vs maxboe log plot
ch_boe = alt.Chart(df_latlong, width=400).encode(x=alt.X('yearmonth(SpudDate_dt)', scale=alt.Scale(zero=False)),
                                                 y=alt.Y(latlong_cols[-2], scale=alt.Scale(type='log')),
                                                 color='LengthDrill').mark_point(filled=True).interactive()
ch_tgt = alt.Chart(df_train, width=400).encode(x=alt.X('yearmonth(SpudDate_dt)', scale=alt.Scale(zero=False)),
                                               y=alt.Y(tgt_dict[tgt], scale=alt.Scale(type='log')),
                                               color='LengthDrill').mark_point(filled=True).interactive()
st.write(ch_boe | ch_tgt)
st.write(
    'We also know that proppant intensity has a prominent effect on IP. However, we are missing this important parameter and therefore can only rely on `SpudDate` acting as a proxy for it.')
st.image('proppant-vs-target.png', format='png')
st.write(
    '*Gas production vs proppant intensity. An upward trend with signs of plateau is visible, regardless of target zone. Courtesy of Verdazo Analytics*')

st.header('Feature Engineering')
st.write(feature_engineering)
st.header('Target transform')
st.write(target_transform_blurb)
# TODO altair transformed pre -after
df_target =df_train.loc[:, [tgt_dict[tgt]]+['EPAssetsId']]
df_target = df_target[df_target[tgt_dict[tgt]] < 150]
df_target[f'Log_{tgt}']= np.log1p(df_train[tgt_dict[tgt]])
ch_tgt = alt.Chart(data=df_target)
ch_pre = ch_tgt.encode(x=alt.X(tgt_dict[tgt],bin=alt.Bin(maxbins=55),title=f'{tgt} raw'), y='count()').mark_bar()
ch_after = ch_tgt.encode(x=alt.X(f'Log_{tgt}',bin=alt.Bin(maxbins=55),title = f'{tgt} after log-transform'), y='count()').mark_bar()
ch_tgt_all = ch_pre | ch_after
st.write(ch_tgt_all)
st.header('Model configuration')
st.write(modelconf_blurb)
st.header('Model result exploration')
st.write(
    'Lets analyse first the residuals: we need to make sure our model has at least some predictive power, and is better than a geometric mean baseline. As our model was trained to predict logarithmic target, '
    'and optimized MSE in log-target space, we can use scatterplots of log target vs log predictions to see whether our predictions are biased.')

st.write(model_blurb)
st.header(f'Feature importance for {tgt}')
st.write(feat_imp_blurb)
shap_tgt = shaps[tgt_dict[tgt]]
# st.write(shaps[0])
shap_df = pd.DataFrame(data=shap_tgt['shap_values'], columns=shap_tgt['feature_names'],
                       index=np.arange(shap_tgt['shap_values'].shape[0]))
shap_df['id'] = shap_df.index
cols_importance = shap_df.abs().sum().sort_values().tail(20).index.tolist()
#st.write()
shap.summary_plot(shap_tgt['shap_values'],shap_tgt['X_train'],show=False,plot_size=(10,7),
                  max_display=10)
#fig, ax = plt.gcf(), plt.gca()
st.pyplot(bbox_inches = "tight",dpi=200)
st.header(f'Partial Dependence Plots for {pdp_feature}')
shap.dependence_plot(pdp_feature,shap_tgt['shap_values'],shap_tgt['X_train'],xmin='percentile(2)',xmax='percentile(97)',show=False,alpha=0.6)
st.pyplot(bbox_inches = "tight",dpi=200)

