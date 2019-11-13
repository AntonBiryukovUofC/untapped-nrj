import os
import pickle
from pathlib import Path
import altair as alt
import shap
import streamlit as st
import sys
import matplotlib
from sklearn.cluster import KMeans

matplotlib.use('Agg')

alt.data_transformers.disable_max_rows()
project_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, '../../')
sys.path.insert(0, project_dir)
from src.visualization.utils import intro_blurb, feature_engineering, model_blurb, modelconf_blurb, \
    feat_imp_blurb, target_transform_blurb, follow_up_blurb, classification_fi_blurb, modelling_approach_class, \
    modelling_approach_reg, cv_blurb
# from src.models.train_model_with_val_feat_select_boxcox import LogLGBM
import pandas as pd
import numpy as np
tgt_dict = dict(zip(['Oil', 'Gas', 'Water'], ['Oil_norm', 'Gas_norm', 'Water_norm']))

def my_theme():
    return {
        'config': {
            'view': {
                'height': 300,
                'width': 400,
            },
            'mark': {
                'color': 'black',
                'fill': 'black'
            },
            'title':{
                'fontSize':14
            },
            'axis':{
                'labelFontSize':13,
                "titleFontSize":15
            },
            'header':{
                "labelFontSize":18,
                "titleFontSize":18
            }
        }
    }




alt.themes.register('my_theme',my_theme)
alt.themes.enable('my_theme')

@st.cache(show_spinner=True)
def get_data():
    input_file_path = os.path.join(project_dir, "data", "final")
    oof_preds = pd.read_pickle(os.path.join(input_file_path, "OOF.pck"))
    train_file_name = os.path.join(input_file_path, "Train_final.pck")
    val_file_name = os.path.join(input_file_path, "Validation_final.pck")
    test_file_name = os.path.join(input_file_path, "Test_final.pck")

    df_train = pd.read_pickle(train_file_name)
    df_val = pd.read_pickle(val_file_name)
    df_test = pd.read_pickle(test_file_name)

    return df_train, df_val, df_test, oof_preds


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

st.markdown(
        f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 1200px;
    }}
    .reportview-container .main {{
    }}
</style>
""",
        unsafe_allow_html=True,
    )

st.image("https://bracewell.com/sites/default/files/styles/banner/public/practice-banners/Header_OilGasProjects_0.jpg")
st.write('*Image - courtesy of Bracewell LLP*')
st.title('\N{writing hand} Model results exploration - Untapped reClaim Challenge')
st.sidebar.title('\U000026A1 Model explorer - regression & classification')
st.sidebar.markdown(intro_blurb)

df_train, df_val, df_test, oof_preds = get_data()
shaps = get_shap()

# models = get_models()

st.sidebar.header('\N{gear} Model parameters')
tgt = st.sidebar.selectbox('Target of prediction', options=list(tgt_dict.keys()), index=0)
pdp_feature = st.sidebar.selectbox('Partial Dependence plot Feature', options=shaps[tgt_dict[tgt]]['feature_names'],
                                   index=len(shaps[tgt_dict[tgt]]['feature_names']) - 1)
st.header('\N{hammer and pick} Modelling approach')
st.write(modelling_approach_reg[0])
st.image('https://github.com/AntonBiryukovUofC/untapped-nrj/raw/master/src/visualization/regression-slide.png',width=1000)
st.write('*Regression in brief*')
st.write(modelling_approach_class)
st.image('https://github.com/AntonBiryukovUofC/untapped-nrj/raw/master/src/visualization/classification-slide.png',width=1000)
st.write('*If you can do Regression, then you can do classification!*')

st.write(modelling_approach_reg[1])

latlong_cols = ['Surf_Longitude', 'Surf_Latitude', 'LengthDrill', 'haversine_Length', '_Max`Prod`(BOE)', 'SpudDate_dt']
vmin = [df_train[x].min() * 0.999 for x in latlong_cols[0:len(latlong_cols) - 1]]
vmax = [df_train[x].max() * 1.001 for x in latlong_cols[0:len(latlong_cols) - 1]]
df_latlong: pd.DataFrame = pd.concat([df_train[latlong_cols], df_val[latlong_cols], df_test[latlong_cols]],
                                     keys=['Train', 'Validation', 'Test'], axis=0).reset_index().rename(
    columns={'level_0': 'Split'}).sort_values('SpudDate_dt')
df_latlong['rolling_mean'] = df_latlong.rolling(window=100)['LengthDrill'].median()


st.header('\N{pick} Feature Engineering')
st.write(feature_engineering)

st.header('\N{sparkles} The evolution of maximum production over vintage ')
st.write(
    f'The effect of *bigger fracs* over time is more evident if we plot Max BOE vs time - there is clearly a drift '
    f'towards higher values. Wells production distribution has long tails - '
    f'therefore logarithm of Max BOE is shown below. Compare that with a trend in target IP {tgt} we need to predict.')
ch_boe = alt.Chart(df_latlong.sample(frac=0.3), width=400).encode(x=alt.X('yearmonth(SpudDate_dt)', scale=alt.Scale(zero=False)),
                                                 y=alt.Y(latlong_cols[-2], scale=alt.Scale(type='log')),
                                                 color='LengthDrill').mark_point(filled=True).interactive()
ch_tgt = alt.Chart(df_train.sample(frac=0.3), width=400).encode(x=alt.X('yearmonth(SpudDate_dt)', scale=alt.Scale(zero=False)),
                                               y=alt.Y(tgt_dict[tgt], scale=alt.Scale(type='log')),
                                               color='LengthDrill').mark_point(filled=True).interactive()
st.write('*The chart above is zoom-able.')
st.write(ch_boe | ch_tgt)
st.write(
    'We also know that proppant intensity has a prominent effect on IP. However, we are missing this important parameter and therefore can only rely on `SpudDate` acting as a proxy for it.')
st.image('proppant-vs-target.png', format='png')
st.write(
    '*Gas production vs proppant intensity. An upward trend with signs of plateau is visible, regardless of target zone. Courtesy of Verdazo Analytics*')


st.header('\N{curly loop} Target transform')
st.write(target_transform_blurb)
df_target = df_train.loc[:, [tgt_dict[tgt]] + ['EPAssetsId']]
df_target = df_target[df_target[tgt_dict[tgt]] < 150]
df_target[f'Log_{tgt}'] = np.log1p(df_train[tgt_dict[tgt]])
ch_tgt = alt.Chart(data=df_target.sample(frac=0.1))
ch_pre = ch_tgt.encode(x=alt.X(tgt_dict[tgt], bin=alt.Bin(maxbins=55), title=f'{tgt} raw'), y='count()').mark_bar()
ch_after = ch_tgt.encode(x=alt.X(f'Log_{tgt}', bin=alt.Bin(maxbins=55), title=f'{tgt} after log-transform'),
                         y='count()').mark_bar()
ch_tgt_all = ch_pre | ch_after
st.write(ch_tgt_all)


st.header('Data split, visualized')
st.write(cv_blurb)

# Add selector:
selector = alt.selection_interval(encodings=['x'], empty='all')

ch_base = alt.Chart(df_latlong.sample(frac=0.1), width=300)

ch_map = ch_base.encode(
    x=alt.Longitude(latlong_cols[0], scale=alt.Scale(domain=(vmin[0], vmax[0]))),
    y=alt.Latitude(latlong_cols[1], scale=alt.Scale(domain=(vmin[1], vmax[1]))),
    column='Split',
    color=alt.condition(selector, 'Split', alt.value('lightgray')),
    opacity=alt.condition(selector, alt.value(0.99), alt.value(0.05))).mark_point(filled=True)

ch_time_component = ch_base.add_selection(selector).encode(
    x=alt.X('yearmonth(SpudDate_dt)', scale=alt.Scale(zero=False)),
    y=alt.Y('LengthDrill', scale=alt.Scale(domain=(1100, 3000))),
    color=alt.condition(selector, 'Split', alt.value('lightgray'))).mark_point(filled=True, clip=True).properties(
    width=700)

ch_rm = ch_base.encode(
    x=alt.X('SpudDate_dt', scale=alt.Scale(zero=False), title=None),
    y='rolling_mean', color=alt.value('black')).mark_point(filled=True)

st.write(ch_map & (ch_time_component + ch_rm))
st.write('** Try selecting / dragging a selection in the panel above (the chart is interactive!)')
# _Max`Prod`(BOE)

st.header('\N{gear} Model configuration')
st.write(modelconf_blurb)
st.header('\U00002139 Model result exploration')
st.write(
    """
Lets analyse first the residuals: we need to make sure our model has at least some predictive power, and is better 
than a geometric mean baseline. As our model was trained to predict logarithmic target,
and optimized MSE in log-target space, we can use scatterplots of log target vs log predictions to see whether our 
predictions are biased.
    
    
As you might notice, RMSE grows as the value of target increases (look at that cone-shaped scatter).
However, RMSLE stays more or less stationary as target value increases. We can also notice a bias in our predictions:
the points are not quite scattered along the `Pred=Target` line. This indicates that there is missing information that
 this model could benefit from to help explain the target. That is something you would certainly expect from a 
     
    """)
oof_preds = oof_preds[oof_preds[f'{tgt_dict[tgt]}'] > 0.5]

oof_preds[f'log_{tgt_dict[tgt]}'] = np.log1p(oof_preds[f'{tgt_dict[tgt]}'])
oof_preds[f'log_gt_{tgt_dict[tgt]}'] = np.log1p(oof_preds[f'gt_{tgt_dict[tgt]}'])
oof_preds = oof_preds.drop_duplicates(subset=[f'log_{tgt_dict[tgt]}'])

ch_oof = alt.Chart(oof_preds.sample(frac=0.1)).encode(x=alt.X(f'{tgt_dict[tgt]}', title='Pred'),
                                     y=alt.Y(f'gt_{tgt_dict[tgt]}', title='Target')).mark_point(filled=True,
                                                                                                opacity=0.4)
ch_oof_log = alt.Chart(oof_preds.sample(frac=0.1)).encode(x=alt.X(f'log_{tgt_dict[tgt]}', title='Log Pred'),
                                         y=alt.Y(f'log_gt_{tgt_dict[tgt]}', title='Log Target')).mark_point(filled=True,
                                                                                                            opacity=0.4)

st.write(ch_oof | ch_oof_log)

st.write(model_blurb)
st.header(f'\N{exclamation mark} Feature importance for {tgt}')
st.write(feat_imp_blurb)
shap_tgt = shaps[tgt_dict[tgt]]
# st.write(shaps[0])
shap_df = pd.DataFrame(data=shap_tgt['shap_values'], columns=shap_tgt['feature_names'],
                       index=np.arange(shap_tgt['shap_values'].shape[0]))
shap_df['id'] = shap_df.index
cols_importance = shap_df.abs().sum().sort_values().tail(20).index.tolist()
# st.write()
shap.summary_plot(shap_tgt['shap_values'], shap_tgt['X_train'], show=False, plot_size=(10, 7),
                  max_display=10)
# fig, ax = plt.gcf(), plt.gca()
st.pyplot(bbox_inches="tight", dpi=200)
st.header(f'\N{sparkle} Partial Dependence Plots for {pdp_feature}')
shap.dependence_plot(pdp_feature, shap_tgt['shap_values'], shap_tgt['X_train'], xmin='percentile(2)',
                     xmax='percentile(97)', show=False, alpha=0.6)
st.pyplot(bbox_inches="tight", dpi=200)

st.write(classification_fi_blurb)
st.image('./fi_Class_1.png',width=1000)
st.image('./fi_Class_2.png',width=1000)

st.header('\U000023E9 Next steps & afterword on the value of the model')
n_cl = 40
r2_dict = dict(zip(np.arange(n_cl), np.random.beta(a=5, b=7, size=n_cl)))
km = KMeans(n_clusters=n_cl)
km.fit(df_latlong[[latlong_cols[0], latlong_cols[1]]])
df_latlong['cluster'] = km.predict(df_latlong[[latlong_cols[0], latlong_cols[1]]])
df_latlong['R2'] = df_latlong['cluster'].apply(lambda x: r2_dict[x])

# selector_cl = alt.selection_interval(encodings=['x', 'y'], empty='all')
ch_cluster = alt.Chart(data=df_latlong.sample(frac=0.05), width=600, height=400).encode(
    x=alt.Longitude(latlong_cols[0], scale=alt.Scale(domain=(vmin[0], vmax[0]))),
    y=alt.Latitude(latlong_cols[1], scale=alt.Scale(domain=(vmin[1], vmax[1]))),
    color=alt.Color('R2', scale=alt.Scale(scheme='viridis')),
).mark_point(filled=True, size=150).interactive()

st.write(follow_up_blurb)
st.altair_chart(ch_cluster,width=700)
st.write('*Try zooming in and out in the chart above*')
