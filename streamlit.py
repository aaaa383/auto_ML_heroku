import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# 評価指標
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# auto-sklearn
import autosklearn.regression
import autosklearn.classification

# タイトルを表示
st.title('AutoMLツール')

st.sidebar.markdown("### 1. データの読み込み")
uploaded_file = st.sidebar.file_uploader("CSVファイルをドラッグ&ドロップ、またはブラウザから選択してください", type='csv', key='train')
if uploaded_file is not None:

    #データの読込み
    df = pd.read_csv(uploaded_file, encoding="shift-jis")

    #object型をcategory型に変更
    df.loc[:, df.dtypes == 'object'] = df.select_dtypes(['object']).apply(lambda x: x.astype('category'))

    #データの表示
    st.sidebar.markdown("### 2. データの情報の表示")
    if st.sidebar.checkbox('データの中身と基本統計量を表示しますか？'):
        st.markdown("### 1. アップロードされたデータを確認します")
        st.dataframe(df)

        st.markdown('### 2. 行数と列数を確認します')
        st.markdown(df.shape)

        st.markdown("### 3. 基本統計量を確認します")
        st.dataframe(df.describe())

        st.markdown("### 4. 欠損値の確認")
        st.dataframe(df.isnull().sum())

        st.markdown("### 5. 相関係数を確認します")
        st.dataframe(df.corr())

    st.sidebar.markdown("### 3. 目的変数の選択")
    target = st.sidebar.selectbox(
        '目的変数を選択してください',
        df.columns
    )

    # チェック時に目的変数と説明変数の相関を可視化
    st.sidebar.markdown("### 4. 説明変数と目的変数の関係性の確認")
    if st.sidebar.checkbox('散布図を表示しますか？'):
        checked_variable = st.selectbox(
            '説明変数を1つ選択してください:',
            df.select_dtypes(include='number').columns
        )
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.scatter(x=df[checked_variable], y=df[target])
        plt.xlabel(checked_variable)
        plt.ylabel(target)
        st.pyplot(fig)
    
    st.sidebar.markdown("### 5. 回帰問題(regression)か分類問題(classification)かの選択")
    ml_usecase = st.sidebar.selectbox(label='ドロップダウンリストからregressionかclassificationを選択してください',
                              options=('', 'regression', 'classification'),
                              key='ml_usecase')
    if ml_usecase == 'regression':
        
        #説明変数と目的変数の設定
        x = df.drop(target,axis=1)
        y = df[target]

        #検証用データの割合を設定
        TEST_SIZE  = st.sidebar.selectbox('検証用データの割合を設定してください',
                                 [0.1, 0.2, 0.3, 0.4, 0.5])


        # 学習データと評価データを作成
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=TEST_SIZE,
            random_state=2021,
            )

        if st.button('予測開始'):
            #モデルの学習
            automl = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=300,
                seed=2021,
                metric=autosklearn.metrics.mean_absolute_error)
            automl.fit(x_train, y_train)

            #予測
            y_pred = automl.predict(x_test)

            #精度評価
            scores = pd.DataFrame({
                "R2": r2_score(y_test, y_pred),
                "MAE": mean_absolute_error(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))},index=["scores"])
            
            st.sidebar.markdown("### 精度評価を表示")
            st.dataframe(scores)

    elif ml_usecase == 'classification':

        #説明変数と目的変数の設定
        x = df.drop(target,axis=1)
        y = df[target]

        #検証用データの割合を設定
        TEST_SIZE  = st.sidebar.selectbox('検証用データの割合を設定してください',
                                 [0.1, 0.2, 0.3, 0.4, 0.5])

        # 学習データと評価データを作成
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=TEST_SIZE,
            random_state=2021,
            )


        if st.button('予測開始'):
            #モデルの学習
            automl = autosklearn.classification.AutoSklearnClassifier(
                time_left_for_this_task=300,
                seed=2021,
                metric=autosklearn.metrics.f1)
            automl.fit(x_train, y_train)

            #予測
            y_pred = automl.predict(x_test)

            #精度評価
            scores = pd.DataFrame({
                "Accracy": r2_score(y_test, y_pred),
                "F1": mean_absolute_error(y_test, y_pred)},index=["scores"])

            st.sidebar.markdown("### 精度評価を表示")
            st.dataframe(scores)

    else:
        st.text('タスクが選択されていません')
