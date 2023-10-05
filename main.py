import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(page_title='House Prices', page_icon='🏠')

st.title("House Prices 🏠")

st.write("Members of the team:")
members = ['Nguyễn Minh Thiên', 'Nguyễn Trương Hoàng Anh', 'Nguyễn Bảo Nguyên', 'Nguyễn Viết Khang']
for member in members:
    st.markdown("- " + member)

st.header("\nDataset")
# read data csv
df = pd.read_csv("./dataset/Housing.csv")
st.write(df)

st.write("\n**Select the CSV file with a similar format dataset above to rebuild the model**")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.title("Data of file csv")
    st.write(df)

view_df = df


def data_preprocess(data):
    cat_cols = [col for col in data.columns if data[col].dtype == 'object']
    le = LabelEncoder()
    for col in cat_cols:
        data[col] = le.fit_transform(data[col])
    return data


def build_model(df):
    df = data_preprocess(df)
    X = df.drop(['price'], axis=1)

    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    reg = Lasso(alpha=1)
    model = reg.fit(X_train, y_train)
    # Lưu mô hình vào một file
    joblib.dump(reg, './model/house_prices_model.pkl')
    return model


if st.button("Rebuild model"):
    build_model(df)
    st.toast('Build successful!', icon='🎉')

tab1, tab2 = st.tabs(["Price Prediction", "Data Visualization"])

with tab1:
    with st.form(key='form_info'):
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: left;} </style>',
                 unsafe_allow_html=True)

        st.write('House Information')

        mapping_yes_no = {"yes": "Yes", "no": "No"}
        mapping_furnishing_status = {"furnished": "Furnished", "semi-furnished": "Semi-Furnished",
                                     "unfurnished": "Unfurnished"}

        area = st.number_input("The total area of the house in square feet", value=0, min_value=0)
        bedrooms = st.number_input("The number of bedrooms in the house", value=0, min_value=0)
        bathrooms = st.number_input("The number of bathrooms in the house", value=0, min_value=0)
        stories = st.number_input("The number of stories in the house", value=0, min_value=0)
        parking = st.number_input("The number of parking spaces available within the house", value=0, min_value=0)
        main_road = st.radio("Whether the house is connected to the main road", ("yes", "no"),
                             format_func=lambda x: mapping_yes_no.get(x))
        guestroom = st.radio("Whether the house has a guest room", ("yes", "no"),
                             format_func=lambda x: mapping_yes_no.get(x))
        basement = st.radio("Whether the house has a basement", ("yes", "no"),
                            format_func=lambda x: mapping_yes_no.get(x))
        hot_water_heating = st.radio("Whether the house has a hot water heating system", ("yes", "no"),
                                     format_func=lambda x: mapping_yes_no.get(x))
        air_conditioning = st.radio("Whether the house has an air conditioning system", ("yes", "no"),
                                    format_func=lambda x: mapping_yes_no.get(x))
        pref_area = st.radio("Whether the house is located in a preferred area", ("yes", "no"),
                             format_func=lambda x: mapping_yes_no.get(x))
        furnishing_status = st.radio("Furnishing status", ("furnished", "semi-furnished", "unfurnished"),
                                     format_func=lambda x: mapping_furnishing_status.get(x))

        submit_form = st.form_submit_button(label="Prediction", help="Click to prediction!")

        # Checking if all the fields are not empty
        if submit_form:
            model = joblib.load('./model/house_prices_model.pkl')

            data = [[area, bedrooms, bathrooms, stories, main_road, guestroom, basement, hot_water_heating,
                     air_conditioning, parking, pref_area, furnishing_status]]
            df_input = pd.DataFrame(data,
                                    columns=['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom',
                                             'basement', 'hotwaterheating', 'airconditioning', 'parking',
                                             'prefarea', 'furnishingstatus'])

            result = model.predict(data_preprocess(df_input))

            # Định dạng giá trị tiền tệ với 2 chữ số thập phân
            formatted_price = "{:,.2f}".format(result[0], 2)
            st.success(
                f"House Prices 🏠 is {formatted_price}"
            )
with tab2:
    st.write("**The graphs show the correlation between data**")
    chart_df = view_df.loc[:, ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']]
    st.bar_chart(chart_df, x='area', y=['bedrooms', 'bathrooms', 'stories', 'parking'])

    fig1, ax1 = plt.subplots()
    sns.histplot(x=view_df['price'], ax=ax1)
    # fig1.suptitle('My Title')
    st.write(fig1)

    fig2, ax2 = plt.subplots()
    sns.scatterplot(y=view_df['price'], x=view_df['area'], hue=view_df['furnishingstatus'], ax=ax2)
    # fig2.suptitle('My Title')
    st.write(fig2)

    fig3, ax3 = plt.subplots()
    sns.barplot(x=view_df['airconditioning'], y=view_df['bedrooms'], hue=view_df["furnishingstatus"], ax=ax3)
    # fig3.suptitle('My Title')
    st.write(fig3)

    fig4, ax4 = plt.subplots()
    sns.barplot(x=view_df['hotwaterheating'], y=view_df['bathrooms'], hue=view_df["furnishingstatus"], ax=ax4)
    # fig4.suptitle('My Title')
    st.write(fig4)

    fig5, ax5 = plt.subplots()
    sns.heatmap(chart_df.corr(), cmap='viridis', annot=True, ax=ax5)
    # fig5.suptitle('My Title')
    st.write(fig5)
