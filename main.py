import streamlit as st

# streamlit run main.py

st.title('Patent classification with Transformers')

st.subheader('Make predictions')

st.write('More details go here...')


# @st.cache(suppress_st_warning=True)
def custom(name):

    st.selectbox('Choose Model', ['Bert', 'RoBerTa'])

    default_patent = 'This is a patent'

    user_input = st.text_area("Patent Text Goes Here:", default_patent)

    if st.button('Get Prediction!'):
        st.text('Finetech')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    custom('PyCharm')