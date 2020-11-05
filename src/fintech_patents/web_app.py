import streamlit as st


def app_details():
    r"""
    Here is where title, subtitle and app description will go
    """

    # Title
    st.title('Patent classification with Transformers')
    # Subtitle
    st.subheader('Make predictions')
    # Description
    st.write('More details go here...')

    return


# @st.cache(suppress_st_warning=True)
def custom(name):
    st.selectbox('Choose Model', ['Bert', 'RoBerTa'])

    default_patent = 'This is a patent'

    user_input = st.text_area("Patent Text Goes Here:", default_patent)

    if st.button('Get Prediction!'):
        
        st.text('Some Label')




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Setup app details
    app_details()

    custom('PyCharm')
