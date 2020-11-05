import streamlit as st
import torch
from transformers import (AutoConfig,
                          AutoModelForSequenceClassification,
                          AutoTokenizer,
                          set_seed)



IDS_LABELS = {0: 'insurance',
              1: 'payments',
              2: 'investment',
              3: 'fraud',
              4: 'data analytics',
              5: 'non-fintech'}

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
        label = inference_transformer(model_name_or_path='gmihaila/distilbert-base-uncased',
                                      text_input=user_input, ids_labels=IDS_LABELS)

        st.text(label)


def inference_transformer(model_name_or_path, text_input, ids_labels):
    # Look for gpu to use. Will use `cpu` by default if no gpu found.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set seed for reproducibility,
    set_seed(123)

    # Get model configuration.
    print('Loading configuraiton...')
    model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                              )

    # Get model's tokenizer.
    print('Loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)

    # Get the actual model.
    print('Loading model...')
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                               config=model_config)

    # Load model to defined device.
    model.to(device)
    print('Model loaded to `%s`' % device)

    inputs = tokenizer(text=text_input, add_special_tokens=True, truncation=True, padding=True, return_tensors='pt')

    model.eval()
    # move batch to device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        # Forward pass, calculate logit predictions.
        # This will return the logits rather than the loss because we have
        # not provided labels.
        # token_type_ids is the same as the "segment ids", which
        # differentiates sentence 1 and 2 in 2-sentence tasks.
        # The documentation for this `model` function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(**inputs)

        # The call to `model` always returns a tuple, so we need to pull the
        # loss value out of the tuple along with the logits. We will use logits
        # later to to calculate training accuracy.
        logits = outputs[0]

        # Get probablities from logits
        probs = torch.softmax(logits, dim=-1)

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        # get predicitons to list
        predict_content = logits.argmax(axis=-1).flatten().tolist()[0]

        # Predicted label
        label = ids_labels.get(predict_content, 'Unknown')

        return label

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Setup app details
    app_details()

    custom('PyCharm')