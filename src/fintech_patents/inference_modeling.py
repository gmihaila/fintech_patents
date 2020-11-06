# coding=utf-8
# Copyright 2020 George Mihaila.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Deal with model inference."""

import pickle


def inference_transformer(model_pickle_path, text_input, ids_labels):
    r"""
    Load model and tokenizer form .pickle and perform prediction using text input.


    """

    with open(model_pickle_path, 'rb') as handle:
        tokenizer, model = pickle.load(handle)
    inputs = tokenizer(text=text_input, add_special_tokens=True, truncation=True, padding=True, return_tensors='pt')

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
    # probs = torch.softmax(logits, dim=-1)

    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()

    # get predicitons to list
    predict_content = logits.argmax(axis=-1).flatten().tolist()[0]

    # Predicted label
    label = ids_labels.get(predict_content, 'Unknown')

    return label
