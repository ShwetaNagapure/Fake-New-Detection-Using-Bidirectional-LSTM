<h1>Fake News Classification using Bidirectional LSTM</h1>
<h4>Introduction</h4>
<p>This project implements a fake news classification model using Bidirectional Long Short-Term Memory (LSTM) neural networks. The model is trained to classify news articles as either real or fake based on their textual content. It utilizes Natural Language Processing (NLP) techniques for text preprocessing and deep learning for classification.</p>
<h4>Dataset</h4>
<p>The dataset used for training the model is the "WELFake Dataset", which contains news articles labeled as real or fake. The dataset is loaded from a CSV file (WELFake_Dataset.csv.zip). The dataset is preprocessed to remove null values and duplicated entries.</p>
<h4>Data Preprocessing</h4>
<pre>
1] Text Cleaning: The text data is cleaned by removing non-alphabetic characters and converting text to lowercase.
2] Tokenization: The cleaned text is tokenized into words.
3] Stopword Removal and Stemming: Stopwords (common words like "the", "is", "and") are removed, and words are stemmed to their root form.
4] Word Embedding: Words are converted into one-hot representations with a vocabulary size of 5000. These representations are then padded to ensure uniform length.
</pre>
<h4>Model Architecture</h4>
<pre>
  <h6>The model architecture consists of:</h6>
1] Embedding Layer: Maps the one-hot encoded words to dense vectors of fixed size.
2] Bidirectional LSTM Layer: Processes the sequence of word embeddings in both forward and backward directions to capture context from both past and future words.
3] Dropout Layer: Regularizes the model to prevent overfitting.
4] Dense Layer: Performs binary classification using a sigmoid activation function
</pre>
<h4>Training</h4>
<p>The model is trained using the Adam optimizer and binary cross-entropy loss function. It is trained for 10 epochs with a batch size of 64.</p>
<h4>Evaluation </h4
<p>The model's performance was assessed using accuracy as the evaluation metric. With an accuracy score of 88% on the test set.</p>
<h4>Requirements</h4>
<pre>
1] Python 3.x
2] Libraries: pandas, numpy, matplotlib, seaborn, nltk, regex, scikit-learn, tensorflow, wordcloud, gradio
</pre>
<h4>Usage</h4>
<pre>
Clone the repository or download the notebook file.
Ensure all dependencies are installed (pip install -r requirements.txt).
Run the notebook (fake_new_classification_using_bidirectional_lstm.ipynb) to train the model.
After training, use the Gradio interface to interact with the model and make predictions on new news articles.
</pre>
<h4>Author</h4>
<p>-SHWETA NAGAPURE</p>
