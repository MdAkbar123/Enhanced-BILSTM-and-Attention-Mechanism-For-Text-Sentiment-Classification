
# Enhanced BILSTM and Attention Mechanism for Text Sentiment Classification

This repository contains an implementation of sentiment analysis using an enhanced Bidirectional Long Short-Term Memory (BiLSTM) network, combined with an attention mechanism for improving text sentiment classification. The model is designed to classify movie reviews or other text data into multiple sentiment categories such as positive, neutral, or negative.

## Project Overview

The project aims to perform sentiment analysis using state-of-the-art deep learning techniques, focusing on enhancing the BiLSTM architecture with attention mechanisms to improve classification accuracy. The model is trained on a large dataset and can be used for text sentiment classification tasks across various domains.

### Features

- **BiLSTM Network:** Utilizes a Bidirectional LSTM to capture both forward and backward dependencies in the text.
- **Attention Mechanism:** Enhances the model by allowing it to focus on important words or phrases that contribute most to sentiment classification.
- **GloVe Embeddings:** Pre-trained word embeddings are used to initialize the word vectors.
- **Flask Web Application:** A lightweight web app to serve the model for sentiment predictions.

## Installation

### Requirements

To get started with the project, clone the repository and install the required dependencies.

1. Clone the repository:

   ```bash
   git clone https://github.com/MdAkbar123/Enhanced-BILSTM-and-Attention-Mechanism-For-Text-Sentiment-Classification.git
````

2. Navigate to the project directory:

   ```bash
   cd Enhanced-BILSTM-and-Attention-Mechanism-For-Text-Sentiment-Classification
   ```

3. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

### Requirements File

The `requirements.txt` file includes all the necessary dependencies for the project. Some key libraries include:

* `tensorflow`: For building and training the BiLSTM model.
* `flask`: For serving the model in a web app.
* `numpy`, `pandas`: For data manipulation.
* `matplotlib`, `seaborn`: For visualizations.

## Usage

Once the environment is set up, you can run the Flask web application to deploy the sentiment analysis model.

1. Run the app:

   ```bash
   python app.py
   ```

2. Access the web application in your browser by navigating to:

   ```
   http://localhost:5000
   ```

3. The app allows users to input text (such as reviews) and get a sentiment prediction (positive, negative, or neutral).

## Project Structure

The directory structure of the project is as follows:

```
Enhanced-BILSTM-and-Attention-Mechanism-For-Text-Sentiment-Classification/
│
├── Datasets/                       # Dataset files
│   ├── Reviews_100.xlsx            # Example dataset for training
│   └── reviews_with_predictions.csv # Example predictions
│
├── models/                          # Trained model and tokenizer
│   ├── sentiment_model.keras        # Trained BiLSTM + Attention model
│   └── tokenizer.pickle            # Tokenizer used for text preprocessing
│
├── templates/                       # HTML templates for the Flask app
│   ├── index.html                  # Main page
│   ├── login.html                  # Login page (optional)
│   └── styles.css                  # CSS for styling
│
├── app.py                           # Flask web application
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

## Model Details

The model utilizes a **BiLSTM architecture** with an attention mechanism for improved context understanding. The attention layer helps the model to focus on important parts of the input text, thus improving classification performance.

### Training the Model

To train the model, run the following command after setting up the dataset and pre-processing steps:

```bash
python train_model.py
```

This will start training the model on the dataset provided in the `Datasets` folder.

## Contributing

Feel free to fork the repository and contribute improvements or suggestions via pull requests. Make sure to follow the code style and document any changes made.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

---

For further assistance, feel free to open an issue on the repository or reach out through email.

```

You can copy and paste this content into your `README.md` file in your GitHub repository. Customize it as necessary to match your project details and requirements.
```
