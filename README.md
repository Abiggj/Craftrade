
# CrafTrade

CraftTrade is a sophisticated simulation tool designed to operate on financial OHLCV (Open, High, Low, Close, Volume) time series data and corresponding news headlines using neural networks, natural language processing (NLP), and Long Short-Term Memory (LSTM) networks. The primary objective of CraftTrade is to analyze and simulate the impact of news on stock prices, providing valuable insights for trading strategies.


## Introduction

In today's fast-paced financial markets, traders and analysts are constantly seeking tools that can provide deeper insights and enhance their trading strategies. Stock prices are influenced by a variety of factors, including market data and news events. CraftTrade is an advanced simulation tool designed to help users analyze the interplay between these factors by leveraging state-of-the-art technologies in neural networks, NLP, and LSTM networks.

CraftTrade operates on financial OHLCV time series data and corresponding news headlines over the same period. By integrating numerical time series data with sentiment and context extracted from news headlines, CraftTrade provides a comprehensive analysis of market conditions. This dual analysis approach allows users to simulate potential market scenarios and evaluate the impact of news on stock prices.

The neural network models in CraftTrade are trained to recognize patterns and relationships within the OHLCV data, while the NLP models process and interpret the sentiment and relevance of news headlines. LSTM networks are particularly well-suited for handling time series data, making them ideal for capturing temporal dependencies in stock price movements.

Whether you're a seasoned trader looking to refine your strategies or a data scientist exploring financial markets, CraftTrade offers a powerful platform to simulate and analyze market dynamics. The insights gained from CraftTrade can help inform decision-making, reduce risk, and potentially enhance trading performance.









## Features

- **Integrated Analysis**: Combines OHLCV data and news headlines for comprehensive analysis.
- **Neural Networks**: Utilizes advanced neural networks for predictive modeling.
- **Natural Language Processing**: Processes and analyzes news headlines to gauge sentiment and potential market impact.
- **LSTM Networks**: Leverages LSTM networks to handle time series data effectively.
- **Simulation**: Provides a simulation environment to test trading strategies based on historical data and news.

## Dataset Description

The dataset provided contains daily financial data for multiple stocks alongside corresponding news headlines and sentiment scores. This data spans a specific period and includes key financial indicators (OHLCV: Open, High, Low, Close, Volume) for various companies as well as a comprehensive list of news headlines that could potentially influence the stock prices. Below is a detailed description of the dataset's structure and the information it contains.

### Columns ###
- Date: The date of the recorded data in YYYY-MM-DD format.

#### Financial Data for Stocks ####

For each stock listed below, the dataset includes the following indicators:

- *Open* : The opening price of the stock on that date.
- *High* : The highest price of the stock on that date.
- *Low* : The lowest price of the stock on that date.
- *Close* : The closing price of the stock on that date.
- *Adj Close* : The adjusted closing price, which accounts for any corporate actions like splits or dividends.
- *Volume* : The number of shares traded on that date.

#### The stocks included are: ####

- *Canara Bank*
- *Axis Bank*
- *Punjab National Bank*
- *ICICI Bank*
- *HDFC Bank*
- *Bank of India*
- *State Bank of India*
- *IDBI Bank*
- *Bank of Baroda*
- *Nifty 50 Index*

#### News Data ####

- *news* : A collection of news headlines relevant to the stock market and the companies included in the dataset. This textual data provides context on market conditions, events, and other factors that might affect stock prices.
- *sentiment*: A numerical sentiment score associated with the news headlines, indicating the sentiment (positive, negative, or neutral) expressed in the news articles.
## Project Architecture
### 1. Data Collection

This section outlines the data collection process for both the news articles and the stock data used in this project.

#### News Data Collection

The news articles are collected from the Economic Times website using a web scraping script written in Python with Selenium. Here's a high-level explanation of the steps involved:

- *Setup Selenium WebDriver* : Initialize the Chrome WebDriver with headless options for automated browsing.
   
- *Iterate Over Date Range* : Loop through each month of the specified years (2015-2021) to gather monthly archives of news articles.

- *Extract Links to Daily Archives* : For each month, navigate to the corresponding archive page and extract the links to daily news archives by locating and parsing HTML elements (tables and anchors).

- *Scrape News Articles* : For each daily archive link, navigate to the page and extract news headlines by locating and parsing HTML elements.

- *Save News Data* : Format the news articles along with their corresponding dates and save them to text files in a designated directory.

#### Stock Data Collection

The stock data is collected from Yahoo Finance. The following steps explain the process:

- *Identify Stocks* : Select the list of stocks for which historical data is needed. This includes various stocks such as CAN, AXI, PNB, ICI, HDB, BAN, SBI, IDB, BOB, and N50.

- *Download Historical Data* : Use Yahoo Finance's historical data feature to download OHLCV (Open, High, Low, Close, Volume) data for each stock. This can be done manually by navigating to the Yahoo Finance website, entering the stock symbol, selecting the "Historical Data" tab, setting the date range, and downloading the data as a CSV file.

- *Save and Organize Data* : Save each stock's historical data CSV file in a designated directory for further preprocessing.


### 2. Data Preprocessing ###

**Step 1** : Setup and File Reading

- *Libraries and Data Files*: Import necessary libraries such as pandas, os nltk, and re. Define paths to directories containing news and stock data files.
- *NLP Resources*: Download required NLTK resources (punkt, stopwords, and vader_lexicon) for text processing and sentiment analysis.
- *File Paths*: Generate lists of file paths for news and stock data files in their respective directories.

**Step 2**: Text Cleaning and Sentiment Analysis
- *Text Cleaning Function* : Define a function clean_text to convert text to lowercase, remove digits and punctuation.

- *Stopwords Removal Function* : Define a function remove_stopwords to remove common English stopwords from the text.

- *Sentiment Analysis Function* : Define a function sentiment_score to compute sentiment scores using the SentimentIntensityAnalyzer from NLTK.

- *Processing News Files* : Iterate through the news files, read each file, clean the text, remove stopwords, compute sentiment scores, and store the results in lists (dates, news, sentiment_scores).

- *Save Cleaned News Data* : Create a DataFrame from the processed news data and save it as news.csv.

**Step 3**: Merging Stock Data
- *Reading Stock Files* : Iterate through stock data files, read each file into a DataFrame, and append the stock ticker symbol as a suffix to each column name (except the 'Date' column).

- *Merge Stock DataFrames* : Merge all stock DataFrames on the 'Date' column using an outer join, sort the merged DataFrame by date, and reset the index.

- *Save Merged Stock Data*: Save the merged stock data as shares.csv.

**Step 4**: Final Merging and Cleaning
- *Read Saved CSVs* : Read news.csv and shares.csv into DataFrames.

- *Date Formatting* : Convert 'Date' columns in both DataFrames to a consistent format (YYYY-MM-DD).

- *Merge News and Stock Data* : Merge the news and stock DataFrames on the 'Date' column using a left join.

- *Sort and Clean Merged Data* : Sort the final merged DataFrame by date, drop any rows with missing values, reset the index, and save the cleaned, merged data as data.csv.

### 3. Adversarial Network ###

* **Generator**:

    * *Embedding Layer* : Converts the input news text into a dense vector representation.
    * *LSTM Layer* : Processes the sequential data (news text) to capture temporal dependencies and generate features.
    * *Dense Layer* : Outputs synthetic OHLCV data with dimensions matching the real financial data.

* **Discriminator**:

    * *Dense Layers* : Takes in the OHLCV data (either real or synthetic) and processes it through multiple layers to determine the authenticity.
    * *Output Layer* : Produces a probability score indicating whether the input data is real or generated.

* **GAN Workflow**
    * *Tokenization and Padding* : The news headlines are tokenized and padded to create uniform input sequences for the Generator.
    * *Generator Training* : The Generator creates synthetic OHLCV data from the news text input.
    * *Discriminator Training* : The Discriminator evaluates both real and synthetic OHLCV data, attempting to distinguish between the two.
    * *Combined Model Training* : The Generator and Discriminator are trained together in an adversarial manner. The Generator aims to improve its ability to create realistic synthetic data, while the Discriminator aims to enhance its ability to detect fake data.

* **Training Process**

    * *Data Preparation* : The dataset consists of news headlines and corresponding OHLCV data. The news text is preprocessed and the financial data is extracted.
    * *Batch Processing* : During each training epoch, batches of real and synthetic OHLCV data are used to train the Discriminator. The Generator is then trained using the feedback from the Discriminator.
    * *Loss Calculation* : The loss for both the Generator and the Discriminator is calculated using binary cross-entropy. The Discriminator's loss reflects its ability to correctly identify real versus fake data, while the Generator's loss reflects its ability to produce convincing synthetic data.
    * *Training Loop* : This process is repeated for multiple epochs, allowing the Generator and Discriminator to iteratively improve.

* **Synthetic Data Generation**
    
    Once the model is trained, the Generator can be used to produce synthetic OHLCV data from new news headlines. This involves feeding the text input into the Generator, which outputs the corresponding financial data predictions.
## FAQ ##

### General Questions

**Q1: What is CraftTrade?**  
CraftTrade is a simulation tool that leverages adversarial neural networks to generate synthetic financial OHLCV (Open, High, Low, Close, Volume) data based on news headlines. It combines natural language processing (NLP) and time series analysis to predict and simulate stock market behavior.

**Q2: Who can benefit from CraftTrade?**  
CraftTrade is designed for financial analysts, traders, data scientists, and researchers who want to explore the relationship between news sentiment and stock market movements. It can also be used by educational institutions for teaching purposes.

**Q3: What makes CraftTrade unique?**  
CraftTrade uniquely combines textual data from news headlines with numerical stock market data using a Generative Adversarial Network (GAN). This allows for sophisticated simulations and predictions based on real-world events and market sentiment.

### Data and Preprocessing

**Q4: What type of data does CraftTrade use?**  
CraftTrade uses two primary types of data: financial OHLCV data and news headlines. The OHLCV data is sourced from Yahoo Finance, and the news headlines are collected from various reputable financial news websites.

**Q5: How is the news data collected?**  
The news data is collected using web scraping techniques with Selenium. The script navigates to financial news archives, extracts headlines, and saves them along with their corresponding dates.

**Q6: How is the data preprocessed?**  
The data preprocessing involves several steps:
- **Text Cleaning**: Removing numbers, punctuation, and converting text to lowercase.
- **Stopword Removal**: Eliminating common stopwords to focus on meaningful words.
- **Sentiment Analysis**: Using the VADER sentiment analysis tool to assign a sentiment score to each news headline.
- **Merging Data**: Combining the preprocessed news data with the OHLCV data based on matching dates.

### Model Architecture

**Q7: What is the architecture of the adversarial network used in CraftTrade?**  
The adversarial network consists of two main components:
- **Generator**: Generates synthetic OHLCV data from news headlines. It uses an Embedding layer, an LSTM layer, and a Dense layer.
- **Discriminator**: Distinguishes between real and synthetic OHLCV data. It uses several Dense layers and an output layer with a sigmoid activation function.

**Q8: How does the training process work?**  
The training process involves:
- **Tokenizing and padding** the news headlines.
- **Training the Discriminator** with both real and synthetic data.
- **Training the Generator** to produce more realistic synthetic data using feedback from the Discriminator.
- **Iterating this process** for multiple epochs to improve the performance of both the Generator and Discriminator.

### Usage and Applications

**Q9: How can I generate synthetic OHLCV data using CraftTrade?**  
Once the model is trained, you can input new news headlines into the Generator. The Generator will produce synthetic OHLCV data that reflects the predicted market response to the given news.

**Q10: Can CraftTrade be used for real-time predictions?**  
While CraftTrade can generate predictions based on news headlines, it is primarily a simulation tool. For real-time predictions, additional components for continuous data collection and model updating would be required.

### Technical Questions

**Q11: What programming languages and libraries are used in CraftTrade?**  
CraftTrade is implemented in Python and uses libraries such as Pandas, Numpy, TensorFlow, NLTK, and Selenium.

**Q12: How can I contribute to the CraftTrade project?**  
Contributions are welcome! You can contribute by improving the model, adding new features, or enhancing the data collection and preprocessing scripts. Please visit the project's GitHub repository for more information on how to contribute.

**Q13: What are the system requirements for running CraftTrade?**  
CraftTrade requires a machine with Python installed and access to internet for downloading necessary libraries and data. For optimal performance, a machine with a modern CPU, sufficient RAM, and a GPU (for training the neural network) is recommended.
