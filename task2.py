import pandas as pd
from io import StringIO
from textblob import TextBlob
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Download required NLTK data
nltk.download('vader_lexicon')

# Initialize VADER Sentiment Analyzer
sid = SentimentIntensityAnalyzer()

# CSV data with quotes for proper parsing
csv_data = '''text
"I love the new design of your website!"
"The flight was delayed and it was so frustrating."
"Looking forward to the weekend :)"
"The customer service was okay, nothing special."
"I'm not sure how I feel about this new update."
'''

# Load tweets into DataFrame
tweets_df = pd.read_csv(StringIO(csv_data))

# TextBlob sentiment analysis
tweets_df['TextBlob_Polarity'] = tweets_df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
tweets_df['TextBlob_Subjectivity'] = tweets_df['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

# VADER sentiment classification function
def vader_sentiment(text):
    score = sid.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

tweets_df['VADER_Sentiment'] = tweets_df['text'].apply(vader_sentiment)
vader_counts = tweets_df['VADER_Sentiment'].value_counts()

# Start plotting with matplotlib
fig, axs = plt.subplots(3, 2, figsize=(14, 14))
fig.suptitle('Sentiment Analysis Dashboard', fontsize=18, fontweight='bold')

# 1. Bar chart: TextBlob polarity per tweet
axs[0, 0].bar(tweets_df.index + 1, tweets_df['TextBlob_Polarity'], color='skyblue')
axs[0, 0].axhline(0, color='gray', linewidth=0.8)
axs[0, 0].set_title('TextBlob Polarity per Tweet')
axs[0, 0].set_xlabel('Tweet Number')
axs[0, 0].set_ylabel('Polarity (-1 to +1)')
axs[0, 0].set_xticks(tweets_df.index + 1)
axs[0, 0].set_ylim(-1, 1)

# 2. Bar chart: TextBlob subjectivity per tweet
axs[0, 1].bar(tweets_df.index + 1, tweets_df['TextBlob_Subjectivity'], color='plum')
axs[0, 1].set_title('TextBlob Subjectivity per Tweet')
axs[0, 1].set_xlabel('Tweet Number')
axs[0, 1].set_ylabel('Subjectivity (0 to 1)')
axs[0, 1].set_xticks(tweets_df.index + 1)
axs[0, 1].set_ylim(0, 1)

# 3. Pie chart: VADER sentiment distribution
colors = ['green', 'red', 'gold']
axs[1, 0].pie(vader_counts.values, labels=vader_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
axs[1, 0].set_title('VADER Sentiment Distribution')

# 4. Bar chart: Counts of VADER sentiment categories
axs[1, 1].bar(vader_counts.index, vader_counts.values, color=colors)
axs[1, 1].set_title('VADER Sentiment Counts')
axs[1, 1].set_xlabel('Sentiment')
axs[1, 1].set_ylabel('Number of Tweets')

# 5. Histogram: Distribution of TextBlob polarity scores
axs[2, 0].hist(tweets_df['TextBlob_Polarity'], bins=10, color='skyblue', edgecolor='black')
axs[2, 0].set_title('Distribution of TextBlob Polarity Scores')
axs[2, 0].set_xlabel('Polarity')
axs[2, 0].set_ylabel('Frequency')
axs[2, 0].set_xlim(-1, 1)

# 6. Histogram: Distribution of TextBlob subjectivity scores
axs[2, 1].hist(tweets_df['TextBlob_Subjectivity'], bins=10, color='plum', edgecolor='black')
axs[2, 1].set_title('Distribution of TextBlob Subjectivity Scores')
axs[2, 1].set_xlabel('Subjectivity')
axs[2, 1].set_ylabel('Frequency')
axs[2, 1].set_xlim(0, 1)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle space
plt.show()
