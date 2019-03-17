from wordcloud import WordCloud, STOPWORDS 
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
#%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

##############################################################################################################
'''(1) Preliminary Text Analysis'''
##############################################################################################################
def wordcloud_draw(text, color = 'darkBlue'):
    words = " ".join([word for word in text.split()])
    # basic construction of wordcloud if words in text are clean text
    if (not words.startswith("http")):
        wordcloud = WordCloud(stopwords=STOPWORDS,background_color=color,width=2500,height=2000).generate(words)
    plt.imshow(wordcloud, cmap=plt.cm.gray, interpolation='bilinear')
    plt.axis('off')
    plt.title("Most Common Words in Text")
    plt.show()
##############################################################################################################
# (2) Nuanced Vader Analysis
##############################################################################################################
nltk.download('vader_lexicon')
# prints polarity scores using nl toolkit vader_lexicon
def sentimentAnaly(text):
    sentiment_analy = SentimentIntensityAnalyzer()
    for row in range(len(text)):
        sentences = text[row]
        for sentence in sentences:
            positivity = sentiment_analy.polarity_scores(sentence)
            for k in sorted(positivity):
                print('{0}: {1}, '.format(k, positivity[k], end=''))
