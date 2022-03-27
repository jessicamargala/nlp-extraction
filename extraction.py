import spacy
import newsapi
import pickle
import pandas as pd
import string
import matplotlib.pyplot as plt
from collections import Counter
from wordcloud.wordcloud import WordCloud
from newsapi import NewsApiClient
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English

nlp_eng = spacy.load('en_core_web_lg')
newsapi = NewsApiClient(api_key='1d4a690e41694470962b99c2fe0d30b6')


# Tokenizes content to parse for Verbs, Nouns, and Proper Nouns
def get_keywords_eng(text):
    result = []
    pos_tag = ['VERB', 'NOUN', 'PROPN']
    text = nlp_eng(text)
    for token in text:
        # print(token.text)
        if (token.text in nlp_eng.Defaults.stop_words or token.text in string.punctuation):
            continue
        if (token.pos_ in pos_tag):
            result.append(token.text)
    return result

def main(): 
    articles = []
    results = []

    temp = newsapi.get_everything(q='coronavirus', language='en', from_param='2022-02-27',
                            to='2022-03-28', sort_by='relevancy', page_size=100)
    articles.append(temp)

    # Save article data
    filename = 'articlesCOVID.pckl'
    pickle.dump(articles, open(filename, 'wb'))

    # Clean data
    dados = []
    for i, article in enumerate(articles):
        for x in article['articles']:
            title = x.get('title')
            description = x.get('description')
            date = x.get('publishedAt')
            content = x.get('content')
            dados.append({'title':title, 'date':date, 'desc':description, 'content':content})
    
    # Create data frame
    df = pd.DataFrame(dados)
    df = df.dropna()
    df.head()
    df.to_csv('data_set.csv')

    # Find 5 most common words from each article
    for content in df.content.values:
        results.append([('#' + x[0]) for x in Counter(get_keywords_eng(content)).most_common(5)])
    df['keywords'] = results


    # Word Cloud Visualization Steps
    text = []
    for result in results:
        for i in result:
            text.append(i)

    text = (" ").join(text)
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
