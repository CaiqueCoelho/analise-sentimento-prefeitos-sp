import re
import tweepy
import csv
import sys
import emoji
import json
from tweepy import OAuthHandler
from textblob import TextBlob
from mtranslate import translate
from datetime import datetime

stop_words = {"de", "a", "o", "que", "e", "do", "da", "das", "em", "um", "para", "é", "com", "não", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos",
              "como", "mas", "foi", "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser", "quando", "muito", "há", "nos", "já", "está", "eu", "também", "só", "pelo",
              "pela", "até", "isso", "ela", "entre", "era", "depois", "sem", "mesmo", "aos", "ter", "seus", "quem", "nas", "me", "esse", "eles", "estão", "você", "tinha",
              "foram", "essa", "num", "nem", "suas", "meu", "às", "minha", "têm", "numa", "pelos", "elas", "havia", "seja", "qual", "será", "nós", "tenho", "lhe", "deles",
              "essas", "esses", "pelas", "este", "fosse", "dele", "tu", "te", "vocês", "vos", "lhes", "meus", "minhas", "teu", "tua", "teus", "tuas", "nosso", "nossa",
              "nossos", "nossas", "dela", "delas", "esta", "estes", "estas", "aquele", "aquela", "aqueles", "aquelas", "isto", "aquilo", "estou", "está", "estamos",
              "estão", "estive", "esteve", "estivemos", "estiveram", "estava", "estávamos", "estavam", "estivera", "estivéramos", "esteja", "estejamos", "estejam",
              "estivesse", "estivéssemos", "estivessem", "estiver", "estivermos", "estiverem", "hei", "há", "havemos", "hão", "houve", "houvemos", "houveram",
              "houvera", "houvéramos", "haja", "hajamos", "hajam", "houvesse", "houvéssemos", "houvessem", "houver", "houvermos", "houverem", "houverei", "houverá",
              "houveremos", "houverão", "houveria", "houveríamos", "houveriam", "sou", "somos", "são", "era", "éramos", "eram", "fui", "foi", "fomos", "foram", "fora",
              "fôramos", "seja", "sejamos", "sejam", "fosse", "fôssemos", "fossem", "for", "formos", "forem", "serei", "será", "seremos", "serão", "seria", "seríamos",
              "seriam", "tenho", "tem", "temos", "tém", "tinha", "tínhamos", "tinham", "tive", "teve", "tivemos", "tiveram", "tivera", "tivéramos", "tenha", "tenhamos",
              "tenham", "tivesse", "tivéssemos", "tivessem", "tiver", "tivermos", "tiverem", "terei", "terá", "teremos", "terão", "teria", "teríamos", "teriam", "vai",
              "vou", "tão", "alguma", "interesse", "ter", "caso", "abaixo", "animais", "ainda", "outras", "etc.", "em", "a", "b", "c",
              "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "x", "y", "z"}


tweetsPositives = []
tweetsNegatives = []
dictPositive = {}
dictNegatives = {}

tweets = []
tweetsText = []
dictWords = {}

query = "Arthur do Val"


class TwitterClient(object):
    '''
    Generic Twitter Class for sentiment analysis.
    '''

    def __init__(self):
        '''
        Class constructor or initialization method.
        '''
        # keys and tokens from the Twitter Dev Console
        consumer_key = ''
        consumer_secret = ''
        access_token = ''
        access_token_secret = ''

        # attempt authentication
        try:
            # create OAuthHandler object
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            # set access token and secret
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")

    def getWordNegatives(self):

        for tweet in tweetsNegatives:
            words = tweet.split()

            for word in words:
                if(word not in stop_words):
                    existe = False
                    if(word in dictNegatives):
                        existe = True

                    if(not existe):
                        dictNegatives[word] = 1
                    if(existe):
                        count = dictNegatives[word]
                        count += 1
                        dictNegatives[word] = count

    def getWordPositives(self):

        for tweet in tweetsPositives:
            words = tweet.split()

            for word in words:
                word = word.lower()
                if(word not in stop_words):
                    existe = False
                    if(word in dictPositive):
                        existe = True

                    if(not existe):
                        dictPositive[word] = 1
                    if(existe):
                        count = dictPositive[word]
                        count += 1
                        dictPositive[word] = count

    def give_emoji_free_text(self, text):
        allchars = [str for str in text]
        emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
        clean_text = ' '.join(
            [str for str in text.split() if not any(i in str for i in emoji_list)])

        return clean_text

    def clean_tweet(self, tweet):

        # Removing á and é from text, why? Study this in future
        '''
        Utility function to clean tweet text by removing links, special characters
        using simple regex statements.
        '''

        text = ' '.join(
            re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", tweet).split())

        '''
        emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F1F2-\U0001F1F4"  # Macau flag
        u"\U0001F1E6-\U0001F1FF"  # flags
        u"\U0001F600-\U0001F64F"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U0001F1F2"
        u"\U0001F1F4"
        u"\U0001F620"
        u"\u200d"
        u"\u2640-\u2642"
        "]+", flags=re.UNICODE)

        text = emoji_pattern.sub(r'', text)
        '''

        text = self.give_emoji_free_text(text)

        return text

    def get_tweet_sentiment(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text

        origin = TextBlob(self.clean_tweet(tweet))
        traducao = None

        try:
            if origin.detect_language() != 'en':
                traducao = TextBlob(str(origin.translate(to='en')))
            print(origin)

        except Exception as e:
            print("WARNING: TRANSLATION ERROR OR TRANSLATION EXCEPTION")
            print(e)
            return

        # print("AAAA")

        # set sentiment
        if traducao.sentiment.polarity > 0:
            tweetsPositives.append(origin)
            return 'positive'
        elif traducao.sentiment.polarity == 0:
            return 'neutral'
        else:
            tweetsNegatives.append(origin)
            return 'negative'

    def get_tweets(self, query, count, tweet_mode, include_entities, truncated, data_since):
        '''
        Main function to fetch tweets and parse them.
        '''
        # empty list to store parsed tweets
        fetched_tweets = []
        try:
            now = datetime.now()
            csvFile = open(query + '_' + str(now) +
                           '.csv', 'w', encoding='utf-8')
            # use csv Writer
            csvWriter = csv.writer(csvFile)
            # call twitter api to fetch tweets
            for tweet in tweepy.Cursor(self.api.search, lang='pt', q=query + ' -RT', since_id=data_since, until='2020-11-14', tweet_mode='extended').items():
                tweet = json.dumps(tweet._json)
                tweet = json.loads(tweet)
                fetched_tweets.append(tweet)
                csvWriter.writerow([tweet['created_at'],
                                    tweet['id'],
                                    tweet['full_text'],
                                    ])

                print(tweet)
                parsed_tweet = {}
                if('full_text' in tweet):
                    clean_tweet = self.clean_tweet(tweet['full_text'])

                    # appending parsed tweet to tweets list
                    # if(TextBlob(parsed_tweet['text']).detect_language() == 'pt'):
                    parsed_tweet['text'] = clean_tweet
                    #parsed_tweet['sentiment'] = self.get_tweet_sentiment(
                    #    clean_tweet)

                    print(parsed_tweet)

                    if tweet['retweet_count'] > 0:
                        # if tweet has retweets, ensure that it is appended only once
                        if clean_tweet not in tweetsText:
                            tweetsText.append(clean_tweet)
                            tweets.append(parsed_tweet)
                            print("Entrou")
                    else:
                        if clean_tweet not in tweetsText:
                            tweetsText.append(clean_tweet)
                            tweets.append(parsed_tweet)
                            print("entrou")

            '''
            # parsing tweets one by one
            for tweet in fetched_tweets:
                # empty dictionary to store required params of a tweet
                parsed_tweet = {}

                # saving text of tweet
                #parsed_tweet['text'] = tweet.text

                if(hasattr(tweet, 'retweeted_status')):

                    parsed_tweet['text'] = tweet.retweeted_status.full_text
                    parsed_tweet['sentiment'] = self.get_tweet_sentiment(
                        tweet.retweeted_status.full_text)
                else:
                    parsed_tweet['text'] = tweet.full_text

                    # saving sentiment of tweet
                    parsed_tweet['sentiment'] = self.get_tweet_sentiment(
                        tweet.full_text)

                    # appending parsed tweet to tweets list
                if tweet.retweet_count > 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)
            '''

            # return parsed tweets
            return tweets

        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))


def main():
    # creating object of TwitterClient Class
    api = TwitterClient()

    # calling function to get tweets
    tweets = api.get_tweets(query=query, count=200, tweet_mode='extended',
                            include_entities=True, truncated=False, data_since='2020-10-01')

    print(tweets)
    #print(type(tweets))

    '''
    # picking positive tweets from tweets
    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
    # percentage of positive tweets
    print("Positive tweets percentage: {} %".format(
        100*len(ptweets)/len(tweets)))
    # picking negative tweets from tweets
    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    # percentage of negative tweets
    print("Negative tweets percentage: {} %".format(
        100*len(ntweets)/len(tweets)))
    # percentage of neutral tweets
    print("Neutral tweets percentage: {} %".format(
        100*(len(tweets) - len(ntweets) - len(ptweets))/len(tweets)))

    api.getWordPositives()
    api.getWordNegatives()

    print(dictPositive)

    with open('positive_words_boulos.csv', 'w') as f:  # Just use 'w' mode in 3.x
        for key in dictPositive.keys():
            f.write("%s,%s\n" % (key, dictPositive[key]))

    with open('negative_words_boulos.csv', 'w') as f:  # Just use 'w' mode in 3.x
        for key in dictNegatives.keys():
            f.write("%s,%s\n" % (key, dictNegatives[key]))

    # printing first 5 positive tweets
    print("\n\nPositive tweets:")
    for tweet in ptweets[:100]:
        print(tweet['text'])

    # printing first 5 negative tweets
    print("\n\nNegative tweets:")
    for tweet in ntweets[:100]:
        print(tweet['text'])
    

    f.close()
    '''

if __name__ == "__main__":

    #sys.stdout = open("out_candidate_boulos.txt", "w")

    # calling main function
    main()
