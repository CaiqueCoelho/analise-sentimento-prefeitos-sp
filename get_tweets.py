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

stop_words = {"de","a","o","que","e","do","da", "das", "em","um","para","é","com","não","uma","os","no","se","na","por","mais","as","dos",
"como","mas","foi","ao","ele","das","tem","à","seu","sua","ou","ser","quando","muito","há","nos","já","está","eu","também","só","pelo",
"pela","até","isso","ela","entre","era","depois","sem","mesmo","aos","ter","seus","quem","nas","me","esse","eles","estão","você","tinha",
"foram","essa","num","nem","suas","meu","às","minha","têm","numa","pelos","elas","havia","seja","qual","será","nós","tenho","lhe","deles",
"essas","esses","pelas","este","fosse","dele","tu","te","vocês","vos","lhes","meus","minhas","teu","tua","teus","tuas","nosso","nossa",
"nossos","nossas","dela","delas","esta","estes","estas","aquele","aquela","aqueles","aquelas","isto","aquilo","estou","está","estamos",
"estão","estive","esteve","estivemos","estiveram","estava","estávamos","estavam","estivera","estivéramos","esteja","estejamos","estejam",
"estivesse","estivéssemos","estivessem","estiver","estivermos","estiverem","hei","há","havemos","hão","houve","houvemos","houveram",
"houvera","houvéramos","haja","hajamos","hajam","houvesse","houvéssemos", "haveria", "houvessem","houver","houvermos","houverem","houverei","houverá",
"houveremos","houverão","houveria","houveríamos","houveriam","sou","somos","são","era","éramos","eram","fui","foi","fomos","foram","fora",
"fôramos","seja","sejamos","sejam","fosse","fôssemos","fossem","for","formos","forem","serei","será","seremos","serão","seria","seríamos",
"seriam","tenho","tem","temos","tém","tinha","tínhamos","tinham","tive","teve","tivemos","tiveram","tivera","tivéramos","tenha","tenhamos",
"tenham","tivesse","tivéssemos","tivessem","tiver","tivermos","tiverem","terei","terá","teremos","terão","teria","teríamos","teriam", "vai",
"vou", "tão", "alguma", "interesse", "ter", "caso", "abaixo", "animais", "ainda", "outras", "etc.", "em", "a", "b", "c", 
"d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "x", "y", "z", ".", ",", "-", "#", 
"black", "#blackfraude", "pra", "preço", "#blackfriday", "friday", "comprar", "tá", "tudo", "promoção", "agora", "metade"
"blackfraude", "desconto", "hoje", "lojas", "#blackfridaybrasil", "preços", "dia", "brasil", "nada", "nessa", "ano", 
"comprei", "aqui", "r$", "reais", "tava", "gente", "ver", "produtos", "todo", "produto", "antes", "loja", "pq", "aí", "fazer",
"ta", "semana", "vc", "coisa", "1", "brasileiro", "povo", "?", "to", "queria", "2", "ontem", "compra", "tô", "vi",
"sempre", "sendo", "quero", "bem", "coisas", "dias", "bom", "nunca", "vcs", "pessoas", "mês", "pode", "3", "pro", "faz", "olha",
"nao", "menos", "friday", "lá", "assim", "real", "melhor", "sabe", "realmente", "parabéns", "friday", "!", "fazendo", "vamos", "hj", "dessa", 
"boa", "ai", "fica", "dar", "quer", "porque", "sei", "vendo", "alguém", "dá", "sobre", "onde", "verdade", "única", "começou", "dizer", 
"apenas", "ficar", "uns", "vão", "acho", "kkkk", "vez", "desde", "kkk", "algo", "normal", "dois", "atrás", "mundo", "chega", 
"pois", "10", "desse", "sim", "...", "6", "galera", "nesse", "duas", "né", "alguns", "blackfraude", "algumas", "ia", "fazem"}
 

tweets = []
dictWords = {}

consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

class TwitterClient(object):
    def __init__(self):
        try:
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            self.auth.set_access_token(access_token, access_token_secret)
            self.api = tweepy.API(self.auth, wait_on_rate_limit=True)
        except:
            print("Error: Authentication Failed")


    def getWords(self):

        for tweet in tweets:
            words = tweet.split()

            for word in words:
                word = word.lower()
                if(word not in stop_words):
                    existe = False
                    if(word in dictWords):
                        existe = True      

                    if(not existe):
                        dictWords[word] = 1
                    if(existe):
                        count = dictWords[word]
                        count += 1
                        dictWords[word] = count

    def give_emoji_free_text(self, text):
        allchars = [str for str in text]
        emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
        clean_text = ' '.join([str for str in text.split() if not any(i in str for i in emoji_list)])
    
        return clean_text
 
    def clean_tweet(self, tweet):

        #text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", tweet).split())

        text = text.replace(',', '')

        text = tweet
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
      
 
    def get_tweets(self, query, count, tweet_mode, include_entities, truncated, data_since):
        fetched_tweets = []
        try:
            now = datetime.now()
            csvFile = open(query + '_' + str(now) + '.csv', 'w', encoding='utf-8')
            # use csv Writer
            csvWriter = csv.writer(csvFile)

            #fetched_tweets = self.api.search(q = query, lang='pt', count = count, tweet_mode = 'extended', data_since=data_since, until='2019-12-01')
            for tweet in tweepy.Cursor(self.api.search, lang='pt', q=query+ ' -RT', since_id=2019-11-27, until='2019-12-01', tweet_mode = 'extended').items():
                tweet = json.dumps(tweet._json)
                tweet = json.loads(tweet)
                fetched_tweets.append(tweet)
                csvWriter.writerow([tweet['created_at'],
                    tweet['id'],
                    tweet['full_text'],
                    ])

            for tweet in fetched_tweets:
                print(tweet)
                if('full_text' in tweet):
                    clean_tweet = self.clean_tweet(tweet['full_text'])
         
                    # appending parsed tweet to tweets list
                    #if(TextBlob(parsed_tweet['text']).detect_language() == 'pt'):
                    if tweet['retweet_count'] > 0:
                        # if tweet has retweets, ensure that it is appended only once
                        if clean_tweet not in tweets:
                            tweets.append(clean_tweet)
                    else:
                        tweets.append(clean_tweet)
 
            return tweets
 
        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))
 
def main():
    
    api = TwitterClient()
    tweets = []

    query = 'BlackFraude'
    since_id = 2019-11-28
    tweets_query = api.get_tweets(query = query, count = 10, tweet_mode='extended', include_entities=True, truncated=False, data_since='2019-11-28')
    #tweets_query = tweepy.Cursor(api.search, q=query+ ' -RT', since_id=2019-11-28)
    tweets = tweets_query

    # printing first 100 tweets
    #print("\n\ntweets:")
    #for tweet in tweets[:99]:
    #     print(tweet['text'])
 
    api.getWords()

    dictWordsSorted = sorted(dictWords.items(), key=lambda x:x[1])
    print(dictWordsSorted)

    with open('words.csv', 'w') as f:  # Just use 'w' mode in 3.x
        for word in dictWordsSorted:
            f.write("%s,%s\n"%(word[0],word[1]))

    print(len(tweets))
    f.close()
 
if __name__ == "__main__":

    #sys.stdout=open("out.txt","w")

    # calling main function
    main()
