import csv
import re

candidate = 'russomanno'

def separando():

	arquivo = open(candidate+'.csv', 'r')
	leitor = csv.reader(arquivo)

	next(leitor)

	i = 1

	#for Tarefa, Estimativa in leitor:
	for id, tweet, sentiment, tweet_translated in leitor:

		nome_arquivo = "tweets_"+candidate+"_" + str(i) + ".txt"

		dado = str(tweet)

		dado = dado.lower()

		#dado = re.sub('http://\S+|https://\S+', '', dado)
		#OR 
		#dado = re.sub('http[s]?://\S+', '', dado)

		#stop_numbers = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
		#for number in stop_numbers:
		#	dado = dado.replace(number, " ")

		dado = dado.replace("r$", " ")
		dado = dado.replace(",", " ")
		dado = dado.replace(".", " ")
		dado = dado.replace("-", " ")
		dado = dado.replace("'", " ")
		dado = dado.replace("@", " ")
		dado = dado.replace("#", " ")
		dado = dado.replace("$", " ")
		dado = dado.replace("%", " ")
		dado = dado.replace("&", " ")
		dado = dado.replace("*", " ")
		dado = dado.replace("(", " ")
		dado = dado.replace(")", " ")
		dado = dado.replace("_", " ")
		dado = dado.replace("<", " ")
		dado = dado.replace(">", " ")
		dado = dado.replace("{", " ")
		dado = dado.replace("}", " ")
		dado = dado.replace("[", " ")
		dado = dado.replace("]", " ")
		dado = dado.replace("\\", " ")
		dado = dado.replace("/", " ")
		dado = dado.replace("\n", " ")
		dado = dado.replace("\t", " ")
		dado = dado.replace("...", " ")
		dado = dado.replace(":", " ")
		dado = dado.replace(";", " ")
		dado = dado.replace("!", " ")
		dado = dado.replace("?", " ")
		dado = dado.replace(".", " ")
		dado = dado.replace("+", " ")

		print(dado)

		#contando palavras na tarefa
		dadosQuebrados = dado.split(' ')

		numeroPalavras = 0
		for palavra in dadosQuebrados:
			numeroPalavras += 1

		#if(numeroPalavras >= 10):

		file = open(nome_arquivo, "w") 
		 
		file.write(dado) 
		file.close() 

		i = i + 1

separando()
