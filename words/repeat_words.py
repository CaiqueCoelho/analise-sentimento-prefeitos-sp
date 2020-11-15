import re
import sys
import csv

read = "boulos_negative_words"

#sys.stdout=open(read + ".txt","w")

def carregar_acessos():

	X = []
	Y = []

	arquivo = open(read + ".csv", 'r')
	leitor = csv.reader(arquivo)
	

	for word, times in leitor:

		X.append(str(word))
		Y.append(int(times))

	return X, Y

# Transforma de data_frames para arrays
words, times = carregar_acessos()

list_words_repetead = ""

index = 0
for word in words:
	for i in range(times[index]):
		list_words_repetead = list_words_repetead + word +", "
	index+=1 

print(list_words_repetead)
