# -*- coding: utf-8 -*-
import sys
import codecs
import nltk
import math
from nltk import bigrams
from math import log

#calcola numero totale di token e frasi, la lunghezza media delle frasi in termini di token
def CalcolaLunghezza_MediaFr_Voc(frasi):
    numToken = []
    numFrasi = 0.0
    for frase in frasi:
        #divido la frase in token
        tokens=nltk.word_tokenize(frase)
        #calcolo la lunghezza del corpus in token
        numToken = numToken+tokens
        lunghezzaToken  = len(numToken)
        #registro il numero delle frasi osservate con lo scorrimento del ciclo for
        numFrasi = numFrasi+1           
    #restituisco il risultato
    return lunghezzaToken, numToken, numFrasi, lunghezzaToken/numFrasi

#calcola numero dei caratteri dell'intero corpus, ad eccezione della punteggiatura  
def CalcolaCaratteri(listaToken):
    contaCaratteri = 0.0
    for token in listaToken:
                #registro il numero dei caratteri osservati con lo scorrimento del ciclo for
                contaCaratteri = contaCaratteri+len(token)
    return contaCaratteri

#funzione che conta gli hapax
def hapax (listaToken,vocabolario):
    conta = 0
    for tok in vocabolario:
        frequenzaToken = listaToken.count(tok) #calcolo la frequenza di quel token  in quel corpus 
        if frequenzaToken == 1: #se la frequenza del token è 1 è un hapax 
            conta = conta + 1 #conto quanti hapax trovo
    return conta #restituisco il risultato

#calcola gli incrementi per porzioni di 1000 token del vocabolario e degli hapax
def calcolaVoc_Hapax (listaToken, numToken):
    n = 1000 #contatore per scorrere token
    listaVocabolario = [] #creo lista dove inserire i risultati del vocabolario per le porzioni incrementali
    listaHapax = [] #creo lista dove inserire i risultati degli hapax per le porzioni incrementali
    while n < numToken: #finchè non viene superato il numero di token calcolo il vocabolario e hapax
        vocabolario = set(listaToken[:n])#calcolo vocabolario sui primi n token
        grandezzaVocabolario = len(vocabolario)
        listaVocabolario.append(grandezzaVocabolario)#inserisco i valori ottenuti nella lista   
        numeroHapax = hapax(listaToken[:n],vocabolario) #richiamo la funzione per contare gli hapax
        listaHapax.append(numeroHapax) #inserisco i risultati nella lista
        n += 1000 #aggiorno il contatore
    return listaVocabolario, listaHapax #restituisco i risultati

#calcola il rapporto tra sostantivi e verbi presenti nel corpus
def RapportoSostantiviVerbi(listaToken):
    #prende in input una lista di token ed esegue l'analisi morfo sintattica
    tokensPOS = nltk.pos_tag(listaToken)
    sostantivi = []
    verbi = []
    for (tok,pos) in tokensPOS:     
        if pos in ["NN", "NNS", "NNP", "NNPS"]: #per ogni sostantivo presente nella lista, viene aggiunto un token nella lista sostantivi
            sostantivi.append(tok)
        if pos in ["VB", "VBD","VBG","VBN","VBP","VBZ"]: #per ogni verbo presente nella lista, viene aggiunto un token nella lista verbi
            verbi.append(tok)
            lunghezzaSostantivi = len(sostantivi)
            lunghezzaVerbi = len(verbi)
            #Calcolo il rapporto tra il numero totale di sostantivi e verbi
            Rapporto = float(lunghezzaSostantivi)/float(lunghezzaVerbi)
    return Rapporto

#trovo le 10 PoS più frequenti
def Part_of_Spech(listaToken):
    tokensPOS=nltk.pos_tag(listaToken)
    listaPOS = []
    #scorro i bigrammi (token, PoS)
    for (tok,pos) in tokensPOS:
        #aggiungo tutte le PoS alla lista
        listaPOS.append(pos)
    #calcolo la frequenza delle PoS
    freqPOS=nltk.FreqDist(listaPOS)
    #trovo le 10 PoS più frequenti
    freq10POS=freqPOS.most_common(10)   
    #restituisco il risultato
    return freq10POS

#calcola la probabilità condizionata e la LMI dei 10 bigrammi di PoS più frequenti
def probCondizionata_LMI (listaPOS, bigrammiPOS):   
    listaProbCond = []
    listaLMI = []   
    N = len(listaPOS)    
    Distribuzione_Frequenza = nltk.FreqDist(bigrammiPOS)
    #ciclo per calcolo della probabilità condizionata sui bigrammi di PoS
    for bigramma in Distribuzione_Frequenza:        
        # Frequenza attesa F(a) * F(b)/ N
        PoS1 = listaPOS.count(bigramma[0])
        PoS2 = listaPOS.count(bigramma[1])
        FE = float((PoS1*PoS2)/N)
        #Calcolo Probabilità Condizionata  (F(a) * F(b))/ N
        ProbCond = (FE/N*1.0)*100       
        listaProbCond.append([ProbCond, bigramma])  
        #ordino la lista
        listaProbCondOrd = sorted(listaProbCond,  reverse=True)   
        FO = Distribuzione_Frequenza[bigramma]
        LMI = (FO*1.0)*math.log((FO*1.0)/(FE*1.0), 2) 
        listaLMI.append([LMI, bigramma])
        listaLMIOrd = sorted(listaProbCond, reverse=True)
    #restituisco il risultato
    return listaProbCondOrd[:10], listaLMIOrd[:10]


    

#estraggo i PoS tag 
def estraiPOS (listaToken):
    tokensPOS=nltk.pos_tag(listaToken)  
    listaPoS = []
    for (tok,pos) in tokensPOS:
        listaPoS.append(pos) 
    return listaPoS     


        
        


def main(file1, file2):
    fileInput1 = codecs.open(file1, "r", "utf-8") 
    fileInput2 = codecs.open(file2, "r", "utf-8") 
    raw1 = fileInput1.read()
    raw2 = fileInput2.read()
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #divido i due file in frasi:
    frasi1 = sent_tokenizer.tokenize(raw1)
    frasi2 = sent_tokenizer.tokenize(raw2)
    numtoken1, listaToken1, numFrasi1, mediaFrasi1 = CalcolaLunghezza_MediaFr_Voc(frasi1) #lunghezza corpus in token,frasi e lunghezza media delle frasi
    numtoken2, listaToken2, numFrasi2, mediaFrasi2 = CalcolaLunghezza_MediaFr_Voc(frasi2)
    totToken1 = len(listaToken1) #totale token del corpus
    totToken2 = len(listaToken2)    
    caratteri1 = CalcolaCaratteri(listaToken1) #conteggio caratteri
    caratteri2 = CalcolaCaratteri(listaToken2)
    mediacaratteri1 = caratteri1/totToken1 #lunghezza media delle parole in termini di caratteri
    mediacaratteri2 = caratteri2/totToken2        
    Vocabolario1, Hapax1 = calcolaVoc_Hapax(listaToken1, numtoken1) #Vocabolario e distrubuzione hapax per porzioni incrementali di 1000 token
    Vocabolario2, Hapax2 = calcolaVoc_Hapax(listaToken2, numtoken2)    
    Rapporto1 = RapportoSostantiviVerbi(listaToken1) #rapporto tra sostantivi e verbi presenti nel corpus
    Rapporto2 = RapportoSostantiviVerbi(listaToken2)
    PoS1 = Part_of_Spech(listaToken1) #10 PoS più frequenti
    PoS2 = Part_of_Spech(listaToken2)   
    listaPOS1 = estraiPOS(listaToken1) #estrae solo PoS tag senza token
    listaPOS2 = estraiPOS(listaToken2)
    bigrammiPOS1 = bigrams(listaPOS1) #estrae coppie (token PoS, token PoS)
    bigrammiPOS2 = bigrams(listaPOS2)   
    probCondizionata1, LMI1  = probCondizionata_LMI(listaPOS1, bigrammiPOS1) #Probabilità condizionata dei 10 bigrammi pos più frequenti
    probCondizionata2, LMI2 = probCondizionata_LMI(listaPOS2, bigrammiPOS2)
    
    

    #Stampo i risultati
    print ("Il file", file1, "contiene", numtoken1, "token.")
    print ("Il file", file2, "contiene", numtoken2, "token.")
    if numtoken1>numtoken2:
        print (file1, "ha più token di", file2)
    elif(numtoken1<numtoken2):
        print (file2, "ha più token di", file1)
    else:
        print ("i due file hanno lo stesso numero di token")
    print ("\n")
    print ("Il file", file1, "contiene", numFrasi1, "frasi.")
    print ("Il file", file2, "contiene", numFrasi2, "frasi.") 
    if numFrasi1>numFrasi2:
        print (file1, "ha più frasi di", file2)
    elif(numFrasi1<numFrasi2):
        print (file2, "ha più frasi di", file1)
    else:
        print ("i due file hanno lo stesso numero di frasi")
    print ("\n")
    print ("Il file", file1, "ha una lunghezza media delle frasi di", mediaFrasi1, "token.")
    print ("Il file", file2, "ha una lunghezza media delle frasi di", mediaFrasi2, "token.")
    if mediaFrasi1>mediaFrasi2:
        print ("Il file", file1, "ha una maggiore lunghezza media delle frasi rispetto al file", file2)
    elif(mediaFrasi1<mediaFrasi2):
        print ("Il file", file2, "ha una maggiore lunghezza media delle frasi rispetto al", file1)
    else:
        print ("i due file hanno la stessa lunghezza media delle frasi.")
    print ("\n")
    print ("Il file", file1, "ha una lunghezza media delle parole di", mediacaratteri1, "caratteri.")
    print ("Il file", file2, "ha una lunghezza media delle parole di", mediacaratteri2, "caratteri.")
    if mediacaratteri1>mediacaratteri2:
        print ("Il file", file1, "ha una lunghezza media delle parole maggiore al file", file2)
    elif mediacaratteri1<mediacaratteri2:
        print ("Il file", file2, "ha una lunghezza media delle parole maggiore al file", file1)
    else:
        print ("I file hanno la stessa lunghezza media delle parole in termine di token.")
    print ("\n")  
    print ("\n\n- INCREMENTO DEL VOCABOLARIO OGNI 1000 TOKEN -\n") 
    print (file1, "\t",file2)
    for Type1, Type2 in zip(Vocabolario1, Vocabolario2):
        print (" - %-20s" % (Type1),"- %-20s" % (Type2))  
    print ("\n")  
    print ("\n\n- INCREMENTO DEGLI HAPAX OGNI 1000 TOKEN -\n") 
    print (file1, "\t",file2)
    for Type1, Type2 in zip(Hapax1, Hapax2):
        print (" - %-20s" % (Type1),"- %-20s" % (Type2))  
    print ("\n")  
    print ("Il rapporto tra Sostantivi e Verbi del file", file1, "è", Rapporto1)
    print ("Il rapporto tra Sostantivi e Verbi del file", file2, "è", Rapporto2)
    print ("\n")
    print ("\n\n- 10 PoS PIÙ FREQUENTI -\n") 
    print ("\t", file1, "\t\t\t\t",file2)
    for Type1, Type2 in zip(PoS1, PoS2):
        print (" - %-20s : %-20s" % (Type1[0], Type1[1]),"- %-20s : %-20s" % (Type2[0], Type2[1]))     
    print ("\n")
    print ("\n\n- 10 BIGRAMMI DI PoS CON PROBABILITÀ CONDIZIONATA MASSIMA -\n") 
    print ("\t",file1, "\t\t\t\t",file2)
    for Type1, Type2 in zip(probCondizionata1, probCondizionata2):
        print (" - %-3s %-7s : %-30s" % (Type1[1][0], Type1[1][1], Type1[0]),"- %-3s %-7s : %-20s" % (Type2[1][0], Type2[1][1], Type2[0]))  
    print ("\n\n- 10 BIGRAMMI DI PoS CON FORZA ASSOCIATIVA MASSIMA -\n") 
    print (file1)
    print (LMI1)
    print ("\n")
    print (file2)
    print (LMI2)
    


main(sys.argv[1], sys.argv[2])
