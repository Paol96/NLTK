# Progetto di Linguistica Computazionale AA 2019-20.

# Obiettivo:
Realizzazione di un programma scritto in Python che utilizzi i moduli presenti in Natural
Language Toolkit per leggere due file di testo in inglese, annotarli linguisticamente, confrontarli
sulla base degli indici statistici richiesti ed estrarne le informazioni richieste.

# Fasi realizzative:
Scegliete e scaricate in formato di testo semplice utf-8 due libri a scelta tra quelli che trovate nel
sito del Progetto Gutenberg (http://www.gutenberg.org/ebooks/).
Sviluppate due programmi che prendono in input i due file da riga di comando, che li analizzano
linguisticamente fino al Part-of-Speech tagging e che eseguono le operazioni richieste.

Confrontate i due testi sulla base delle seguenti informazioni statistiche: <br>
 il numero totale di frasi e di token; <br>
 la lunghezza media delle frasi in termini di token e la lunghezza media delle parole in
termini di caratteri; <br>
 la grandezza del vocabolario e la distribuzione degli hapax all'aumentare del corpus per
porzioni incrementali di 1000 token (1000 token, 2000 token, 3000 token, etc.); <br>
 il rapporto tra Sostantivi e Verbi; <br>
 le 10 PoS (Part-of-Speech) più frequenti; <br>
 estraete ed ordinate i 10 bigrammi di PoS: <br>
◦ con probabilità condizionata massima, indicando anche la relativa probabilità; <br>
◦ con forza associativa massima (calcolata in termini di Local Mutual Information),
indicando anche la relativa forza associativa.
