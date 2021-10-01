#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#
# Generiert N Wortwolken zu N Themenanzahlen
#
# Aktuelle Verwendung = Analyse der Social Media Texte (Facebook und Instagram)
# der verschiedenen Bundestagsparteien in Hessen vor und nach der #BTW21
#
#
# Hessischer Rundfunk
# Autor: Sandra Kiefer
# Datum: 01.10.2021
#
#


import re
import glob
import nltk
import spacy
import gensim
import numpy as np
from os.path import isfile
from os import access, R_OK
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from spacy.lang.de import German


CURRENT_PARTEI = None
PARTEIEN = {'CDU': [], 'SPD': [], 'AfD': [], 'FDP': [], 'DIE LINKE': [], 'GRÜNE': [], 'Volt': [], 'TNT': []}
DIRECTORY = 'D:/Users'


def tokenize(text):
    """
    Tokenisiert einen beliebigen String in deutscher Sprache, zu einzelnen Woertern

    :param text: String             Text der tokenisiert werden soll
    :return:     list [ String ]    liefert eine Liste mit den Tokens
    """
    lda_tokens = []
    parser = German()
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace(): continue
        elif token.like_url: continue
        else: lda_tokens.append(token.lower_)
    return lda_tokens


def get_lemma(word):
    """
    Lemmatisiert ein beliebiges Wort

    :param word: String     Wort dessen Lemma gebraucht wird
    :return:     String     liefert die Lemmatisierung des Wortes
    """
    return ' '.join([x.lemma_ for x in nlp(word)])


def prepare_text_for_lda(text):
    """
    Bereitet einen beliebigen Text für die Verwendung des LDA vor
    (Stoppwörter entfernen, tokenisieren, Lemmatisieren)

    :param text: String             Text der für das LDA aufbereitet werden soll
    :return:     list [ String ]    liefert eine Liste mit den lemmatisierten Tokens
    """
    tokens = [get_lemma(token) for token in tokenize(text)]
    return [token.upper() for token in tokens if len(token) > 3 and token not in de_stopwords]


def changeColor(word, font_size, position, orientation, random_state=None, **kwargs):
    """
    Hilfsfunktion um die Farben der Parteien mit leichter Veraenderung darzustellen
    (Farbintensitaet wird zufaellig ausgewaehlt)

    :return: Farbwert
    """
    colors = {'CDU': ["hsl(0, 0%%, %d%%)", 0, 50], 'SPD': ["hsl(1, 76%%, %d%%)", 50, 100],
              'AfD': ["hsl(215, 51%%, %d%%)", 40, 100], 'FDP': ["hsl(48, 100%%, %d%%)", 70, 100],
              'DIE LINKE': ["hsl(331, 59%%, %d%%)", 60, 100], 'GRÜNE': ["hsl(85, 75%%, %d%%)", 40, 100],
              'TNT': ["hsl(217, 0%%, %d%%)", 25, 70], 'Volt': ["hsl(217, 0%%, %d%%)", 25, 70]}
    return colors[CURRENT_PARTEI][0] % np.random.randint(colors[CURRENT_PARTEI][1], colors[CURRENT_PARTEI][2])


def drawWordcloud(lda, partei, filename):
    """
    Generiert automatisch ein Bild mit den häufigsten Wörter in den Wortthemen
    Das Wortwolkenbild wird automatisch gespeichert

    :param lda:         Latent Dirichlet Allocation
    :param partei:      aktueller Parteiname
    :param filename:    aktueller Dateiname
    :return:            speichert das generierte Bild einer Wortwolke als PNG im Verzeichnis
    """
    for t in range(lda.num_topics):
        plt.figure()
        wc = WordCloud(background_color="white", width=1280, height=720).fit_words(dict(lda.show_topic(t, 10)))
        wc.recolor(color_func=changeColor)
        plt.imshow(wc)
        plt.axis("off")
        plt.title(partei + " Wortwolke #" + str(t + 1), fontsize=20, fontweight='bold', pad=20)
        plt.savefig('%s%d' % (filename, t))
        plt.show()


if __name__ == "__main__":
    # Natural Language Processing vorbereiten (benoetigte Bibliotheken einbinden ...)
    nlp = spacy.load("de_core_news_lg")
    nltk.download('wordnet')
    de_stopwords = nlp.Defaults.stop_words
    de_stopwords.update(['mein', 'klaren', 'degen', 'innen', 'inner', 'https', 'vorab', 'link', 'solch', 'insbesondere',
        'spdbergstrasse', 'ederseestehen', 'bundestag', 'bundestagswahl', 'bundestagsabgeordneter', 'teamherkules',
        'zusammenmachen', 'wegenmorgen', 'guterplan', 'btw21', 'cdukassel', 'erststimmeaufenager', 'werrameissnerkreis',
        'starkeheimat', 'fürunsindenbundestag', 'fürdeutschland', 'guterplan', 'klarelinie', 'lahndillkreis', 'teamirmer',
        'irmerstimme', 'findeneinrichtungen', 'johannesunterwegs', '19uhr', 'stimme', 'erststimme', 'zweitstimme',
        'allesistdrin', 'youtu', 'youtube', 'bereitweilihresseid', 'wahlkampf', 'wahl', 'wählen', 'bundestag',
        'mitherzdabei', 'echtesther', 'bundestagsabgeordnete', 'missionzuversicht', 'sozialpolitikfürdich',
        'erststimmeistfrankestimme', 'bundesmitteln', 'döringwählenunterstützen', 'scholzpacktdasan',
        'landkreislimburgweilburg', 'teamalicia', 'infostand', 'natalie2021', 'spdruedesheim', 'erststimmenadine',
        'rufnachberlin', 'ausrespektvordeinerzukunft', 'driljakristinseewald', 'wahlkreis', 'scholzpacktdasan', 'ausrespekt',
        'heusenstammerschloss', 'spdbergstrasse', 'svenwingerter', 'cdukassel', 'erststimmeaufenager', 'veranstaltung',
        'info', 'thema', 'erststimmecdu', 'stefanheck', 'obertorstrasse', 'wahlkreisabgeordneten', 'btw2021diskussionsrunde',
        'silberbachhalle', 'veranstaltung', 'samstag', 'maintaunusbraun', 'wahlkreis184trebur', 'wahlprogramm',
        'unserestimmeimbundestag', 'kw187team', 'beid', 'gespräch', 'btw2021alisha', 'bereitweilihresseid', 'kandidat',
        'bereiten', 'wahlinfostand', 'moderieren', 'demnächstkanzlerolafscholz', 'bundestagsabgeordneten', 'pdfaktuelle',
        'hauptstadtinfos', 'sozialepolitikfürdich', 'grüne', 'diegrünen', 'hälftedermachtdenfrauen', 'vonbensheimindenbundestag',
        'zudem', 'mdbartol', 'gestern', 'stunden', 'teamirmer', 'sonntag', 'politikchristian', 'stefansauermdb', 'teamcdu',
        'natürlichnordhessen', 'dernächstekanzlerolafscholz', 'döring21', 'erststimmeaufenager', 'michaelaufenager', 'beid',
        'cduidda', 'cdufriedberg', 'wahlkampftour', 'kalender', 'uhrzeit', 'dieschmidt', 'themen', 'bundestagskandidatin',
        'zusammenzukunftgestalten', 'abgeordnet', 'wahlzettel', 'direktmandat', 'bundestagswahl2021', 'eurezukunfteurewahl', '2021'])

    for partei in PARTEIEN:
        CURRENT_PARTEI = partei
        path_list = glob.glob(DIRECTORY + partei + '_*.txt')
        for path in path_list:
            if isfile(path) and access(path, R_OK):
                with open(path, mode='r', encoding="utf8") as file:
                    tokens = prepare_text_for_lda(re.sub(r'[^\w\s]', ' ', file.read()))
                    PARTEIEN[partei] = tokens
            text_data = [d.split() for d in PARTEIEN[partei]]
            dictionary = gensim.corpora.Dictionary(text_data)
            corpus = [dictionary.doc2bow(text) for text in text_data]
            ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=3, id2word=dictionary, passes=100)
            topics = ldamodel.print_topics(num_words=20)
            print('--> ', partei)
            for topic in topics:
                print(topic)        # Konsolenausgabe der aktuellen Ergebnisse
            print()
            drawWordcloud(ldamodel, partei, path.replace(DIRECTORY, '').replace('.txt', ''))
