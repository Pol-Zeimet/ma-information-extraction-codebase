# Informationsextraktion aus semistrukturierten Daten

## Setup

### SROIE Datensatz

#### Zugriff
Der Datensatz ist auf der [SROIE 2019 Webseite](https://rrc.cvc.uab.es/?ch=13&com=downloads) zum Download verfügbar. Hierfür ist eine Registrierung erforderlich.
Empfohlen wird der Google Drive Ordner zum Zufriff auf die Daten.

#### Download und Speicherort

Die Daten sind in 3 Ordner aufgeteilt, entsprechend den drei Teilaufgaben der SROIE Challenge. 
Für die Aufgabe der Informationsextraction sind folgende Daten interessant:
  * **OCR Daten:** Diese befinden sich in Ordner zur Task 1
  * **Ground Truth für die gesuchten Entitäten:** Befindet sich im Ordner für Task 2
  * **Scans:** Die Scans der Kassenzettel sind in beiden Ordnern vorhanden. 


Diese Struktur können wir größtenteils bestehen lassen. Es empfiehlt sich jedoch, die Scans für die folgenden Vorverarbeitungsschritten in einen eigenen Ordner zu verschieben.

#### Preprocessing/Vorverarbeitung

Die Vorverarbeitung ist in mehrere Schritte aufgetrennt. 
  * ein genereller Vorverarbeitungsschritt
  * eine individuelle Weiterverarbeitung für jede der Modellarten (BERT, LayoutLM, Graphnets und RNN)

##### Generelles Proprocessing
    
Die Methoden zum ersten Vorverarbeitungs Schritt sind in [`preprocess_SROIE.py`](./src/data_preprocessing/preprocess_SROIE.py)  hinterlegt.
Das Preprocessing führt folgende Schritte durch:
  * Zeilenweises auslesen der gegeben OCR Informationen
  * Labeling der Token: Matching der Token zu der entsprechenden Groundtruth
  * Anhängen vorn BILUO Präfixen an die Label
  * Zusammenfassen der gelabelten Token in zwei DataFrames:
    * einem NER Dataframe. Dieses enthält für jedes Dokument zeil Listen: Eine mit den Token und eine mit den entsprechenden Labels.
    * einem "Result" DataFrame mit Position und Label für jedes Token oder jede Zeile. (je nach Wahl der Preprocessing Methode)

**Mögliche Methoden:**
Die Datei ist als Script ausführbar und bietet drei Methoden zum Preprocessing.
  * v1 matched ganze zeilen zu einem Label und gibt jedem token der Zeile die gleiche bounding box.
  * v2 matched ganze Zeilen zu einem Label und teilt gegebene Boundingbox auf einzelne Wörter der Zeile auf
  * v3 versucht, jedes Token einzeln einem Label zuzuordnen und gibt jedem Token seine eigene Bounding Box

v2 funktioniert am Besten und ist Default für Graphen. LayoutLM benötigt v1 für Zeilenweise Positionen und zusätzlich wahlweise v2 oder v3 für Token Informationen.

##### Preprocessing für die Modelldatensätze

###### Graphennetze und RNN
Die Garphennetze und RNN Modelle sind strukturell identisch. Bei RNN wird lediglich die Faltungsschicht übersprungen. Somit benötigen beide Modelle Graphen als Inpit.
Die Methoden zum Erstellen der Graphen finden sich in [`graph_construction.py`](./src/data_preprocessing/graph_construction.py). Auch diese Datei ist als Script ausführbar und benötigt das zuvor erstellte Result DataFrame. (v1, v2 und v3 funktionieren alle)
Ebenfalls benötigt wird ein Modell zum Embedden der Token. Für Word2Vec wir kein Modellpfad benötigt. Für BERT kann der Pfad zu einem nachtrainiertes NETZ oder einfach ein Modellname wie `'bert-base-uncased'` eingegeben werden.

###### BERT
BERT benötigt nur das NER DataFrame. die Methoden finden sich in [`bert_data_construction.py`](./src/data_preprocessing/bert_data_construction.py) und die Datei ist als Script ausführbar. Erstellt werden 3 json Dateien für Training, Testing und Evaluierung. Hierfür muss die gewünschte Aufteilung angegeben werden (zb. `[0.7, 0.2, 0.1]`)
Die Anzahl der Klassen variiert zwischen 5 (Also DATE, MONEY, NAME, ADDR und O), 9 (idem, mit B und I Präfixen) und 13 (mit kompletten BILUO Präfixen).

###### LayoutLM
LayoutLM benötigt 2 Dataframes, einmal ein Labeling mit Bounding Boxen auf Zeilenebene und einemal auf Wortebene (also einmal das DataFrame aus Methode v1 und einmal aus v2 oder v3).
Die Verarbeitung benötigt ebenfalls die Scans der Dokumente. Hierfür empfiehlt es sich, die Scans in einem eigenen Ordner zusammenzufassen. Die Methoden finden sich in [`layoutlm_data_construction.py`](./src/data_preprocessing/layoutlm_data_construction.py).
Die Datei ist NICHT als Script ausführbar. Die Einstiegsmethode wäre `format_and_preprocess`.
Die Anzahl der Klassen variiert zwischen 5 (Also DATE, MONEY, NAME, ADDR und O), 9 (idem, mit B und I Präfixen) und 13 (mit kompletten BILUO Präfixen).
Die Verarbeitung erstellt für jedes Dokument eine eigene JSON Datei mit Informationen für jede Zeile und jedes darin enthaltene Wort. Darüber hinaus werden Text Dokumente generiert die Infiormationen enthalten bezüglich der Position der Token im Bild ihrer Labels etc. 
Diese Schritte sind Boilerplate und werden benötigt, um einen für LayoutLM geeigneten Datensatz zu erstellen. Hier besteht noch Aufräumbedarf.

### Pro Publica- free the Files Datensatz

Der Free the files Datensatz wurde nur kurz ausgewertet, bevor sich für SROIE entschieden wurde. 
Die Auswertung basiert aus den durch [deepform](https://github.com/project-deepform/deepform) vorgelegten Preprocessing der Daten.
Die Daten liegen im Repository bereits im "data" ordner vor.
Die eigene weiterverarbeitung und kurze Analyse beschränkt sich auf zwei kurze [Jupyter Notebooks](./notebooks/ProPublica/).


## Experiment Scripte

#### Abhängigkeiten
**Zu beachten:** 
  * Experimente für LayoutLM müssen über eine eigene virtuelle Umgebung gestartet werden, da es sonst zu einem Konflikt bei den Abhängigkeiten zu Graphennetzen und BERT kommt.
  * Graphennetze können nur auf GPU trainiert werden, da sie beim Padding auf einen kleinen Trick setzen. Nachlesbar in der Padding Methode der  [DeepMind Graphnets Library](https://github.com/deepmind/graph_nets/blob/64771dff0d74ca8e77b1f1dcd5a7d26634356d61/graph_nets/utils_tf.py#L1456)

#### Ausführung
[Die Experiment Scripte](./scripts/experiment_scripts/) umfassen eine Vielzahl an Experimenten mit den verschiedenen Modellen. Die Experimente trainieren die Netze in unterschiedlichen Konfigurationen. Es werden keine Modelle gespeichert.
Werden alle Standardpfade für die Datenspeicherung beibehalten, können die Experiment Scripte in ihrer gegebenen Form ausgeführt werden.

## Prediction Pipeline

[`predictor.py`](./src/prediction_pipeline/predictor_class/predictor.py) stellt eine `Predictor` Klassen zur Verfügung. mit dieser können Scans von Kassenzetteln ausgewertet werden. Die gefundenen Entitäten werden auf dem gegebenen scan eingezeichnet.
Bisher besteht die Möglichkeit, BERT, LayoutLM, RNN und Graphennetze mit Softmax Output zu verwenden, sofern trainierte Modelle gespeichert wurden.
Modelle mit CRF Output werdendurch die PRedictor Klasse derzeit noch nicht unterstützt.
