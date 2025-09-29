**Ecco una suite di casi di test strutturata per categorie, pensata per valutare il modello su un'ampia gamma di scenari, dalle citazioni standard agli edge case più complessi. Ogni caso di test include un input e l'output atteso (le entità che il modello dovrebbe idealmente riconoscere).**

### Struttura della Suite di Test

**I test sono suddivisi nelle seguenti categorie:**

* **T1 - Riconoscimento di Base**: Valuta l'identificazione di citazioni normative comuni e ben formattate.
* **T2 - Variazioni di Formattazione e Stile**: Testa la robustezza del modello a diversi stili di scrittura, abbreviazioni e formati non canonici.
* **T3 - Complessità e Combinazioni**: Verifica la capacità di gestire citazioni multiple, rinvii complessi e nidificati all'interno della stessa frase.
* **T4 - Contesto Esteso e Riferimenti a Distanza**: Misura la performance su testi lunghi dove le entità sono separate da molto testo non rilevante.
* **T5 - Edge Cases e Falsi Positivi**: Spinge il modello ai suoi limiti con testi ambigui, errori di battitura e frasi che assomigliano a citazioni ma non lo sono.
* **T6 - Riferimenti Interni e Relativi**: Valuta la capacità (se addestrato per farlo) di riconoscere rinvii generici come "il comma precedente".

---

### T1: Riconoscimento di Base (Baseline)

 **Obiettivo**: Verificare che il modello riconosca correttamente le forme più comuni e standard di citazioni normative e giurisprudenziali.

| **ID Test** | **Descrizione**            | **Testo di Input**                                                                             | **Entità Attese (**NORMATIVA**,**GIURISPRUDENZA**)** |
| ----------------- | -------------------------------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| **T1.1**    | **Legge con data**         | **Si applica la Legge 7 agosto 1990, n. 241 in materia di procedimento amministrativo.**       | **Legge 7 agosto 1990, n. 241**                                   |
| **T1.2**    | **Decreto Legislativo**    | **Il rapporto di lavoro è disciplinato dal Decreto Legislativo 30 marzo 2001, n. 165.**       | **Decreto Legislativo 30 marzo 2001, n. 165**                     |
| **T1.3**    | **Articolo di Codice**     | **La responsabilità è definita dall'art. 2043 del Codice Civile.**                           | **art. 2043 del Codice Civile**                                   |
| **T1.4**    | **Regolamento UE**         | **Il trattamento dei dati personali è conforme al Regolamento (UE) 2016/679.**                | **Regolamento (UE) 2016/679**                                     |
| **T1.5**    | **Sentenza di Cassazione** | **Come stabilito da Cass. civ., Sez. Un., 22/12/2015, n. 25723, il principio è consolidato.** | **Cass. civ., Sez. Un., 22/12/2015, n. 25723**                    |
| **T1.6**    | **Decreto Ministeriale**   | **Le modalità sono indicate nel D.M. 1 aprile 2021 del Ministero della Salute.**              | **D.M. 1 aprile 2021 del Ministero della Salute**                 |

---

### T2: Variazioni di Formattazione e Stile

 **Obiettivo**: Testare la robustezza del modello rispetto alle diverse forme in cui una stessa norma può essere citata.

| **ID Test** | **Descrizione**               | **Testo di Input**                                                                                  | **Entità Attese (**NORMATIVA**)**     |
| ----------------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| **T2.1**    | **Abbreviazioni Comuni**      | **Si fa riferimento al D.Lgs. 165/2001 e alla L. 241/90.**                                          | **D.Lgs. 165/2001**,**L. 241/90**      |
| **T2.2**    | **Misto Maiuscole/Minuscole** | **La norma è l'articolo 18 della legge n. 300 del 1970.**                                          | **articolo 18 della legge n. 300 del 1970**  |
| **T2.3**    | **Ordine Invertito**          | **La disciplina si trova nella legge 7 agosto 1990, numero 241.**                                   | **legge 7 agosto 1990, numero 241**          |
| **T2.4**    | **Senza tipo di atto**        | **Il riferimento è al 165/2001, che ha riformato il pubblico impiego.**                            | **165/2001**(più difficile, ma auspicabile) |
| **T2.5**    | **Con "recante"**             | **Si applica il decreto-legge 19 maggio 2020, n. 34, recante misure urgenti in materia di salute.** | **decreto-legge 19 maggio 2020, n. 34**      |
| **T2.6**    | **Abbreviazione GDPR**        | **La privacy è tutelata dal GDPR (Regolamento UE 679/2016).**                                      | **GDPR**,**Regolamento UE 679/2016**   |

---

### T3: Complessità e Combinazioni

 **Obiettivo**: Valutare la capacità di districarsi in frasi complesse con citazioni multiple, rinvii interni e nidificazioni.

| **ID Test** | **Descrizione**                          | **Testo di Input**                                                                                                | **Entità Attese (**NORMATIVA**)**                             |
| ----------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **T3.1**    | **Rinvio nidificato**                    | **La procedura è descritta nell'art. 12, comma 2, lettera b), del D.Lgs. n. 50/2016.**                           | **art. 12, comma 2, lettera b), del D.Lgs. n. 50/2016**              |
| **T3.2**    | **Elenco di articoli**                   | **Vengono in rilievo gli articoli 1341, 1342 e 1469-bis del codice civile.**                                      | **articoli 1341, 1342 e 1469-bis del codice civile**                 |
| **T3.3**    | **Norme separate da congiunzione**       | **Il ricorso è ammissibile ai sensi della Legge 241/1990 e del Codice del Processo Amministrativo.**             | **Legge 241/1990**,**Codice del Processo Amministrativo**      |
| **T3.4**    | **Riferimenti multipli e complessi**     | **In base all'art. 3 della L. n. 241/1990, e in deroga all'art. 5 del D.P.R. 380/2001, si dispone quanto segue.** | **art. 3 della L. n. 241/1990**,**art. 5 del D.P.R. 380/2001** |
| **T3.5**    | **Articoli multipli dello stesso testo** | **Si applicano gli artt. 2 e 5 della medesima legge n. 300/1970.**                                                | **artt. 2 e 5 della medesima legge n. 300/1970**                     |

---

### T4: Contesto Esteso e Riferimenti a Distanza

 **Obiettivo**: Verificare che il modello non perda il contesto in paragrafi lunghi e sia in grado di identificare entità molto distanti tra loro.

| **ID Test** | **Descrizione**                      | **Testo di Input**                                                                                                                                                                                                                                                                                                                                                                                | **Entità Attese (**NORMATIVA**)**                                                                                      |
| ----------------- | ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **T4.1**    | **Entità distanti**                 | **Conformemente a quanto previsto dal D.Lgs. 18 aprile 2016, n. 50, l'amministrazione ha avviato la procedura di gara, tenendo conto di numerosi fattori economici e sociali che influenzano il mercato di riferimento, e dopo un'attenta valutazione comparativa delle offerte pervenute, ha infine deliberato l'aggiudicazione secondo i criteri stabiliti dall'art. 95 del medesimo decreto.** | **D.Lgs. 18 aprile 2016, n. 50**,**art. 95 del medesimo decreto**                                                       |
| **T4.2**    | **Citazione in apertura e chiusura** | **La Costituzione della Repubblica Italiana stabilisce all'articolo 2 i diritti inviolabili dell'uomo. [INSERIRE QUI 300 PAROLE DI TESTO LEGALE GENERICO]. In conclusione, ogni atto deve essere interpretato alla luce dei principi supremi, come ribadito dalla nota sentenza della Corte Costituzionale n. 269 del 2017.**                                                                     | **Costituzione della Repubblica Italiana**,**articolo 2**,**sentenza della Corte Costituzionale n. 269 del 2017** |

---

### T5: Edge Cases e Falsi Positivi

 **Obiettivo**: Identificare i punti deboli del modello, testando la sua capacità di gestire l'ambiguità e di non classificare erroneamente testo che assomiglia a entità legali.

| **ID Test** | **Descrizione**                          | **Testo di Input**                                                                                 | **Entità Attese**                                                                                                                         |
| ----------------- | ---------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **T5.1**    | **Falso positivo (numeri)**              | **L'ordine del giorno numero 15 del 2023 è stato approvato. Il lotto 300/1970 è stato venduto.** | **Nessuna entità**                                                                                                                        |
| **T5.2**    | **Falso positivo (contesto non legale)** | **Il manuale utente, al capitolo 5, comma 2, descrive la procedura di reset.**                     | **Nessuna entità**                                                                                                                        |
| **T5.3**    | **Errore di battitura (typo)**           | **Si applica l'art. 18 della Legge n. 300/19710, come modificato dal D.Lgs. n. 23/20015.**         | **art. 18 della Legge n. 300/19710**,**D.Lgs. n. 23/20015**(il modello dovrebbe comunque riconoscerle)                               |
| **T5.4**    | **Citazione incompleta/troncata**        | **Il fatto è disciplinato dall'articolo 5 del decreto... come successivamente emendato.**         | **articolo 5 del decreto**(parziale)                                                                                                       |
| **T5.5**    | **Ambiguità (Delibera vs. numero)**     | **La delibera 15 del consiglio comunale ha stabilito i nuovi criteri.**                            | **delibera 15 del consiglio comunale**(potrebbe essere un**ATTO_AMMINISTRATIVO**o non essere un'entità, a seconda delle specifiche) |

---

### T6: Riferimenti Interni e Relativi

 **Obiettivo**: Se il modello è stato addestrato a riconoscere una classe specifica per questi casi (es. **RIFERIMENTO_RELATIVO**), questi test ne verificano l'efficacia. Altrimenti, servono a confermare che **non** vengano erroneamente classificati come **NORMATIVA**.

| **ID Test** | **Descrizione**                     | **Testo di Input**                                                          | **Entità Attese (**RIFERIMENTO_RELATIVO**)**                                                           |
| ----------------- | ----------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| **T6.1**    | **Riferimento generico precedente** | **La procedura è indicata nel precedente articolo.**                       | **precedente articolo**                                                                                       |
| **T6.2**    | **Riferimento a comma**             | **Come specificato al comma 3 del presente decreto.**                       | **comma 3 del presente decreto**                                                                              |
| **T6.3**    | **Riferimento a "medesimo"**        | **I criteri sono quelli indicati all'art. 95 del medesimo D.Lgs. 50/2016.** | **art. 95 del medesimo D.Lgs. 50/2016**(questa è una**NORMATIVA**che contiene un riferimento relativo) |
| **T6.4**    | **Riferimento a "ultimo comma"**    | **L'eccezione prevista dall'ultimo comma non si applica in questo caso.**   | **ultimo comma**                                                                                              |

### Come Utilizzare questa Suite di Test

* **Creare un Golden Dataset**: Trasforma questi esempi in un formato strutturato (es. JSON o CSV) che contenga il testo e le posizioni esatte (start/end char) di ogni entità attesa. Questo diventerà il tuo "golden set" di valutazione.
* **Test Automatizzati**: Integra questo dataset in una pipeline di test automatici. Dopo ogni addestramento o modifica al codice di inferenza, esegui il modello sul golden set.
* **Calcolare le Metriche**: Valuta le performance calcolando **Precision**, **Recall** e **F1-score** a livello di entità. Questo ti darà una misura oggettiva della qualità del modello.
* **Analisi degli Errori**: Analizza manualmente i casi in cui il modello fallisce.

  * **Falsi Positivi (FP)**: Entità estratte dal modello ma non presenti nel golden set.
* **Falsi Negativi (FN)**: Entità presenti nel golden set ma non trovate dal modello.
* **Classificazione Errata**: Entità trovata correttamente ma con il tipo sbagliato.
* **Iterare**: Usa l'analisi degli errori per guidare i tuoi prossimi passi. Potresti aver bisogno di:

  * **Aggiungere più dati di training simili ai casi falliti.**
* **Regolare i pesi nell'ensemble.**
* **Migliorare la logica del **SemanticValidator**.**
* **Correggere bug nel pre-processing del testo.**
