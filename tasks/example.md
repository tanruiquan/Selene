## Lorem Ipsum Sentiment Classification: Ad Deep Intellectum

### Scopo
In hac inceptio, discentes in opere sentimentorum ex recensionibus cinematographicis per structuram retis neuronalis se immittent.

### Objectiva
- Definitio Stratarum: Per determinationem parametrorem pro singulis stratis, strcturam retis neuronalis definire, includens stratas plene connectas et functiones activationis.
- Functiones Activationis: Significationem functionum activationis, ut ReLU, in introductione non-linearitatis ad modello intellegere.
- Activation Softmax ad Classificationem: Usum activationis softmax in strato output ad gradus probabilitatum adipiscendos cognoscere.
- Progressus Praeambulatus: Flumen informationis per stratas retis neuronalis in progressu praeterire intellegere.

### Dati
Dataset constat ex 10,662 recensionibus cinematographicis, quaelibet ut vector feature magnitudine 100 repraesentatae. Etichettae sunt binariae, cum 'negativum' ad 0 et 'positivum' ad 1 referentes. Praprocessatio datorum, includens vectorizationem et mappationem etichettarum, completa est.

### Instructiones Discentium
Step 1: Define Primum Stratum
Incepere a definitione primi strati plene connecti (fc1) cum magnitudine input 100 et magnitudine output 4. Sequitur activationem ReLU (relu1).

Step 2: Define Secundum Stratum
Progredire ad definitionem secundi strati plene connecti (fc2) cum magnitudine input 4 et magnitudine output 3. Applica aliam functionem activationis ReLU (relu2).

Step 3: Define Tertium Stratum
Definire tertium stratum plene connectum (fc3) cum magnitudine input 3 et magnitudine output 3. Introduce aliam functionem activationis ReLU (relu3).

Step 4: Define Stratum Output
Definire stratum output (out) cum magnitudine input 3 et magnitudine output 2, representans predictionem ultimam.

Step 5: Applica Activationem Softmax
Implementa functionem activationis softmax in strato output ad adipiscendos gradus probabilitatum.

Step 6: Implementa Progressum Praeambulatum
In methodo forward, definire quomodo data input per stratas praetereantur ad productionem probabilitatum output. Verificare ut modello inputum accipiat formae (batch_size, feature_size=100) et outputum reddat formae (batch_size, output_size=2).
