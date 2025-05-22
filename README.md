# PythonKurs_In2_SKARP

## Pythonprogrammering för AI-utveckling / Egna projektet - del 2
**Skilj på katter och hundar** från foton ur verkliga livet. Bygga upp en modell som som kan skilja på katter och hundar utifrån fotografier. Träningsdata baseras på foton med de enskilda djuren.  

### Förberredelse av data
**File: steg1_CatDog.ipynb** - Använder **jupyter notebook** till förberedelser av bilder
- Samla in data
- Tvätta data
- Dela upp data - data för Träning & Test
    - 7700 foton för  & Evaluering (70/30) intern anpassning
    - 7700 foton för Träning, 3500 kattbilder & 4200 hundbilder
    - 880 foton för Test av framtagen modell, 400 bilder på katter och 480 på hundar
    - Har tidigare noterat det som enklare att anpassa modell mot kattbilder, därför fler hundbilder  
- Sparar det tvättade och uppdelade datat på disk:
    - Träningsdata; bildinformation som X_train, klassinformation som y_train
    - Testdata sparas motsvarande som X_test och y_test
#### import av python moduler
    - import numpy as np                   # för array matrix operationer
    - import matplotlib.pyplot as plt      # för att visualisera data
    - import os                            # för att iterera genom bibliotek med bilder
    - import cv2             # för att uföra image operationer
    - import random          # för att placera bilder i slumpmässig kö inför modellanpassning
    - import pickle          # för att flusha ut dataström till disk, att representera in-/utdata


### Fullskaleförsök bygga modell på hel grupp av träningsdata
**File: steg2-testa-fooo.ipynb**
Den här filen används endast som förberedelse inför nästa steg, för att verifiera att Datat är kompatibelt med CNN modell. Att Datat fungerar att läsa från disk och är användbart för modellanpassning.
- Läser in träningsdata från fil på disk
- Normerar bildernas gråskalenivå på heltalsvärde 0->255 till flyttalsvärde 0->1
- Modellbygge:
    - Sequential() modell
    - Använder Conv2D() NN lager
    - aktiverar med "relu" - för den likriktade linjära enheten
    - Reducerar spatial dimension med MaxPooling2D()
    - Avslutningsvis används en sigmoid funktion att aktivera utgångsdatat
- Sammansättning av modellen inför träning med model.compile(...):
    - loss="binary_crossentropy",        # Konfigurerar modellen inför träning
    - optimizer="adam"
    - metrics=['accuracy']
- Modellanpassning med funktionen model.fit(X_train, y_train, batch_size=32, epochs=10, validation_split=0.3)
- 
#### import av python moduler
    - import pickle
    - import tensorflow as tf
    - from tensorflow.keras.datasets import cifar10
    - from tensorflow.keras.preprocessing.image import ImageDataGenerator
    - from tensorflow.keras.models import Sequential
    - from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


