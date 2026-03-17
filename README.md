# Uputstvo za upotrebu

Potrebno instalirati [Python] (https://www.python.org/) programski jezik.

Potrebne biblioteke:
- [numpy](https://pypi.org/project/numpy/)
- [OpenCV](https://pypi.org/project/opencv-python/)

Instalacija biblioteka:
```sh
pip install numpy
pip install opencv-python
```

Skripte
---

**1_data_creation.py**

Uživo prikaz sa kamere, pritiskom na SPACE trenutni kadar (frame) se čuva u zadatom direktorijumu ***output_dir***. Potrebno je oko 20 kadroma u nekom opštem slučaju.

**2_intrinsic_calibration.py**

Učivata sve PNG i JPG fotografije iz zadatog direktorijuma, na osnovu prosleđenih parametara table za kalibraciju generiše idealne koordinate tačaka od interesa na tabli (raskršća ivica kvadrata), vrši detekciju tačaka od interesa na svakoj slici iz direktorijuma. Na kraju vrši optimizaciju estimirajući parametre kamere koji minimizuju razliku između projektovanih i idealnih koordinata na šahovskoj tabli.

Funkcija zahteva sledeće ulazne parametre:
- **images_dir** - putanja do direktorijuma gde se nalaze slike iz skripte 1
- **rows** - broj redova na kalibracionoj tabli (ako tabla ima 5 redova kvadratića, ovaj broj je 4)
- **cols** - broj kolona na kalibracionoj tabli (ako tabla ima 10 kolona kvadratića, ovaj broj je 9)
- **cell_size** - fizička veličina kvadrata na tabli u milimetrima
- **output_file** - putanja do izlazne datoteke, npr. ***output/calib/camera_params.npz***  
  
Izlaz iz funkcije za kalibraciju sastoji se od:
- matrice unutrašnjih parametara kamere
- parametri izobličenja: radijalno i tangencijalno
- root means square (**RMS**): prosečna greška reprojekcije, težimo ka 0 u idelanom slučaju
- rezolucija slike

**3_extrincit_calibration.py**

**TODO**

**4_undistord.py**

Učitava prosleđene kalibracione parametre sačuvane u **npz** formatu. Vrši uživo otklanjanje distorzije slike spram parametara dobijenih kalibracijom. 
Pritiskom na taster ***U*** možemo paliti/gasiti otklanjanje distorzije, na ***SPACE*** možemo sačuvati trenutni kadar.
