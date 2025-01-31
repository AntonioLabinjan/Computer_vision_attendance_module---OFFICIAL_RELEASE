# COMPUTER VISION ATTENDANCE MODULE (name still not definitive)

Fakultet informatike u Puli:
https://fipu.unipu.hr/

- Autor: Antonio Labinjan
- Kolegij: Web-aplikacije
- Mentor: doc.dr.sc. Nikola Tanković

Kratki opis:
Aplikacija za prepoznavanje lica i bilježenje prisutnosti na nastavi (ili u drugim slučajevima) implementirana koristeći Flask, Clip i Faiss. 

Funkcionalnosti:
- Student:
- Za korištenje ove grupe funkcionalnosti nije potrebna autentifikacija

  1) Spremanje face_embeddinga putem live_feeda
  2) Spremanje face_embeddinga putem uploadanja lokalnih slika
  3) Prijava prisutnosti pomoću face scanninga

- Profesor(admin):
  1) auth
  2) definiranje predmeta/kolegija sa nazivom i intervalom u kojem se može prijaviti prisutnost
  3) dodavanje studenata u sustav; svaki student imat će ime i slike prema kojima će ga model prepoznati
  4) primanje email notifikacije svaki put kad student uspješno prijavi prisutnost
  5) pregled i filtriranje podataka o prisutnosti
  6) preuzimanje izvještaja o prisutnosti u csv formatu
  7) dijeljenje izvještaja mailom
  8) brisanje prisutnosti
  9) pregled statistike
  10) pregled popisa svih studenata
  11) pregled vizualizacije podataka kroz 3 (za sad) vrste grafikona
  12) korištenje internog profesorskog announcement foruma (CRUD operacije nad obavijestima)
  13) pregled postotka prisutnosti za pojedinog studenta na određenim predmetima
  14) analiza kasnih dolazaka studenata
  15) pregled official academic kalendara

Prijava:
Preporuča se napraviti vlastiti account s vlastitim mailom, ali ukoliko vam se to ne da, možete koristiti moj:
- username: Antonio
- password: 4uGnsUh9!!!
- email: alabinjan6@gmail.com

YouTube: https://youtu.be/hQDcAjGRHMQ
