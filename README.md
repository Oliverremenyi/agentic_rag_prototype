# Agentic RAG Éves Jelentés Elemző Asszisztens

## 1. Miről szól?

A rendszer 6 valós vállalat (Microsoft, Toshiba, HCA Healthcare, Insperity,
1-800-FLOWERS.COM, MGM Resorts International) éves jelentéseit ismeri, és
ezekről tud kérdésre válaszolni.

**Miért hasznos?** Egy éves jelentés nagyon hosszú és nehezen átnézhető.
A chatbot helyette megkeresi a választ, megmondja, honnan vette, és képes
két vállalat adatait is összehasonlítani ("melyiknek volt magasabb a
bevétele, és mennyivel?").

**Miért "agentic"?** Mert nem csak szöveget keres vissza:
- egyszerű kérdésre egyből válaszol a jelentésekből,
- összetett kérdést szétbont kisebb részekre,
- összehasonlító kérdésnél két számot kér le, és azokon **ténylegesen
  számol** (nem csak találgat).

## 2. Hogyan működik?

A rendszer egy LangGraph-gráf, ami lépésről lépésre dolgozza fel a kérdést:

1. **Eldönti**, hogy a kérdés idevág-e, és hogy egyszerű vagy
   összehasonlító kérdésről van-e szó.
2. **Szétbontja** a kérdést kisebb részekre, ha kell.
3. **Minden részt a megfelelő helyre irányít**: vagy visszakeresi a
   választ a jelentésekből, vagy (összehasonlításnál) meghívja a
   számoló eszközt.
4. **Összeállítja a végső választ**, a forrás megjelölésével.

A felhasználói felületen (Streamlit) minden lépés látható: mit keresett
vissza a rendszer, honnan, és ha számolt, milyen két értéket hasonlított
össze.

**Két eszközt használ:**
- egy keresőt, ami a jelentésekben keres,
- egy számológépet, ami két már megtalált számot hasonlít össze
  (ez nem keres semmit, csak számol).

**Hogyan dolgozza fel az adatot?** A jelentéseket nem csak gépiesen
vágja darabokra, hanem a jelentés saját fejezetfelosztását használja,
hogy egy táblázat ne szakadjon el a hozzá tartozó címtől.

**Milyen modellt használ?** Helyben futó, ingyenes modellt (Ollama,
`llama3.2:3b`), nincs fizetős API. Ennek ára, hogy lassabb és néha
pontatlanabb, mint egy nagy kereskedelmi modell.

## 3. Mennyire jó? (valós mérési eredmények)

Nem csak elméletben lett tesztelve, hanem valódi futtatással.

**Kiértékelés** (16 teszt-kérdés, valós jelentésekből): **7/16 helyes
(44%)** — ez elmarad a 80%-os céltól:

1. A keresés néha rossz vállalatból vagy rossz sorból hoz vissza adatot,
   mert a használt embedding-modell nem elég pontos ezen a korpuszon.
2. A kis modell néha a saját tudásából válaszol a jelentés helyett
   (pl. "Mi Franciaország fővárosa?" kérdésre válaszolt, ahelyett hogy
   jelezte volna: ez nem idevágó kérdés).
3. Néha egy egyszerű kérdést is összehasonlításnak néz.
4. Fejlesztés közben egy valódi hibát is találtunk és kijavítottunk: a
   számkinyerő logika néha egy évszámot (pl. 2021) vett figyelembe a
   tényleges pénzösszeg helyett. Ezt javítottuk, és teszttel is
   lefedtük.

A lényeg: amikor a rendszer nem tudta a választ, **mindig ezt mondta**,
sosem talált ki adatot.

**Terheléses teszt** (50 valós kérdés): egy válasz átlagosan kb.
3,7 másodpercig tart, a leghosszabb kb. 6 másodpercig. A legnagyobb
lassító tényező egyértelműen a helyi modell válaszideje, nem a keresés.

Két ötlet a gyorsításra:
1. Az ismétlődő döntéseket (pl. "ez keresés vagy számolás legyen")
   lehetne cache-elni, hogy ne kelljen mindig újra megkérdezni a modellt.
2. A rövid, egyszerű döntésekhez (pl. "keresés vagy számolás?") elég
   lenne egy kisebb, gyorsabb modell is.

## 4. Hogyan indítsd el?

Csak Docker kell hozzá. Az adatok már benne vannak a projektben, nincs
szükség se Kaggle-fiókra, se előkészítésre.

```bash
docker compose up --build
# Felület: http://localhost:8501
```

Ha más vállalatokat szeretnél (nem a már beállított 6-ot), ahhoz kell egy
saját Kaggle API-token:

```bash
uv add kagglehub
uv run python scripts/prepare_dataset.py
```

Docker nélkül, helyben futtatva (Ollama-nak helyben kell futnia):

```bash
uv sync
uv run streamlit run src/ui/streamlit_app.py

# Kiértékelés futtatása
uv run python eval/run_eval.py

# Terheléses teszt futtatása
uv run python loadtest/run_loadtest.py --num-queries 100

# Tesztek futtatása (nem igényel modellt)
uv run pytest
```

## 5. Megjegyzés az adatokról

A jelentések (`data/docs/`, `data/toc/`) be vannak téve a git repóba,
hogy bárki egyből ki tudja próbálni a rendszert. Csak a generált keresési
index (`data/index/`) nincs commitolva, mert az automatikusan újraépül
minden indításkor.
