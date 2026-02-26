# inspirations

https://github.com/ml-explore/mlx-lm

https://github.com/simonw/llm



# Usage

_First, `pip install -e .` to get the app project scripts_

Always required is a database, so that's always the first argument.

```bash
rag db "What is AFM?"
rag db "What is AFM?"

rag db "What is AFM?" --llm mistral

rag db "What is AFM?" --table-name embeddings
rag db "What is AFM?" --max_tokens 500
rag db "What is AFM?" --top-k 5

rag db "What is AFM?" --llm mistral --top-k 10 --max_tokens 500 

rag db.sqlite "What is AFM"?
rag db.sqlite "What are the key features of AFM?"

rag s3:/lance-db "What is AFM"?
rag s3:/lance-db "What is AFM"? --llm mistral

rag s3:/lance-db/embeddings "What are the key features of AFM, and what are the responsible AI principles?:
```


To load or add files

```{r}
rag.load db context
rag.load db afm.txt

rag.load db context --skip-extensions png jpg

rag.load db afm.txt --chunk basic
rag.load db afm.txt --chunk unstructured

rag.load db afm.txt --chunk-size 500
rag.load db afm.txt --chunk-overlap 100

rag.load db afm.txt --embed voyageai
rag.load db afm.txt --embed gemini

## consolidate --db with db name? 
rag.load db afm.txt --db postgres
rag.load db afm.txt --db sqlite
rag.load db afm.txt --db lancedb
rag.load db afm.txt --db google-store
```

Implement:

  * Notify if creating a db, or using existing one
  * parse --llm for mistral or claude or gpt and use it to select an llm provider


# another concept

```bash
rag -i context -o ./db -q "What is AFM?"
```


# Prev

To load or add file:

```{r}
rag -db ./db -d context
rag -db ./db -d afm.txt

rag -db ./db -d context --skip-extensions png jpg

rag -db ./db -d afm.txt --chunk basic
rag -db ./db -d afm.txt --chunk unstructured

rag -db ./db -d afm.txt --chunk-size 500
rag -db ./db -d afm.txt --chunk-overlap 100

rag -db ./db -d afm.txt --embed voyageai
rag -db ./db -d afm.txt --embed gemini

rag -db ./db -d afm.txt --db postgres
rag -db ./db -d afm.txt --db sqlite
rag -db ./db -d afm.txt --db lancedb
rag -db ./db -d afm.txt --db google-store
```



# OLD

Basic usage:
```
python rag.py -q "What is AFM?" --doc context/afm_short.txt
python rag.py -q "What is AFM?" --dir context

```

Specify providers:

```
python rag.py -q "What is AFM?" --dir context --embeddings voyage-3.5-lite
python rag.py -q "What is AFM?" --dir context --embeddings voyage-3.5-lite --llm mistral-large-2512
```

Specify where to put embeddigns DB:

```
python rag.py -q "What is AFM?" --dir context --embeddings voyage-3.5-lite --llm mistral-large-2512 --db ./my_db
```

Only create embeddings DB by skipping `-q`

```
python rag.py --dir context --db ./my_db
```


Use exiting DB and just run query by omitting both --doc and --dir

```
python rag.py -q "What is AFM?" --db ./my_db
```
