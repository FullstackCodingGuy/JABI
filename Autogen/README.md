
## Conda - Setup

> Conda is an isolated environment to install the dependencies, which will not impact the actual host system's dependencies

## Autogen - Setup

To activate autogen environment in conda, run         
```
conda activate autogen
```

Setup autogen - first time

```
pip install -U autogenstudio
```


Setup open AI key

```
export OPENAI_API_KEY=ollama
```

Start up the studio

```
autogenstudio ui --port 8090 --appdir ./studiodata
```
Navigate to http://127.0.0.1:8090

