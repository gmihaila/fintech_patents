python src/fintech_patents/downloads_models.py

python src/fintech_patents/pickle_models.py

mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml