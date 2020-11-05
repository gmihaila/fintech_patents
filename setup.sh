wget -nc https://www.dropbox.com/s/wu3ofv6u7ehaj01/distilbert-base-uncased.pickle

mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\n\
enableCORS = false\n\
headless = true\n\
\n\
" > ~/.streamlit/config.toml