# Showmate AI

## Running Locally
```sh
git clone https://github.com/showmatetech/showmate_ai.git # or clone your own fork
cd showmate_ai
python3 -m pip install -r requirements.txt
python3 uvicorn main:app --reload --host 0.0.0.0 --port 7000
```

The app should now be running on [localhost:7000](http://localhost:7000/).