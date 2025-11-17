import os
from config.paths import RAW_PATH, PROCESSED_PATH

def create_directories():
    pastas = [
        "data/raw",
        "data/processed",
        "reports/tables",
        "notebooks",
        "src",
        "config",
        "init"
    ]

    for pasta in pastas:
        os.makedirs(pasta, exist_ok=True)
        print(f"✔ Verificado/criado: {pasta}")

def check_raw_data():
    if os.path.exists(RAW_PATH):
        print(f"✔ Arquivo bruto encontrado: {RAW_PATH}")
    else:
        print(f"⚠ Atenção: arquivo bruto NÃO encontrado em {RAW_PATH}")

def init_project():
    print("\n🔧 Inicializando projeto...\n")
    create_directories()
    check_raw_data()
    print("\n🚀 Projeto pronto!\n")

if __name__ == "__main__":
    init_project()

