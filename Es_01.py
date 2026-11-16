"""Esercizio 1: Max-Cut pesato mediante QUBO e simulated annealing.

Installazione: python -m pip install -r requirements.txt
Esecuzione:   python Es_01.py [percorso/max_cut.txt]

Per ogni nodo i, x_i = 0 indica S1 e x_i = 1 indica S2.
Il peso del taglio e' C(x) = sum_(i,j) w_ij (x_i + x_j - 2*x_i*x_j).
Si minimizza quindi f(x) = -C(x), senza vincoli o variabili ausiliarie.
Nella convenzione triangolare f(x) = sum_(i<=j) Q_ij*x_i*x_j:
Q_ii = -sum_j w_ij e Q_ij = 2*w_ij per i < j.
"""

# Strumenti per leggere gli argomenti del terminale, generare combinazioni
# binarie, confrontare numeri reali e gestire i percorsi dei file.
import argparse
from itertools import product
from math import isclose, isfinite
from pathlib import Path

# QUBO rappresenta il polinomio da minimizzare; boolean_var crea le sue
# variabili simboliche binarie. anneal_qubo esegue il simulated annealing.
from qubovert import QUBO, boolean_var
from qubovert.sim import anneal_qubo


# Costruisce il percorso rispetto alla posizione dello script, cosi' il file
# viene trovato anche avviando il programma da una cartella diversa.
DEFAULT_INPUT = Path(__file__).resolve().parent / "Lab4_input_files/Lab4/max_cut.txt"


def read_graph(path):
    """Legge n e gli archi (i, j, peso), con nodi numerati da 0 a n-1."""
    # Legge il testo e lo divide in righe. La prima contiene il numero di nodi;
    # ogni riga successiva descrive un arco non orientato e il relativo peso.
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    # Ignora righe vuote ed elimina spazi iniziali e finali.
    lines = [line.strip() for line in lines if line.strip()]
    if not lines:
        raise ValueError("Il file del grafo e' vuoto.")
    n = int(lines[0])  # Include anche eventuali nodi senza archi.
    if n < 1:
        raise ValueError("Il numero di nodi deve essere positivo.")
    edges = []
    # Il contatore parte da 2 perche' la prima riga filtrata contiene n.
    # Se il file contiene righe vuote, la numerazione si riferisce alle righe
    # rimaste dopo il filtraggio, non alle posizioni originali nel file.
    for line_number, line in enumerate(lines[1:], start=2):
        # split() gestisce sia spazi multipli sia tabulazioni.
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"Riga {line_number}: attesi nodo1 nodo2 peso.")
        i, j = map(int, fields[:2])
        # I nodi sono interi; float permette di usare anche pesi decimali.
        weight = float(fields[2])
        # Rifiuta indici fuori dall'intervallo e pesi NaN o infiniti.
        if not (0 <= i < n and 0 <= j < n) or not isfinite(weight):
            raise ValueError(f"Riga {line_number}: nodo o peso non valido.")
        # Ogni riga contribuisce al costo: archi ripetuti vengono sommati.
        edges.append((i, j, weight))
    return n, edges


def build_qubo(n, edges):
    """Costruisce simbolicamente il QUBO per un grafo pesato generico."""
    # Una variabile per nodo: 0 significa appartenenza a S1, 1 a S2.
    # Queste sono espressioni simboliche, non valori gia' assegnati.
    # Le etichette intere permettono di riconoscere i nodi nella soluzione.
    x = [boolean_var(i) for i in range(n)]
    # Parte dal polinomio nullo e aggiunge il contributo di ciascun arco.
    objective = QUBO()
    for i, j, weight in edges:
        # Tabella di 2*x_i*x_j - x_i - x_j:
        #   (0,0) -> 0; (0,1) -> -1; (1,0) -> -1; (1,1) -> 0.
        # Un arco tagliato abbassa quindi l'energia di weight. Minimizzare
        # questa somma equivale a massimizzare il peso totale del taglio.
        # qubovert raccoglie automaticamente i coefficienti dei termini
        # uguali e usa x_i**2 = x_i. Anche gli autoarchi si annullano.
        objective += weight * (2 * x[i] * x[j] - x[i] - x[j])
    # Non servono penalita': qualsiasi assegnazione binaria e' ammissibile.
    return objective


def cut_weight(state, edges):
    """Calcola direttamente il peso degli archi che attraversano il taglio."""
    # Se i due estremi hanno bit diversi, appartengono a insiemi diversi.
    # Questo calcolo segue la definizione del problema e non usa il QUBO,
    # quindi permette di controllare indipendentemente l'energia ottenuta.
    return sum(weight for i, j, weight in edges if state[i] != state[j])


def main():
    # Crea l'interfaccia da terminale; --help mostra la descrizione iniziale.
    parser = argparse.ArgumentParser(description=__doc__)
    # nargs="?" rende il percorso facoltativo: senza argomenti usa DEFAULT_INPUT.
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    # Numero di ricerche indipendenti e durata del raffreddamento per ricerca.
    # Aumentarli puo' migliorare il risultato, ma richiede piu' tempo.
    parser.add_argument("--num-anneals", type=int, default=100)
    parser.add_argument("--duration", type=int, default=1000)
    # Il seed controlla la casualita' e favorisce la riproducibilita' nello
    # stesso ambiente software; non costituisce una garanzia di ottimalita'.
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_anneals < 1 or args.duration < 1:
        parser.error("num-anneals e duration devono essere positivi")

    # Prima legge il grafo, poi traduce il problema nel polinomio quadratico.
    n, edges = read_graph(args.input)
    qubo = build_qubo(n, edges)
    if qubo:
        # Simulated annealing: a temperatura alta possono essere accettate
        # anche mosse che aumentano l'energia, per uscire dai minimi locali.
        # Raffreddando si favoriscono progressivamente stati a energia bassa.
        # Si usano temperatura e programma di raffreddamento predefiniti.
        # Si tratta di un metodo euristico classico: il miglior stato trovato
        # potrebbe non essere l'ottimo globale.
        results = anneal_qubo(
            qubo, num_anneals=args.num_anneals,
            anneal_duration=args.duration, seed=args.seed,
        )
        # Le variabili assenti dal QUBO (ad esempio nodi isolati) sono libere.
        # best seleziona la prova con energia minima. get(i, 0) mette in S1
        # i nodi non restituiti dal solutore, senza alterare il costo.
        state = {i: results.best.state.get(i, 0) for i in range(n)}
        energy = results.best.value
    else:
        # Se il polinomio e' nullo, tutte le configurazioni hanno energia 0.
        # Non occorre ottimizzare: si assegnano convenzionalmente tutti a S1.
        state = dict.fromkeys(range(n), 0)
        energy = 0

    weight = cut_weight(state, edges)
    # Deve valere f(x) = -C(x) per qualsiasi stato, anche non ottimo.
    # isclose tollera piccoli errori numerici nei calcoli con pesi decimali.
    if not isclose(energy, -weight, abs_tol=1e-9):
        raise RuntimeError("Energia QUBO e peso del taglio non coerenti.")

    # Nella stampa del QUBO, (i,) identifica il coefficiente lineare di x_i,
    # mentre (i, j) identifica il coefficiente del prodotto x_i*x_j.
    # Le due liste di nodi traducono la soluzione binaria in S1 e S2.
    # :g stampa i numeri senza aggiungere decimali finali non necessari.
    print(f"Grafo: {n} nodi, {len(edges)} archi")
    print(f"QUBO: {qubo}")
    print(f"Soluzione simulated annealing (seed={args.seed}): {state}")
    print("S1:", [i for i in range(n) if state[i] == 0])
    print("S2:", [i for i in range(n) if state[i] == 1])
    print(f"Energia QUBO: {energy:g}")
    print(f"Peso del taglio: {weight:g}")
    print("Archi del taglio:", [(i, j, w) for i, j, w in edges if state[i] != state[j]])

    # Verifica esatta solo per piccole istanze; il solutore principale resta SA.
    # Fissare x_0=0 elimina la simmetria tra un taglio e il suo complementare.
    # Scambiare S1 e S2 lascia invariato il peso: bastano 2**(n-1) stati.
    # Il costo della ricerca esaustiva e' esponenziale, per questo viene
    # limitata ai grafi con al massimo 20 nodi.
    if n <= 20:
        optimum = max(
            # product genera tutti i bit dei nodi 1,...,n-1; (0,) antepone
            # il valore fissato del nodo 0. max sceglie il taglio migliore.
            cut_weight((0,) + bits, edges)
            for bits in product((0, 1), repeat=n - 1)
        )
        print(f"Ottimo esatto: {optimum:g}")
        # Lo scarto misura quanto SA si discosta dall'ottimo certificato.
        # Se e' zero (entro tolleranza), SA ha trovato una soluzione ottima.
        print(f"Scarto dall'ottimo: {optimum - weight:g}")
        print("Ottimalita' verificata:", isclose(weight, optimum, abs_tol=1e-9))
    else:
        print("Verifica esaustiva omessa: piu' di 20 nodi.")


# Avvia il programma solo se il file e' eseguito direttamente. Importandolo
# da un altro modulo si possono riutilizzare le funzioni senza eseguire main.
if __name__ == "__main__":
    main()
