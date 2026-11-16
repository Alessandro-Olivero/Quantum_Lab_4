"""Esercizio 1: Max-Cut pesato mediante QUBO e simulated annealing.

Installazione: python -m pip install -r requirements.txt
Esecuzione:   python Es_01.py [percorso/max_cut.txt]

Per ogni nodo i, x_i = 0 indica S1 e x_i = 1 indica S2.
Il peso del taglio e' C(x) = sum_(i,j) w_ij (x_i + x_j - 2*x_i*x_j).
Si minimizza quindi f(x) = -C(x), senza vincoli o variabili ausiliarie.
Nella convenzione triangolare f(x) = sum_(i<=j) Q_ij*x_i*x_j:
Q_ii = -sum_j w_ij e Q_ij = 2*w_ij per i < j.

Infatti x_i + x_j - 2*x_i*x_j vale 1 solo se i due bit sono diversi:
    x_i  x_j   contributo al taglio   contributo a f(x)
     0    0             0                    0
     0    1            w_ij                -w_ij
     1    0            w_ij                -w_ij
     1    1             0                    0
Il minimo dell'energia coincide quindi con il massimo peso del taglio.
La formula per Q_ii considera gli archi incidenti senza autoarchi; un
autoarco contribuisce zero al taglio e si annulla nella formula simbolica.
"""

# argparse gestisce gli argomenti passati al programma dal terminale.
import argparse
# product genera tutte le combinazioni di bit per la verifica esaustiva.
from itertools import product
# isclose confronta numeri reali con tolleranza; isfinite esclude NaN e infinito.
from math import isclose, isfinite
# Path permette di costruire e leggere percorsi senza concatenare stringhe.
from pathlib import Path

# QUBO rappresenta un polinomio quadratico in variabili binarie.
# boolean_var crea una variabile simbolica che puo' assumere solo 0 oppure 1.
from qubovert import QUBO, boolean_var
# Il solutore cerca un minimo del QUBO tramite simulated annealing classico.
from qubovert.sim import anneal_qubo


# Il percorso parte dalla posizione dello script: funziona anche se lo script
# viene avviato da una cartella di lavoro diversa da quella del laboratorio.
DEFAULT_INPUT = Path(__file__).resolve().parent / "Lab4_input_files/Lab4/max_cut.txt"


def read_graph(path):
    """Legge n e gli archi (i, j, peso), con nodi numerati da 0 a n-1.

    La prima riga contiene n; ciascuna riga successiva contiene due indici
    e un peso. Restituisce (n, edges), dove edges e' una lista di tuple.
    Gli archi sono non orientati e ogni riga fornisce un contributo al costo:
    eventuali righe duplicate vengono quindi sommate, non eliminate.
    """
    # Legge il file intero e separa le righe; gli errori di accesso al file
    # vengono lasciati a Python, che mostra il percorso coinvolto.
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    # Rimuove spazi alle estremita' e ignora le righe vuote.
    lines = [line.strip() for line in lines if line.strip()]
    if not lines:
        raise ValueError("Il file del grafo e' vuoto.")
    n = int(lines[0])  # Numero complessivo di nodi, inclusi quelli isolati.
    if n < 1:
        raise ValueError("Il numero di nodi deve essere positivo.")
    edges = []
    # start=2 conta le righe dei dati filtrati a partire dalla prima dopo n.
    for line_number, line in enumerate(lines[1:], start=2):
        # split() accetta anche tabulazioni o spazi multipli come separatori.
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"Riga {line_number}: attesi nodo1 nodo2 peso.")
        i, j = map(int, fields[:2])
        # float consente di usare anche grafi con pesi non interi.
        weight = float(fields[2])
        # Gli indici devono appartenere al grafo e i pesi devono essere finiti.
        if not (0 <= i < n and 0 <= j < n) or not isfinite(weight):
            raise ValueError(f"Riga {line_number}: nodo o peso non valido.")
        edges.append((i, j, weight))
    return n, edges


def build_qubo(n, edges):
    """Costruisce simbolicamente il QUBO per un grafo pesato generico."""
    # L'etichetta intera i identifica la variabile del nodo i anche negli
    # stati restituiti dal solutore. x e' una lista di espressioni simboliche,
    # non una lista contenente gia' una soluzione numerica.
    x = [boolean_var(i) for i in range(n)]
    # Inizializza il polinomio nullo a cui aggiungere tutti i contributi.
    objective = QUBO()
    for i, j, weight in edges:
        # Un arco tagliato abbassa l'energia di weight; uno non tagliato
        # contribuisce zero. qubovert raccoglie i coefficienti dei monomi
        # uguali e applica l'identita' binaria x_i**2 = x_i.
        objective += weight * (2 * x[i] * x[j] - x[i] - x[j])
    # Nessuna penalita': qualunque assegnazione binaria definisce un taglio.
    return objective


def cut_weight(state, edges):
    """Calcola direttamente il peso degli archi che attraversano il taglio."""
    # Questo calcolo usa la definizione del problema, indipendentemente dal
    # polinomio QUBO: un arco attraversa il taglio se i bit sono diversi.
    # state puo' essere un dizionario o una sequenza indicizzata per nodo.
    return sum(weight for i, j, weight in edges if state[i] != state[j])


def main():
    # Il percorso e' opzionale (nargs="?"): senza argomenti si usa il file
    # fornito dal laboratorio. --help mostra anche la descrizione iniziale.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    # Ogni anneal e' una nuova ricerca; piu' ricerche aumentano le occasioni
    # di trovare un buon minimo. duration controlla la durata del raffreddamento.
    parser.add_argument("--num-anneals", type=int, default=100)
    parser.add_argument("--duration", type=int, default=1000)
    # Il seed rende riproducibile la ricerca nello stesso ambiente software.
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.num_anneals < 1 or args.duration < 1:
        parser.error("num-anneals e duration devono essere positivi")

    n, edges = read_graph(args.input)
    qubo = build_qubo(n, edges)
    # Un QUBO vuoto non ha coefficienti non nulli: tutte le assegnazioni
    # hanno la stessa energia, quindi non occorre chiamare il solutore.
    if qubo:
        # Il simulated annealing parte da stati casuali e raffredda un sistema
        # fittizio: a temperatura alta puo' accettare mosse che peggiorano
        # l'energia, aiutando a uscire dai minimi locali. Raffreddando,
        # favorisce sempre piu' le configurazioni con energia bassa.
        # E' un metodo euristico: da solo non certifica l'ottimo globale.
        # Si usano intervallo di temperatura e schedule predefiniti di qubovert.
        results = anneal_qubo(
            qubo, num_anneals=args.num_anneals,
            anneal_duration=args.duration, seed=args.seed,
        )
        # Le variabili assenti dal QUBO (ad esempio nodi isolati) sono libere.
        # results.best seleziona il risultato con energia minima tra le prove.
        # get(i, 0) assegna a S1 gli eventuali nodi non presenti nello stato.
        state = {i: results.best.state.get(i, 0) for i in range(n)}
        energy = results.best.value
    else:
        # In assenza di interazioni scegliamo convenzionalmente tutti i bit a 0.
        state = dict.fromkeys(range(n), 0)
        energy = 0

    weight = cut_weight(state, edges)
    # Controllo di coerenza della formulazione, distinto dalla verifica
    # dell'ottimalita': per qualunque soluzione deve valere energia = -taglio.
    # La tolleranza assoluta evita errori dovuti agli arrotondamenti dei float.
    if not isclose(energy, -weight, abs_tol=1e-9):
        raise RuntimeError("Energia QUBO e peso del taglio non coerenti.")

    # Nel QUBO stampato, (i,) indica il coefficiente di x_i e (i, j) quello
    # di x_i*x_j. Le liste S1 e S2 traducono i bit in una partizione del grafo.
    # Il formato :g stampa i valori evitando decimali finali superflui.
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
    # Infatti scambiare S1 e S2 non cambia gli archi tagliati. Bastano quindi
    # 2**(n-1) configurazioni. Il costo resta esponenziale: la soglia evita
    # di applicare questa verifica a grafi troppo grandi.
    if n <= 20:
        optimum = max(
            # bits contiene i valori dei nodi da 1 a n-1; il primo bit e' 0.
            cut_weight((0,) + bits, edges)
            for bits in product((0, 1), repeat=n - 1)
        )
        print(f"Ottimo esatto: {optimum:g}")
        # Scarto nullo significa che SA ha raggiunto il massimo globale.
        print(f"Scarto dall'ottimo: {optimum - weight:g}")
        print("Ottimalita' verificata:", isclose(weight, optimum, abs_tol=1e-9))
    else:
        print("Verifica esaustiva omessa: piu' di 20 nodi.")


# Esegue main solo quando si avvia questo file direttamente; importandolo
# da un altro script si possono riutilizzare le funzioni senza avviare la CLI.
if __name__ == "__main__":
    main()
