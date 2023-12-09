
import numpy as np
import matplotlib.pyplot as plt
encryptedText = ""
with open("encrypted.txt", "r") as file:
    encryptedText = file.read()
encryptedText = encryptedText.replace("\n", "")
# encryptedText = "F RZXN JNVINOTFXN AS TRN LTZSS"
letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

matrix = np.loadtxt("pairFreq.dat")
M = {}


for r in zip(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ "), matrix):
    M[r[0]] = dict(zip(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ "), r[1]))


def new_random_mapping() -> dict:
    map = dict(zip(np.random.permutation(letters), letters))
    map[" "] = " "
    return map


def text_after_mapping(mapping: dict, text: str) -> str:
    t = list(text)
    res = ""
    for i in range(len(t)):
        res += mapping[t[i]]
    return res


def log_pi(text: str) -> float:
    result = list()
    for i in range(len(text) - 1):
        x = np.log(M[text[i]][text[i+1]] + 1e-27)
        result.append(x)
    return np.sum(result)


def proposal(original: dict) -> dict:
    r1, r2 = np.random.choice(list(letters), 2, replace=True)
    proposal = original.copy()
    proposal[r1] = original[r2]
    proposal[r2] = original[r1]
    return proposal


def accept(new_map: dict, current_map: dict) -> bool:
    log_P = log_pi(text_after_mapping(current_map, encryptedText))
    log_P_proposed = log_pi(text_after_mapping(new_map, encryptedText))
    ratio = np.exp(log_P_proposed - log_P)
    accept = False

    if ratio > 1.:
        accept = True
    elif (np.random.rand()) < ratio:
        accept = True

    if not accept:
        new_map = current_map

    return accept


initial_mapping = {'A': 'O', 'B': 'Y', 'C': 'Z', 'D': 'V', 'E': 'Q', 'F': 'A', 'G': 'X', 'H': 'W', 'I': 'C', 'J': 'P',
                   'K': 'M', 'L': 'S', 'M': 'K', 'N': 'E', 'O': 'N', 'P': 'B', 'Q': 'I', 'R': 'H', 'S': 'F', 'T': 'T',
                   'U': 'L', 'V': 'R', 'W': 'D', 'X': 'G', 'Y': 'J', 'Z': 'U', ' ': ' '}


def mcmc():
    AccRatio = 0.0
    NumSucc = 0
    log_pis = list()
    mapping = initial_mapping

    thin = 10

    for i in range(11):

        if (i % thin == 0):
            print("=======================")
            print("Iteration :", i)
            print(text_after_mapping(mapping, encryptedText))

        map_proposed = proposal(mapping)
        acc = accept(map_proposed, mapping)

        if acc:
            NumSucc += 1
            mapping = map_proposed
            log_pis.append(
                log_pi(text_after_mapping(map_proposed, encryptedText)))

    plt.plot(log_pis)


mcmc()
