import os
import time
import torch
import sentencepiece as spm
import torch.nn.functional as F
import warnings

warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage and will change in the near future.*"
)

torch.set_grad_enabled(False)


def load_optimized_model(model_path, tokenizer_path):
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(tokenizer_path)
    model = torch.jit.load(model_path)
    model.eval()
    print(f"Modello '{os.path.basename(model_path)}' caricato con successo.")
    return model, sp_model


@torch.inference_mode()
def predict_with_model(model, src_sentence, tokenizer, max_len=500, temperature=1.0, top_k=50, stream=False):
    src_indexes = tokenizer.Encode(src_sentence)
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0)
    trg_indexes = [tokenizer.bos_id()]

    # Buffer per il testo decodificato
    decoded_text = ""

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0)
        preds = model(src_tensor, trg_tensor)
        logits = preds[:, -1, :] / temperature

        # Implementazione del top-k sampling
        if top_k > 0:
            # Ottieni i top_k valori e indici più alti
            top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))

            # Crea una distribuzione di probabilità solo per i top_k token
            top_k_probs = F.softmax(top_k_values, dim=-1)

            # Campiona dai top_k token
            idx = torch.multinomial(top_k_probs, 1)
            pred_token = top_k_indices[0][idx[0]].item()
        else:
            # Comportamento originale se top_k è disabilitato (top_k=0)
            probs = F.softmax(logits, dim=-1)
            pred_token = torch.multinomial(probs, 1).item()

        trg_indexes.append(pred_token)

        if stream:
            # Decodifica il testo corrente
            new_text = tokenizer.Decode(trg_indexes[1:])  # Exclude BOS token

            # Ottieni solo i caratteri nuovi
            if len(new_text) > len(decoded_text):
                diff = new_text[len(decoded_text):]
                decoded_text = new_text
                yield diff

        if pred_token == tokenizer.eos_id():
            break

    if not stream:
        return trg_indexes


def load_and_predict(model_path, tokenizer_path):
    model, tokenizer = load_optimized_model(model_path, tokenizer_path)

    while True:
        src_sentence = input("Inserisci la frase sorgente (o 'quit' per uscire): ")
        if src_sentence.lower() == 'quit':
            break
        try:
            temperature = float(input("Temperatura (es. 1.0): "))
        except ValueError:
            temperature = 1.0

        use_stream = input("Usare streaming? (s/n): ").lower() == 's'

        start_time = time.time()

        if use_stream:
            print("Output: ", end="", flush=True)
            for text_chunk in predict_with_model(model, src_sentence, tokenizer, temperature=temperature, stream=True):
                print(text_chunk, end="", flush=True)
            print()  # Nuova riga alla fine
        else:
            predicted = predict_with_model(model, src_sentence, tokenizer, temperature=temperature)
            print("Output:", tokenizer.Decode(predicted))

        elapsed = time.time() - start_time
        print(f"Inferenza: {elapsed:.2f} secondi")


if __name__ == "__main__":
    MODEL_CHOICES = {
        "discussion": "discussion.pt",
        "multitask": "multitask.pt"
    }

    print("Scegli il modello:")
    for key in MODEL_CHOICES:
        print(f"- {key}")
    choice = input("Modello: ").strip().lower()
    model_path = MODEL_CHOICES.get(choice)

    if not model_path or not os.path.exists(model_path):
        print("Modello non valido o file mancante.")
        exit(1)

    TOKENIZER_PATH = "ai_models/tokenizer/model.model"
    if not os.path.exists(TOKENIZER_PATH):
        print("Tokenizer non trovato.")
        exit(1)

    load_and_predict(model_path, TOKENIZER_PATH)