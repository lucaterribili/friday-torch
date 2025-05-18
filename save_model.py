import sentencepiece as spm
import torch
import sys
from friday_transformer import FridayTransformer

# === Input da terminale ===
model_choice = input("Scegli il modello da convertire ('multitask' o 'discussion'): ").strip().lower()
if model_choice not in ['multitask', 'discussion']:
    print("Scelta non valida. Deve essere 'multitask' o 'discussion'.")
    sys.exit(1)

try:
    num_layers = int(input("Inserisci il numero di layer (es. 2, 4, 6): ").strip())
except ValueError:
    print("Valore non valido per num_layers.")
    sys.exit(1)

# === Configurazione ===
tokenizer_path = 'ai_models/tokenizer/model.model'
checkpoint_file = f"{model_choice}.ckpt"
output_file = f"{model_choice}.pt"

sp_model = spm.SentencePieceProcessor()
sp_model.Load(tokenizer_path)

embed_dim = 512
vocab_size = sp_model.vocab_size()
n_heads = 8

model = FridayTransformer(
    embed_dim=embed_dim,
    src_vocab_size=vocab_size,
    target_vocab_size=vocab_size,
    num_layers=num_layers,
    n_heads=n_heads
)

# === Caricamento pesi ===
try:
    with torch.no_grad():
        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
        state_dict = {k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items() if k.startswith("model.")}
        model.load_state_dict(state_dict)
except FileNotFoundError:
    print(f"Errore: File di checkpoint '{checkpoint_file}' non trovato.")
    sys.exit(1)
except Exception as e:
    print(f"Errore durante il caricamento del checkpoint: {e}")
    sys.exit(1)

# === TorchScript ===
scripted_model = torch.jit.script(model)
optimized_model = torch.jit.optimize_for_inference(scripted_model)
torch.jit.save(optimized_model, output_file)

print(f"Modello TorchScript salvato come '{output_file}'")
